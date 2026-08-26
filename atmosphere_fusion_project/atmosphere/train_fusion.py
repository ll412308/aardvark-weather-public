from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.losses import latent_coverage_mse, log_density_mse
from atmosphere.models import AtmosphereFusionForecastModel
from atmosphere.utils import (
    amp_dtype_from_name,
    cosine_warmup_lambda,
    make_grad_scaler,
    save_checkpoint,
    seed_everything,
)


def move_batch(batch, device):
    return {
        "time": batch["time"].to(device),
        "latents": {k: v.to(device) for k, v in batch["latents"].items()},
        "densities": {k: v.to(device) for k, v in batch["densities"].items()},
        "available": {k: v.to(device) for k, v in batch["available"].items()},
    }


def apply_instrument_dropout(available, probability):
    if probability <= 0:
        return {k: v.clone() for k, v in available.items()}
    out = {k: v.clone() for k, v in available.items()}
    names = list(out)
    b = next(iter(out.values())).shape[0]
    for bi in range(b):
        present = [n for n in names if bool(out[n][bi])]
        if len(present) <= 1:
            continue
        for name in present:
            if torch.rand((), device=out[name].device) < probability:
                out[name][bi] = False
        if not any(bool(out[name][bi]) for name in present):
            keep = present[int(torch.randint(len(present), (1,), device=out[present[0]].device))]
            out[keep][bi] = True
    return out


def build_model(cfg, dataset):
    model_cfg = cfg.get("model", {})
    swin = model_cfg.get("swin", {})
    instrument_dims = {n: s.latent_dim for n, s in dataset.specs.items()}
    return AtmosphereFusionForecastModel(
        instrument_dims=instrument_dims,
        atmosphere_dim=int(model_cfg.get("atmosphere_dim", 96)),
        latent_levels=int(model_cfg.get("latent_levels", 4)),
        pressure_levels_hpa=model_cfg.get("pressure_levels_hpa"),
        fusion_refine_blocks=int(model_cfg.get("fusion_refine_blocks", 2)),
        swin_depth=int(swin.get("depth", 6)),
        swin_num_heads=int(swin.get("num_heads", 6)),
        swin_window_size=tuple(swin.get("window_size", [2, 7, 6])),
        swin_mlp_ratio=float(swin.get("mlp_ratio", 4.0)),
        swin_drop=float(swin.get("drop", 0.0)),
        swin_attn_drop=float(swin.get("attn_drop", 0.0)),
        swin_drop_path=float(swin.get("drop_path", 0.1)),
    )


def step_loss(model, batch, cfg, training=True):
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})
    recon_weight = float(loss_cfg.get("current_reconstruction_weight", 0.25))
    forecast_weight = float(loss_cfg.get("forecast_weight", 1.0))
    density_weight = float(loss_cfg.get("density_weight", 0.05))
    density_threshold = float(loss_cfg.get("density_threshold", 1.0e-6))
    dropout = float(train_cfg.get("instrument_dropout", 0.0)) if training else 0.0

    input_available = {
        name: batch["available"][name][:, 0] for name in model.instrument_names
    }
    if training:
        input_available = apply_instrument_dropout(input_available, dropout)
    initial_latents = {
        name: batch["latents"][name][:, 0] for name in model.instrument_names
    }
    initial_density = {
        name: batch["densities"][name][:, 0] for name in model.instrument_names
    }
    state, _ = model.fuse(initial_latents, initial_density, input_available)

    total = state.new_zeros(())
    terms = {"current_latent": 0.0, "future_latent": 0.0, "density": 0.0}
    if recon_weight > 0:
        current = model.decode_state(state)
        current_loss = state.new_zeros(())
        density_loss = state.new_zeros(())
        for name in model.instrument_names:
            target_avail = batch["available"][name][:, 0]
            current_loss = current_loss + latent_coverage_mse(
                current[name]["latent"], batch["latents"][name][:, 0],
                batch["densities"][name][:, 0], target_avail, density_threshold
            )
            density_loss = density_loss + log_density_mse(
                current[name]["log_density"], batch["densities"][name][:, 0], target_avail
            )
        current_loss = current_loss / len(model.instrument_names)
        density_loss = density_loss / len(model.instrument_names)
        total = total + recon_weight * current_loss + density_weight * density_loss
        terms["current_latent"] = float(current_loss.detach())
        terms["density"] += float(density_loss.detach())

    future_latent_total = state.new_zeros(())
    future_density_total = state.new_zeros(())
    rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
    for step in range(1, rollout_steps + 1):
        state = model.forecast_state(state)
        pred = model.decode_state(state)
        step_latent = state.new_zeros(())
        step_density = state.new_zeros(())
        for name in model.instrument_names:
            target_avail = batch["available"][name][:, step]
            step_latent = step_latent + latent_coverage_mse(
                pred[name]["latent"], batch["latents"][name][:, step],
                batch["densities"][name][:, step], target_avail, density_threshold
            )
            step_density = step_density + log_density_mse(
                pred[name]["log_density"], batch["densities"][name][:, step], target_avail
            )
        future_latent_total = future_latent_total + step_latent / len(model.instrument_names)
        future_density_total = future_density_total + step_density / len(model.instrument_names)
    if rollout_steps > 0:
        future_latent_total = future_latent_total / rollout_steps
        future_density_total = future_density_total / rollout_steps
        total = total + forecast_weight * future_latent_total + density_weight * future_density_total
        terms["future_latent"] = float(future_latent_total.detach())
        terms["density"] += float(future_density_total.detach())
    terms["total"] = float(total.detach())
    return total, terms


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None,
              amp_enabled=False, amp_dtype=torch.float16, max_steps=None, desc="train"):
    training = optimizer is not None
    model.train(training)
    total_sum = 0.0
    count = 0
    progress = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(progress, 1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, terms = step_loss(model, batch, cfg, training=training)
        if training:
            scaler.scale(loss).backward()
            grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        total_sum += float(loss.detach())
        count += 1
        progress.set_postfix(loss=f"{loss.item():.5f}", fcast=f"{terms['future_latent']:.5f}")
        if max_steps is not None and step >= max_steps:
            break
    return total_sum / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed, bool(train_cfg.get("deterministic", False)))

    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"],
        rollout_steps=int(data_cfg.get("rollout_steps", 1)),
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    train_idx, val_idx, test_idx = dataset.split_chronological(
        float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
    )
    if not train_idx or not val_idx:
        raise ValueError(f"Insufficient split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"sequences train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("instrument dims:", {n: s.latent_dim for n, s in dataset.specs.items()})

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        batch_size=int(train_cfg.get("batch_size", 1)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, dataset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    # Scheduler is stepped once per epoch below, so its schedule is defined in epochs.
    total_steps = epochs
    warmup_steps = int(train_cfg.get("warmup_epochs", 5))
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    base_lr = float(train_cfg.get("lr", 1.0e-4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_warmup_lambda(s, total_steps, warmup_steps, min_lr / base_lr),
    )
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(train_cfg.get("amp_dtype", "bfloat16"))
    scaler = make_grad_scaler(amp_enabled)

    output_dir = Path(cfg.get("output", {}).get("dir", "runs/atmosphere_fusion"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    start_epoch = 0
    best_val = math.inf
    bad_epochs = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))

    patience = int(train_cfg.get("early_stopping_patience", 20))
    max_steps = train_cfg.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    for epoch in range(start_epoch, epochs):
        train_loss = run_epoch(
            model, train_loader, device, cfg, optimizer=optimizer, scaler=scaler,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, max_steps=max_steps,
            desc=f"train {epoch:03d}",
        )
        scheduler.step()
        with torch.no_grad():
            val_loss = run_epoch(
                model, val_loader, device, cfg, optimizer=None, scaler=scaler,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, max_steps=max_steps,
                desc=f"val {epoch:03d}",
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(record)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "config": cfg,
            "instrument_specs": {
                n: {"latent_dim": s.latent_dim, "height": s.height, "width": s.width}
                for n, s in dataset.specs.items()
            },
        }
        save_checkpoint(output_dir / "last.pth", payload)
        if improved:
            save_checkpoint(output_dir / "best.pth", payload)
        if patience > 0 and bad_epochs >= patience:
            print(f"early stopping at epoch {epoch}; best_val={best_val:.6f}")
            break


if __name__ == "__main__":
    main()
