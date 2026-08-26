"""Train multi-instrument latent fusion and atmospheric forecasting."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.early_stopping import EarlyStopping
from atmosphere.losses import latent_mse
from atmosphere.models import AtmosphereFusionForecastModel
from atmosphere.optimizer import build_optimizer
from atmosphere.scheduler import build_scheduler, step_scheduler
from atmosphere.utils import amp_dtype_from_name, make_grad_scaler, seed_everything


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
    batch_size = next(iter(out.values())).shape[0]
    for batch_index in range(batch_size):
        present = [name for name in names if bool(out[name][batch_index])]
        if len(present) <= 1:
            continue
        for name in present:
            if torch.rand((), device=out[name].device) < probability:
                out[name][batch_index] = False
        if not any(bool(out[name][batch_index]) for name in present):
            keep_index = torch.randint(
                len(present), (1,), device=out[present[0]].device
            ).item()
            out[present[keep_index]][batch_index] = True
    return out


def build_model(cfg, dataset):
    model_cfg = cfg.get("model", {})
    swin = model_cfg.get("swin", {})
    instrument_dims = {
        name: spec.latent_dim for name, spec in dataset.specs.items()
    }
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
        spatial_multiple=int(model_cfg.get("spatial_multiple", 9)),
    )


def step_loss(model, batch, cfg, training=True):
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})
    reconstruction_weight = float(
        loss_cfg.get("current_reconstruction_weight", 0.25)
    )
    forecast_weight = float(loss_cfg.get("forecast_weight", 1.0))
    use_density_mask = bool(loss_cfg.get("use_density_mask", False))
    density_threshold = float(loss_cfg.get("density_threshold", 1.0e-6))
    dropout = (
        float(train_cfg.get("instrument_dropout", 0.0)) if training else 0.0
    )

    input_available = {
        name: batch["available"][name][:, 0]
        for name in model.instrument_names
    }
    if training:
        input_available = apply_instrument_dropout(input_available, dropout)
    initial_latents = {
        name: batch["latents"][name][:, 0]
        for name in model.instrument_names
    }
    initial_densities = {
        name: batch["densities"][name][:, 0]
        for name in model.instrument_names
    }
    output_shapes = model.spatial_shapes(initial_latents)
    state, _ = model.fuse(
        initial_latents, initial_densities, input_available
    )

    total = state.new_zeros(())
    current_latent_loss = state.new_zeros(())
    if reconstruction_weight > 0:
        current = model.decode_state(state, output_shapes)
        for name in model.instrument_names:
            target_available = batch["available"][name][:, 0]
            current_latent_loss = current_latent_loss + latent_mse(
                current[name]["latent"], batch["latents"][name][:, 0],
                target_available, density=batch["densities"][name][:, 0],
                use_density_mask=use_density_mask,
                density_threshold=density_threshold,
            )
        current_latent_loss = current_latent_loss / len(model.instrument_names)
        total = total + reconstruction_weight * current_latent_loss

    future_latent_loss = state.new_zeros(())
    rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
    for rollout_step in range(1, rollout_steps + 1):
        state = model.forecast_state(state)
        prediction = model.decode_state(state, output_shapes)
        step_latent_loss = state.new_zeros(())
        for name in model.instrument_names:
            target_available = batch["available"][name][:, rollout_step]
            step_latent_loss = step_latent_loss + latent_mse(
                prediction[name]["latent"],
                batch["latents"][name][:, rollout_step],
                target_available,
                density=batch["densities"][name][:, rollout_step],
                use_density_mask=use_density_mask,
                density_threshold=density_threshold,
            )
        future_latent_loss = (
            future_latent_loss
            + step_latent_loss / len(model.instrument_names)
        )

    if rollout_steps > 0:
        future_latent_loss = future_latent_loss / rollout_steps
        total = total + forecast_weight * future_latent_loss
    terms = {
        "total": float(total.detach()),
        "current_latent": float(current_latent_loss.detach()),
        "future_latent": float(future_latent_loss.detach()),
    }
    return total, terms


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None,
              scheduler=None, scheduler_name="none", amp_enabled=False,
              amp_dtype=torch.float16, max_steps=None, desc="train"):
    training = optimizer is not None
    model.train(training)
    sums = {
        "total": 0.0, "current_latent": 0.0,
        "future_latent": 0.0,
    }
    count = 0
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    progress = tqdm(loader, total=total_steps, desc=desc, unit="batch", leave=True)
    for step, batch in enumerate(progress, 1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            loss, terms = step_loss(model, batch, cfg, training=training)
        if training:
            scaler.scale(loss).backward()
            grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if str(scheduler_name).lower() == "warmup_cosine":
                step_scheduler(scheduler, scheduler_name)
        for name in sums:
            sums[name] += terms[name]
        count += 1
        progress.set_postfix(
            loss=f"{terms['total']:.5f}",
            forecast=f"{terms['future_latent']:.5f}",
        )
        if max_steps is not None and step >= max_steps:
            break
    return {name: value / max(count, 1) for name, value in sums.items()}


def capture_random_state(loader_generator):
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state, loader_generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_latest_checkpoint(runs_dir, run_name):
    candidates = list((runs_dir / run_name).glob("*/latest.pth"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def trim_log_after_epoch(path, completed_epoch):
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["epoch"]) <= completed_epoch:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def save_loss_history(path, history):
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def checkpoint_state(epoch, model, optimizer, scheduler, scaler,
                     early_stopping, history, cfg, loader_generator):
    return {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "early_stopping": early_stopping.state_dict(),
        "history": history, "config": cfg,
        "random_state": capture_random_state(loader_generator),
    }


def choose_device(name):
    name = str(name)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is cuda, but CUDA is unavailable")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", nargs="?", const="auto",
        help="Checkpoint path, or omit the path to find the latest checkpoint",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    early_cfg = cfg.get("early_stopping", {})
    output_cfg = cfg.get("output", {})

    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", False))
    seed_everything(seed, deterministic)
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"],
        rollout_steps=int(data_cfg.get("rollout_steps", 1)),
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    train_indices, val_indices, test_indices = dataset.split_chronological(
        float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
    )
    if not train_indices or not val_indices:
        raise ValueError(
            f"Insufficient split: train={len(train_indices)} "
            f"val={len(val_indices)} test={len(test_indices)}"
        )

    device = choose_device(train_cfg.get("device", "auto"))
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size, "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True,
        generator=loader_generator, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_kwargs
    )

    model = build_model(cfg, dataset).to(device)
    optimizer_name = optimizer_cfg.get("name", "adamw")
    optimizer = build_optimizer(
        model, name=optimizer_name,
        lr=float(optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    max_steps = train_cfg.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    max_val_steps = train_cfg.get("max_validation_steps")
    max_val_steps = int(max_val_steps) if max_val_steps is not None else None
    steps_per_epoch = len(train_loader)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    scheduler_name = scheduler_cfg.get("name", "warmup_cosine")
    scheduler_options = {
        key: value for key, value in scheduler_cfg.items() if key != "name"
    }
    scheduler_options.update({
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
    })
    scheduler = build_scheduler(optimizer, scheduler_name, **scheduler_options)

    mixed_precision = bool(train_cfg.get("mixed_precision", True))
    amp_enabled = mixed_precision and device.type == "cuda"
    amp_dtype_name = train_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = amp_dtype_from_name(amp_dtype_name)
    scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    validate_every = int(train_cfg.get("validate_every_epochs", 1))
    save_every = int(train_cfg.get("save_every_epochs", 1))
    if validate_every < 1 or save_every < 1:
        raise ValueError("validation/save frequencies must be at least one")
    early_stopping = EarlyStopping(
        patience=int(early_cfg.get("patience", 20)),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
    )

    runs_dir = Path(output_cfg.get("runs_dir", "runs"))
    run_name = str(output_cfg.get("run_name", "atmosphere_fusion"))
    resume = args.resume if args.resume is not None else train_cfg.get("resume", False)
    if resume is True:
        resume = "auto"
    resume_path = None
    if resume == "auto":
        resume_path = find_latest_checkpoint(runs_dir, run_name)
        if resume_path is None:
            print("No latest checkpoint found; starting a new run.")
    elif resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if resume_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_path.parent
    log_path = run_dir / "loss_log.jsonl"
    history_path = run_dir / "loss_history.json"
    resolved_path = run_dir / (
        "resolved_config_resume.json" if resume_path else "resolved_config.json"
    )
    resolved_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    history = {"train": [], "val": [], "learning_rate": []}
    start_epoch = 1
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        early_state = checkpoint.get("early_stopping")
        if early_state is None:
            early_state = {
                "best_loss": checkpoint.get("best_val"),
                "bad_validations": checkpoint.get("bad_epochs", 0),
            }
        early_stopping.load_state_dict(early_state)
        history = checkpoint.get("history", history)
        history.setdefault("train", [])
        history.setdefault("val", [])
        history.setdefault("learning_rate", [])
        restore_random_state(checkpoint.get("random_state"), loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        trim_log_after_epoch(log_path, int(checkpoint["epoch"]))
        print(f"resumed_from={resume_path} next_epoch={start_epoch}")

    print(f"run_dir={run_dir}")
    print(
        f"device={device} train={len(train_indices)} val={len(val_indices)} "
        f"test={len(test_indices)} steps_per_epoch={steps_per_epoch}"
    )
    print("instrument_dims=", {
        name: spec.latent_dim for name, spec in dataset.specs.items()
    })

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, cfg,
            optimizer=optimizer, scaler=scaler,
            scheduler=scheduler, scheduler_name=scheduler_name,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            max_steps=max_steps, desc=f"train epoch {epoch}",
        )
        val_metrics = None
        improved = False
        should_stop = False
        if epoch % validate_every == 0 or epoch == epochs:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, device, cfg,
                    scaler=scaler, amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype, max_steps=max_val_steps,
                    desc=f"val   epoch {epoch}",
                )
            improved, should_stop = early_stopping.update(val_metrics["total"])
            if str(scheduler_name).lower() == "reduce_on_plateau":
                step_scheduler(scheduler, scheduler_name, val_metrics["total"])
        if str(scheduler_name).lower() == "step":
            step_scheduler(scheduler, scheduler_name)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train"].append({"epoch": epoch, **train_metrics})
        if val_metrics is not None:
            history["val"].append({"epoch": epoch, **val_metrics})
        history["learning_rate"].append({"epoch": epoch, "lr": learning_rate})
        record = {
            "epoch": epoch, "train": train_metrics,
            "val": val_metrics, "learning_rate": learning_rate,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_loss_history(history_path, history)
        print(json.dumps(record))

        state = checkpoint_state(
            epoch, model, optimizer, scheduler, scaler,
            early_stopping, history, cfg, loader_generator,
        )
        if improved:
            torch.save(state, run_dir / "best.pth")
        if epoch % save_every == 0 or epoch == epochs or should_stop:
            torch.save(state, run_dir / f"epoch_{epoch:04d}.pth")
            torch.save(state, run_dir / "latest.pth")
        if should_stop:
            print(
                f"early_stopping at epoch={epoch}; "
                f"best_val_loss={early_stopping.best_loss:.6f}"
            )
            break


if __name__ == "__main__":
    main()
