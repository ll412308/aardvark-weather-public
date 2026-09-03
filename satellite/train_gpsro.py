"""Train the single-instrument GPSRO 3-D SetConv autoencoder."""

import argparse
import json
import random
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from satellite.datasets import GPSROZarrDataset
from satellite.early_stopping import EarlyStopping
from satellite.gpsro_config import GPSROConfig
from satellite.gpsro_plotting import (
    decode_global_3d,
    load_refractivity_stats,
    save_global_3d,
    save_loss_plot,
    save_reconstruction_3d,
)
from satellite.loss import masked_mse_loss
from satellite.models import GPSROAutoEncoder
from satellite.optimizer import build_optimizer
from satellite.scheduler import build_scheduler, step_scheduler


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_config(raw):
    values = dict(raw.get("model", {}))
    for key in ("n_context", "n_target", "target_overlap"):
        if key in raw.get("data", {}):
            values[key] = raw["data"][key]
    allowed = {field.name for field in fields(GPSROConfig)}
    return GPSROConfig(**{
        key: value for key, value in values.items() if key in allowed
    })


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def move(mapping, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in mapping.items()
    }


def split_indices(length, val_fraction, test_fraction):
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1)")
    train_end = max(1, round(length * (1 - val_fraction - test_fraction)))
    val_end = max(train_end + 1, round(length * (1 - test_fraction)))
    val_end = min(val_end, length - 1)
    train = list(range(0, train_end))
    val = list(range(train_end, val_end))
    test = list(range(val_end, length))
    if not train or not val or not test:
        raise ValueError(
            f"Insufficient chronological split: train={len(train)} "
            f"val={len(val)} test={len(test)}"
        )
    return train, val, test


def model_report(model, path):
    details = []
    for name, parameter in model.named_parameters():
        details.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "trainable": bool(parameter.requires_grad),
        })
    total = sum(item["numel"] for item in details)
    trainable = sum(item["numel"] for item in details if item["trainable"])
    report = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "parameter_size_mib_float32": total * 4 / 1024 ** 2,
        "parameters": details,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"parameters total={total:,} trainable={trainable:,} "
        f"float32_size={report['parameter_size_mib_float32']:.2f} MiB"
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_epoch(model, loader, device, amp_enabled, amp_dtype, max_steps=None,
              optimizer=None, scaler=None, scheduler=None,
              scheduler_name="none", grad_clip=1.0, description="train",
              capture_reconstruction=False):
    training = optimizer is not None
    model.train(training)
    error_sum = 0.0
    valid_count = 0
    reconstruction = None
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    progress = tqdm(loader, total=total_steps, desc=description, unit="batch")
    for step, batch in enumerate(progress, start=1):
        context = move(batch["context"], device)
        target = move(batch["target"], device)
        target_value = batch["target_refractivity"].to(
            device, non_blocking=True
        )
        target_valid = batch["target_valid"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            prediction, latent, _ = model(context, target)
            loss = masked_mse_loss(prediction, target_value, target_valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {description} loss at step={step}"
            )
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler_name.lower() == "warmup_cosine":
                step_scheduler(scheduler, scheduler_name)
        count = int(target_valid.sum().item())
        error_sum += float(loss.item()) * count
        valid_count += count
        if capture_reconstruction and reconstruction is None:
            reconstruction = {
                "prediction": prediction[0].float().cpu(),
                "target": target_value[0].float().cpu(),
                "valid": target_valid[0].cpu(),
                "lon": target["lon"][0].float().cpu(),
                "lat": target["lat"][0].float().cpu(),
                "height": target["height"][0].float().cpu(),
                "satellite_id": target["satellite_id"][0].cpu(),
                "sample_time": target["sample_time"][0].cpu(),
                "latent": latent[:1].float().cpu(),
            }
        progress.set_postfix(loss=f"{loss.item():.5f}")
        if max_steps is not None and step >= max_steps:
            break
    return error_sum / max(valid_count, 1), reconstruction


def _select_points(points, indices):
    """Index point arrays while retaining scalar sample_time unchanged."""
    return {
        name: value if value.ndim == 0 else value[indices]
        for name, value in points.items()
    }


def _batch_to_device(points, device):
    return {
        name: value.unsqueeze(0).to(device, non_blocking=True)
        for name, value in points.items()
    }


def save_validation_plot(reconstruction, output_dir, epoch, plot_options,
                         refractivity_stats):
    if reconstruction is None or not plot_options["validation_enabled"]:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reconstruction, output_dir / "reconstruction.pth")
    sample_time = int(reconstruction["sample_time"])
    save_reconstruction_3d(
        lon=reconstruction["lon"], lat=reconstruction["lat"],
        height=reconstruction["height"], truth=reconstruction["target"],
        prediction=reconstruction["prediction"],
        valid=reconstruction["valid"], output_dir=output_dir,
        prefix=f"validation_epoch_{epoch:04d}",
        stats=refractivity_stats,
        value_space=plot_options["value_space"],
        max_points=plot_options["max_points"],
        point_size=plot_options["point_size"],
        seed=plot_options["seed"] + epoch,
        title_prefix=(
            f"GPSRO validation epoch {epoch} | "
            f"time={np.datetime64(sample_time, 'ns')}"
        ),
    )


@torch.no_grad()
def save_test_plots(model, dataset, dataset_index, output_dir, test_options,
                    plot_options, refractivity_stats, device,
                    amp_enabled, amp_dtype):
    """Plot one random real-point reconstruction and one regular 3-D globe."""
    item = dataset.get_full_sample(dataset_index)
    observations = item["observations"]
    count = int(item["count"])
    generator = torch.Generator().manual_seed(
        int(test_options["seed"]) + int(dataset_index)
    )
    permutation = torch.randperm(count, generator=generator)
    context_count = int(test_options["context_points"])
    query_count = int(test_options["random_query_points"])
    context_count = count if context_count <= 0 else min(context_count, count)
    query_count = count if query_count <= 0 else min(query_count, count)
    context_indices = permutation[:context_count].sort().values
    heldout_indices = permutation[context_count:]
    heldout_query_count = min(query_count, len(heldout_indices))
    context_query_count = query_count - heldout_query_count
    query_indices = torch.cat([
        heldout_indices[:heldout_query_count],
        permutation[:context_query_count],
    ]).sort().values
    context = _batch_to_device(
        _select_points(observations, context_indices), device
    )
    query = _select_points(observations, query_indices)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        latent, density = model.encode(
            **context, chunk_size=int(test_options["context_chunk_size"])
        )
        prediction = model.decode(
            latent=latent,
            lon=query["lon"].unsqueeze(0).to(device),
            lat=query["lat"].unsqueeze(0).to(device),
            height=query["height"].unsqueeze(0).to(device),
            satellite_id=query["satellite_id"].unsqueeze(0).to(device),
            is_land=query["is_land"].unsqueeze(0).to(device),
            sample_time=query["sample_time"].reshape(1).to(device),
            chunk_size=int(test_options["query_chunk_size"]),
        )
    prediction = prediction[0].float().cpu()
    valid = query["valid"]
    error = (prediction - query["refractivity"]).square()
    test_mse = float(
        (error * valid.float()).sum() / valid.sum().clamp_min(1)
    )
    source_index = int(item["sample_index"])
    sample_time = int(item["sample_time"])
    sample_dir = output_dir / f"sample_{source_index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sample_{source_index:04d}_context_{context_count}_query_{query_count}"
    save_reconstruction_3d(
        lon=query["lon"], lat=query["lat"], height=query["height"],
        truth=query["refractivity"], prediction=prediction,
        valid=valid, output_dir=sample_dir, prefix=prefix,
        stats=refractivity_stats, value_space=plot_options["value_space"],
        max_points=plot_options["test_max_points"],
        point_size=plot_options["point_size"], seed=test_options["seed"],
        title_prefix=(
            f"GPSRO test | source={source_index} | "
            f"time={np.datetime64(sample_time, 'ns')}"
        ),
    )
    metrics = {
        "dataset_index": int(dataset_index),
        "source_sample_index": source_index,
        "sample_time_ns": sample_time,
        "sample_time": str(np.datetime64(sample_time, "ns")),
        "available_observations": count,
        "context_points": context_count,
        "random_query_points": query_count,
        "heldout_query_points": heldout_query_count,
        "context_query_points": context_query_count,
        "query_context_overlap_fraction": context_query_count / max(query_count, 1),
        "masked_mse": test_mse,
    }
    if plot_options["global_enabled"]:
        heights = plot_options["global_heights_m"]
        if heights is None:
            # vertical_max_m is the exclusive upper edge of the final
            # standardisation bin. Querying exactly that boundary creates an
            # unsupported extra level (for example 64 km) and can dominate the
            # log(N) colour scale, so default global plots stop one level below.
            heights = np.arange(
                model.config.vertical_min_m,
                model.config.vertical_max_m,
                model.config.vertical_resolution_m,
                dtype=np.float32,
            ).tolist()
        configured_id = plot_options["global_satellite_id"]
        satellite_id = (
            int(query["satellite_id"][0])
            if configured_id is None else int(configured_id)
        )
        lon, lat, height, global_prediction = decode_global_3d(
            model=model, latent=latent, sample_time=sample_time,
            satellite_id=satellite_id,
            horizontal_resolution_deg=plot_options["global_resolution_deg"],
            heights_m=heights,
            chunk_size=plot_options["global_query_chunk_size"],
            device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        save_global_3d(
            lon, lat, height, global_prediction,
            output_dir=sample_dir, prefix=prefix,
            stats=refractivity_stats,
            value_space=plot_options["value_space"],
            max_points=plot_options["global_max_points"],
            point_size=plot_options["global_point_size"],
            seed=test_options["seed"],
            title=(
                f"GPSRO global reconstruction | satellite={satellite_id} | "
                f"time={np.datetime64(sample_time, 'ns')}"
            ),
        )
        metrics["global_query_points"] = int(len(lon))
        metrics["global_height_levels_m"] = [float(value) for value in heights]
    torch.save({
        "prediction": prediction,
        "target": query["refractivity"],
        "valid": valid,
        "lon": query["lon"], "lat": query["lat"],
        "height": query["height"], "latent": latent.float().cpu(),
        "density": density.float().cpu(), "metrics": metrics,
    }, sample_dir / "random_reconstruction.pth")
    (sample_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"test_plots={sample_dir} random_masked_mse={test_mse:.6f}")
    return metrics


def checkpoint_state(epoch, model, optimizer, scheduler, scaler,
                     early_stopping, history, raw_config):
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "early_stopping": early_stopping.state_dict(),
        "history": history,
        "config": raw_config,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip training/validation and test the requested checkpoint",
    )
    args = parser.parse_args()
    raw = load_yaml(args.config)
    config = make_config(raw)
    data_cfg = raw["data"]
    train_cfg = raw.get("train", {})
    optimizer_cfg = raw.get("optimizer", {})
    scheduler_cfg = raw.get("scheduler", {})
    early_cfg = raw.get("early_stopping", {})
    test_cfg = raw.get("test", {})
    plot_cfg = raw.get("plot", {})
    output_cfg = raw.get("output", {})
    test_only = bool(args.test_only or raw.get("test_only", False))

    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed, bool(train_cfg.get("deterministic", False)))
    device_name = str(train_cfg.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    mixed_precision = bool(train_cfg.get("mixed_precision", True))
    amp_enabled = mixed_precision and device.type == "cuda"
    amp_name = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16

    dataset = GPSROZarrDataset(
        data_cfg["zarr"], config.n_context, config.n_target,
        config.target_overlap, seed,
    )
    train_indices, val_indices, test_indices = split_indices(
        len(dataset), float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
    )
    batch_size = int(train_cfg.get("batch_size", 1))
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", device.type == "cuda")),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True,
        generator=generator, **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), shuffle=False, **loader_kwargs
    )

    resume = args.resume or train_cfg.get("resume")
    resume_path = Path(resume) if resume else None
    if test_only and resume_path is None:
        raise ValueError("test_only requires --resume or train.resume in the YAML")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    if resume_path is None:
        run_dir = (
            Path(output_cfg.get("runs_dir", "runs"))
            / str(output_cfg.get("run_name", "gpsro_autoencoder"))
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_path.parent
    plot_options = {
        "enabled": bool(plot_cfg.get("enabled", True)),
        "loss_enabled": bool(plot_cfg.get("loss_enabled", True)),
        "loss_log_scale": bool(plot_cfg.get("loss_log_scale", True)),
        "validation_enabled": bool(plot_cfg.get("validation_enabled", True)),
        "test_enabled": bool(plot_cfg.get("test_enabled", True)),
        # GPSRO validation/test figures always show inverse-standardized
        # log(N). Model losses and saved tensors remain standardized.
        "value_space": "log",
        "max_points": int(plot_cfg.get("max_points", 30_000)),
        "test_max_points": int(plot_cfg.get("test_max_points", 50_000)),
        "point_size": float(plot_cfg.get("point_size", 4.0)),
        "global_enabled": bool(plot_cfg.get("global_enabled", True)),
        "global_resolution_deg": float(
            plot_cfg.get("global_resolution_deg", 5.0)
        ),
        "global_heights_m": plot_cfg.get("global_heights_m"),
        "global_query_chunk_size": int(
            plot_cfg.get("global_query_chunk_size", config.decode_chunk_size)
        ),
        "global_max_points": int(plot_cfg.get("global_max_points", 100_000)),
        "global_point_size": float(plot_cfg.get("global_point_size", 3.0)),
        "global_satellite_id": plot_cfg.get("global_satellite_id"),
        "seed": int(train_cfg.get("seed", 42)),
    }
    if plot_options["global_resolution_deg"] <= 0:
        raise ValueError("plot.global_resolution_deg must be positive")
    test_options = {
        "enabled": bool(test_cfg.get("enabled", True)) or test_only,
        "max_samples": int(test_cfg.get("max_samples", 1)),
        "context_points": int(test_cfg.get("context_points", config.n_context)),
        "random_query_points": int(
            test_cfg.get("random_query_points", config.n_target)
        ),
        "context_chunk_size": int(
            test_cfg.get("context_chunk_size", config.setconv_chunk_size)
        ),
        "query_chunk_size": int(
            test_cfg.get("query_chunk_size", config.decode_chunk_size)
        ),
        "seed": int(test_cfg.get("seed", seed + 2)),
    }
    plots_dir = run_dir / "plots"
    loss_plot_dir = plots_dir / "loss"
    validation_plot_dir = plots_dir / "validation"
    test_plot_dir = plots_dir / "test"
    if plot_options["enabled"]:
        loss_plot_dir.mkdir(parents=True, exist_ok=True)
        validation_plot_dir.mkdir(parents=True, exist_ok=True)
        test_plot_dir.mkdir(parents=True, exist_ok=True)
    refractivity_stats = (
        load_refractivity_stats(data_cfg["zarr"])
        if plot_options["enabled"] else None
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(raw, indent=2), encoding="utf-8"
    )

    model = GPSROAutoEncoder(config).to(device)
    model_report(model, run_dir / "model_parameters.json")
    optimizer = build_optimizer(
        model, optimizer_cfg.get("name", "adamw"),
        lr=float(optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    max_train_steps = train_cfg.get("max_steps_per_epoch")
    max_train_steps = None if max_train_steps is None else int(max_train_steps)
    max_val_steps = train_cfg.get("max_validation_steps")
    max_val_steps = None if max_val_steps is None else int(max_val_steps)
    steps_per_epoch = min(
        len(train_loader),
        len(train_loader) if max_train_steps is None else max_train_steps,
    )
    scheduler_name = str(scheduler_cfg.get("name", "warmup_cosine"))
    scheduler_options = {
        key: value for key, value in scheduler_cfg.items() if key != "name"
    }
    scheduler_options.update({
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
    })
    scheduler = build_scheduler(
        optimizer, scheduler_name, **scheduler_options
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and amp_dtype == torch.float16
        )
    early_stopping = EarlyStopping(
        int(early_cfg.get("patience", 20)),
        float(early_cfg.get("min_delta", 1.0e-5)),
    )
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    start_epoch = 1
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"])
        history = checkpoint.get("history", history)
        if test_only:
            start_epoch = epochs + 1
            print(f"test_only_checkpoint={resume_path}")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
            start_epoch = int(checkpoint["epoch"]) + 1
            print(f"resumed_from={resume_path} next_epoch={start_epoch}")

    print(f"run_dir={run_dir}")
    print(
        f"device={device} amp={amp_enabled} grid=[{config.grid_depth},"
        f"{config.grid_height},{config.grid_width}] "
        f"train_bins={len(train_indices)} val_bins={len(val_indices)} "
        f"test_bins={len(test_indices)}"
    )
    validate_every = int(train_cfg.get("validate_every_epochs", 1))
    save_every = int(train_cfg.get("save_every_epochs", 2))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    history_path = run_dir / "loss_history.json"
    log_path = run_dir / "loss_log.jsonl"
    for epoch in range(start_epoch, epochs + 1):
        dataset.set_epoch(epoch)
        train_loss, _ = run_epoch(
            model, train_loader, device, amp_enabled, amp_dtype,
            max_steps=max_train_steps, optimizer=optimizer, scaler=scaler,
            scheduler=scheduler, scheduler_name=scheduler_name,
            grad_clip=grad_clip, description=f"train epoch {epoch}",
        )
        val_loss = None
        improved = should_stop = False
        if epoch % validate_every == 0 or epoch == epochs:
            dataset.set_epoch(0)
            val_loss, reconstruction = run_epoch(
                model, val_loader, device, amp_enabled, amp_dtype,
                max_steps=max_val_steps, scaler=scaler,
                description=f"val epoch {epoch}", capture_reconstruction=True,
            )
            if plot_options["enabled"]:
                save_validation_plot(
                    reconstruction,
                    validation_plot_dir / f"epoch_{epoch:04d}",
                    epoch, plot_options, refractivity_stats,
                )
            improved, should_stop = early_stopping.update(val_loss)
            if scheduler_name.lower() == "reduce_on_plateau":
                step_scheduler(scheduler, scheduler_name, val_loss)
        if scheduler_name.lower() == "step":
            step_scheduler(scheduler, scheduler_name)
        learning_rate = optimizer.param_groups[0]["lr"]
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})
        if val_loss is not None:
            history["val_loss"].append({"epoch": epoch, "loss": val_loss})
        history["learning_rate"].append({"epoch": epoch, "lr": learning_rate})
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "learning_rate": learning_rate,
        }
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if plot_options["enabled"] and plot_options["loss_enabled"]:
            loss_path = save_loss_plot(
                history, loss_plot_dir, epoch,
                log_scale=plot_options["loss_log_scale"],
            )
            print(f"loss_plot={loss_path}")
        print(json.dumps(record))
        state = checkpoint_state(
            epoch, model, optimizer, scheduler, scaler,
            early_stopping, history, raw,
        )
        if improved:
            torch.save(state, run_dir / "best.pth")
        if epoch % save_every == 0 or epoch == epochs or should_stop:
            torch.save(state, run_dir / f"epoch_{epoch:04d}.pth")
            torch.save(state, run_dir / "latest.pth")
        if should_stop:
            print(
                f"early_stopping epoch={epoch} "
                f"best_val={early_stopping.best_loss:.6f}"
            )
            break

    best_path = run_dir / "best.pth"
    if not test_only and best_path.exists():
        model.load_state_dict(load_checkpoint(best_path, device)["model"])
    if test_options["enabled"]:
        dataset.set_epoch(0)
        test_loss, _ = run_epoch(
            model, test_loader, device, amp_enabled, amp_dtype,
            scaler=scaler, description="test",
        )
        test_metrics = {"sampled_test_masked_mse": test_loss}
        if plot_options["enabled"] and plot_options["test_enabled"]:
            selected = test_indices[:max(test_options["max_samples"], 0)]
            test_metrics["plotted_samples"] = [
                save_test_plots(
                    model, dataset, dataset_index, test_plot_dir,
                    test_options, plot_options, refractivity_stats,
                    device, amp_enabled, amp_dtype,
                )
                for dataset_index in selected
            ]
        (run_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        print(f"test_masked_mse={test_loss:.6f}")


if __name__ == "__main__":
    main()

#  python -m satellite.train_gpsro --config satellite/configs/gpsro_train.yaml
