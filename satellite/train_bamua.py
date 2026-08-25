import argparse
import json
import math
import random
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from satellite.config import BAMUAConfig
from satellite.datasets import BAMUAZarrDataset
from satellite.early_stopping import EarlyStopping
from satellite.loss import masked_mse_loss
from satellite.models import BAMUAAutoEncoder
from satellite.optimizer import build_optimizer
from satellite.scheduler import build_scheduler, step_scheduler


def to_device(mapping, device):
    return {key: value.to(device) for key, value in mapping.items()}


def seed_everything(seed, deterministic=False):
    """Seed the random-number generators used by this training program."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def capture_random_state(loader_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
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


def load_yaml_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def pick(cli_value, yaml_value, default=None):
    if cli_value is not None:
        return cli_value
    return yaml_value if yaml_value is not None else default


def make_bamua_config(raw):
    model_raw = raw.get("model", {})
    data_raw = raw.get("data", {})
    allowed = {field.name for field in fields(BAMUAConfig)}
    values = {key: value for key, value in model_raw.items() if key in allowed}
    for key in ("n_context", "n_target", "target_overlap"):
        if key in data_raw:
            values[key] = data_raw[key]
    return BAMUAConfig(**values)


def split_datasets_by_time(zarr_path, config, val_fraction, test_fraction,
                           sampling_seed):
    """Use early 6h bins for train and later bins for validation/test."""
    common = dict(
        path=zarr_path,
        n_context=config.n_context,
        n_target=config.n_target,
        target_overlap=config.target_overlap,
    )
    train_dataset = BAMUAZarrDataset(**common, seed=sampling_seed)
    val_dataset = BAMUAZarrDataset(**common, seed=sampling_seed + 1)
    test_dataset = BAMUAZarrDataset(**common, seed=sampling_seed + 2)
    if len(train_dataset) < 3:
        raise ValueError("At least three 6h bins are required for train/val/test")
    root = train_dataset._open()
    times = train_dataset._int64_time(root["time_series"][:])
    indices = np.argsort(times, kind="stable").tolist()
    n_test = max(1, round(len(indices) * test_fraction))
    n_val = max(1, round(len(indices) * val_fraction))
    if n_val + n_test >= len(indices):
        raise ValueError("val_fraction + test_fraction leaves no training bins")
    n_train = len(indices) - n_val - n_test
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    return (
        train_dataset,
        Subset(train_dataset, train_indices),
        val_dataset,
        Subset(val_dataset, val_indices),
        test_dataset,
        test_indices,
        times,
    )


def add_batch_dimension(mapping, device):
    """Convert one unbatched sample from [N,...] to [1,N,...]."""
    return {key: value.unsqueeze(0).to(device) for key, value in mapping.items()}


def select_context(observations, indices):
    """Select context points while preserving the scalar sample_time."""
    context = {}
    for key, value in observations.items():
        context[key] = value if value.ndim == 0 else value[indices]
    return context


def full_test_sample(model, dataset, sample_index, context_fractions, device,
                     output_dir, context_chunk_size, query_chunk_size,
                     amp_enabled=False, amp_dtype=torch.float16, seed=0):
    """Reconstruct every query point using several context percentages."""
    item = dataset.get_full_sample(sample_index)
    observations = item["observations"]
    n_points = item["count"]
    sample_dir = output_dir / f"sample_{sample_index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bt": observations["bt"],
        "valid": observations["valid"],
        "lon": observations["lon"],
        "lat": observations["lat"],
        "sample_time": observations["sample_time"],
    }, sample_dir / "target_all.pth")

    generator = torch.Generator().manual_seed(seed + sample_index)
    permutation = torch.randperm(n_points, generator=generator)
    results = []
    model.eval()
    with torch.no_grad():
        for fraction in context_fractions:
            n_context = max(1, min(n_points, round(n_points * fraction)))
            context_indices = permutation[:n_context].sort().values
            context = add_batch_dimension(
                select_context(observations, context_indices), device
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                latent, density = model.encode_chunked(
                    **context, chunk_size=context_chunk_size
                )

            is_context = torch.zeros(n_points, dtype=torch.bool)
            is_context[context_indices] = True
            predictions = []
            all_error_sum = all_valid_count = 0.0
            context_error_sum = context_valid_count = 0.0
            heldout_error_sum = heldout_valid_count = 0.0
            channel_error_sum = torch.zeros(observations["bt"].shape[-1])
            channel_valid_count = torch.zeros_like(channel_error_sum)

            for start in range(0, n_points, query_chunk_size):
                end = min(start + query_chunk_size, n_points)
                query = {}
                for key in ("lon", "lat", "satellite_id", "is_land",
                            "obs_time", "zenith", "azimuth"):
                    query[key] = observations[key][start:end].unsqueeze(0).to(device)
                query["sample_time"] = observations["sample_time"].unsqueeze(0).to(device)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                ):
                    pred = model.decode(latent=latent, density=density, **query)
                pred = pred.squeeze(0).float().cpu()
                predictions.append(pred)
                truth = observations["bt"][start:end]
                valid = observations["valid"][start:end]
                error = (pred - truth).square() * valid.float()
                all_error_sum += error.sum().item()
                all_valid_count += valid.sum().item()
                channel_error_sum += error.sum(0)
                channel_valid_count += valid.sum(0)
                chunk_context = is_context[start:end].unsqueeze(-1)
                context_mask = valid & chunk_context
                heldout_mask = valid & ~chunk_context
                context_error_sum += (error * context_mask).sum().item()
                context_valid_count += context_mask.sum().item()
                heldout_error_sum += (error * heldout_mask).sum().item()
                heldout_valid_count += heldout_mask.sum().item()

            all_mse = all_error_sum / max(all_valid_count, 1.0)
            context_mse = context_error_sum / max(context_valid_count, 1.0)
            heldout_mse = (
                heldout_error_sum / heldout_valid_count
                if heldout_valid_count > 0 else None
            )
            percent = round(fraction * 100)
            metrics = {
                "sample_index": sample_index,
                "context_fraction": fraction,
                "n_context": n_context,
                "n_query": n_points,
                "all_mse": all_mse,
                "all_rmse": math.sqrt(all_mse),
                "context_mse": context_mse,
                "heldout_mse": heldout_mse,
                "heldout_rmse": math.sqrt(heldout_mse) if heldout_mse is not None else None,
                "channel_mse": (
                    channel_error_sum / channel_valid_count.clamp_min(1)
                ).tolist(),
            }
            torch.save({
                "pred": torch.cat(predictions),
                "context_indices": context_indices,
                "metrics": metrics,
            }, sample_dir / f"context_{percent:03d}_percent.pth")
            results.append(metrics)
            print(
                f"test_sample={sample_index} context={percent}% "
                f"all_rmse={metrics['all_rmse']:.6f} "
                f"heldout_rmse={metrics['heldout_rmse']}"
            )
    (sample_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def run_full_test(model, dataset, test_indices, context_fractions, device,
                  output_dir, context_chunk_size, query_chunk_size,
                  max_samples=None, amp_enabled=False,
                  amp_dtype=torch.float16, seed=0):
    selected = test_indices if max_samples is None else test_indices[:max_samples]
    all_results = []
    for sample_index in selected:
        all_results.extend(full_test_sample(
            model, dataset, sample_index, context_fractions, device,
            output_dir, context_chunk_size, query_chunk_size,
            amp_enabled, amp_dtype, seed,
        ))
    (output_dir / "all_metrics.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    return all_results


def run_validation(model, loader, device, reconstruction_path=None,
                   amp_enabled=False, amp_dtype=torch.float16,
                   epoch=None):
    model.eval()
    squared_error_sum = 0.0
    valid_count = 0
    channel_error_sum = torch.zeros(15, dtype=torch.float64)
    channel_valid_count = torch.zeros(15, dtype=torch.float64)
    first_reconstruction = None
    with torch.no_grad():
        progress = tqdm(
            loader, desc=f"val   epoch {epoch}" if epoch is not None else "validation",
            unit="batch", leave=False,
        )
        for batch in progress:
            context = to_device(batch["context"], device)
            target = to_device(batch["target"], device)
            target_bt = batch["target_bt"].to(device)
            target_valid = batch["target_valid"].to(device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                pred, _, _ = model(context, target)
                loss = masked_mse_loss(pred, target_bt, target_valid)
            count = int(target_valid.sum().item())
            squared_error_sum += float(loss.item()) * count
            valid_count += count
            error = (pred.float() - target_bt.float()).square()
            valid_float = target_valid.float()
            channel_error_sum += (error * valid_float).sum(dim=(0, 1)).cpu().double()
            channel_valid_count += valid_float.sum(dim=(0, 1)).cpu().double()
            progress.set_postfix(loss=f"{loss.item():.5f}")
            if first_reconstruction is None:
                # Save one batch for inspecting the actual reconstructed BT values.
                first_reconstruction = {
                    "pred": pred.cpu(),
                    "target_bt": target_bt.cpu(),
                    "target_valid": target_valid.cpu(),
                    "lon": target["lon"].cpu(),
                    "lat": target["lat"].cpu(),
                    "sample_time": target["sample_time"].cpu(),
                }
    if reconstruction_path is not None and first_reconstruction is not None:
        torch.save(first_reconstruction, reconstruction_path)
    channel_loss = channel_error_sum / channel_valid_count.clamp_min(1.0)
    return squared_error_sum / max(valid_count, 1), channel_loss.tolist()


def save_loss_history(path, history):
    """Keep an easy-to-read loss file separate from model checkpoints."""
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def format_channel_losses(values):
    if values is None:
        return "not_run"
    return " ".join(
        f"ch{channel + 1:02d}={value:.6f}"
        for channel, value in enumerate(values)
    )


def checkpoint_state(epoch, model, optimizer, scheduler, scaler, early_stopping,
                     history, config, resolved, loader_generator):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "early_stopping": early_stopping.state_dict(),
        "history": history,
        "config": vars(config),
        "resolved_config": resolved,
        "random_state": capture_random_state(loader_generator),
    }


def find_latest_checkpoint(runs_dir, run_name):
    # Only search directly inside timestamped run directories. This excludes
    # reconstructions/*.pth, which contain predictions rather than model state.
    candidates = list((runs_dir / run_name).glob("*/*.pth"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def trim_log_after_epoch(log_path, completed_epoch):
    """Remove unsaved trailing log rows when resuming an older checkpoint."""
    if not log_path.exists():
        return
    kept = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["epoch"]) <= completed_epoch:
            kept.append(line)
    text = "\n".join(kept)
    log_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML experiment config")
    parser.add_argument("--zarr")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-context", type=int)
    parser.add_argument("--n-target", type=int)
    parser.add_argument("--target-overlap", type=float)
    parser.add_argument("--resolution", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir")
    parser.add_argument(
        "--test-only", action="store_true",
        help="Load a checkpoint and run full 100/90/80 percent reconstruction only",
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto",
        help="Checkpoint path, or omit the path to find the latest checkpoint",
    )
    args = parser.parse_args()

    raw = load_yaml_config(args.config)
    data_raw = raw.setdefault("data", {})
    train_raw = raw.get("train", {})
    optimizer_raw = raw.get("optimizer", {})
    scheduler_raw = raw.get("scheduler", {})
    early_raw = raw.get("early_stopping", {})
    test_raw = raw.get("test", {})
    output_raw = raw.get("output", {})

    if args.n_context is not None:
        data_raw["n_context"] = args.n_context
    if args.n_target is not None:
        data_raw["n_target"] = args.n_target
    if args.target_overlap is not None:
        data_raw["target_overlap"] = args.target_overlap
    if args.resolution is not None:
        raw.setdefault("model", {})["grid_resolution_deg"] = args.resolution
    config = make_bamua_config(raw)

    zarr_path = pick(args.zarr, data_raw.get("zarr"))
    if zarr_path is None:
        raise ValueError("Provide --zarr or data.zarr in YAML")
    epochs = pick(args.epochs, train_raw.get("epochs"), 10)
    batch_size = pick(args.batch_size, train_raw.get("batch_size"), 1)
    num_workers = pick(args.num_workers, train_raw.get("num_workers"), 0)
    max_steps = pick(args.max_steps, train_raw.get("max_steps"))
    seed = int(train_raw.get("seed", 42))
    deterministic = bool(train_raw.get("deterministic", False))
    mixed_precision = bool(train_raw.get("mixed_precision", True))
    amp_dtype_name = str(train_raw.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in ("float16", "bfloat16"):
        raise ValueError("train.amp_dtype must be 'float16' or 'bfloat16'")
    validate_every = int(train_raw.get("validate_every_epochs", 1))
    save_every = int(train_raw.get("save_every_epochs", 1))
    if validate_every < 1 or save_every < 1:
        raise ValueError("validate_every_epochs and save_every_epochs must be >= 1")
    val_fraction = float(data_raw.get("val_fraction", 0.2))
    test_fraction = float(data_raw.get("test_fraction", 0.1))
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction and test_fraction must be positive and sum to less than 1"
        )
    test_enabled = bool(test_raw.get("enabled", True)) or args.test_only
    context_fractions = [float(value) for value in
                         test_raw.get("context_fractions", [1.0, 0.9, 0.8])]
    if any(value <= 0 or value > 1 for value in context_fractions):
        raise ValueError("test.context_fractions values must be in (0, 1]")
    context_chunk_size = int(test_raw.get("context_chunk_size", 16_384))
    query_chunk_size = int(test_raw.get("query_chunk_size", 16_384))
    test_max_samples = test_raw.get("max_samples")
    if test_max_samples is not None:
        test_max_samples = int(test_max_samples)
    run_name = pick(args.run_name, output_raw.get("run_name"), "bamua_autoencoder")
    runs_dir = Path(pick(args.runs_dir, output_raw.get("runs_dir"), "runs"))

    optimizer_name = optimizer_raw.get("name", "adamw")
    lr = pick(args.lr, optimizer_raw.get("lr", train_raw.get("lr")), 1.0e-3)
    weight_decay = float(optimizer_raw.get("weight_decay", 1.0e-4))
    scheduler_name = scheduler_raw.get("name", "reduce_on_plateau")
    resume = pick(args.resume, train_raw.get("resume"))
    if args.test_only and not resume:
        resume = "auto"
    if resume is True:
        resume = "auto"

    resume_path = None
    if resume == "auto":
        resume_path = find_latest_checkpoint(runs_dir, run_name)
        if resume_path is None:
            print("No previous latest.pth found; starting a new run.")
    elif resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if args.test_only and resume_path is None:
        raise FileNotFoundError("--test-only requires an existing checkpoint")

    if resume_path is not None:
        run_dir = resume_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
    reconstruction_dir = run_dir / "reconstructions"
    reconstruction_dir.mkdir(exist_ok=True)
    test_output_dir = run_dir / "test_reconstructions"
    test_output_dir.mkdir(exist_ok=True)
    log_path = run_dir / "loss_log.jsonl"
    history_path = run_dir / "loss_history.json"

    resolved = {
        "data": {**data_raw, "zarr": zarr_path},
        "model": vars(config),
        "train": {
            "epochs": epochs, "batch_size": batch_size,
            "num_workers": num_workers, "max_steps": max_steps,
            "seed": seed, "deterministic": deterministic,
            "mixed_precision": mixed_precision, "amp_dtype": amp_dtype_name,
            "validate_every_epochs": validate_every,
            "save_every_epochs": save_every,
        },
        "optimizer": {"name": optimizer_name, "lr": lr,
                      "weight_decay": weight_decay},
        "scheduler": scheduler_raw,
        "early_stopping": early_raw,
        "test": test_raw,
        "output": {"run_name": run_name, "run_dir": str(run_dir)},
    }
    config_name = "resolved_config_resume.json" if resume_path else "resolved_config.json"
    (run_dir / config_name).write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    seed_everything(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = mixed_precision and device.type == "cuda"
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    if mixed_precision and not amp_enabled:
        print("mixed_precision requested, but CUDA is unavailable; using float32.")
    (train_base, train_subset, val_base, val_subset,
     test_dataset, test_indices, sample_times) = split_datasets_by_time(
        zarr_path, config, val_fraction, test_fraction, seed
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    model = BAMUAAutoEncoder(config).to(device)
    optimizer = build_optimizer(
        model, optimizer_name, lr=lr, weight_decay=weight_decay
    )
    steps_per_epoch = len(train_loader)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    scheduler_options = {
        key: value for key, value in scheduler_raw.items() if key != "name"
    }
    scheduler_options.update({
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
    })
    scheduler = build_scheduler(optimizer, scheduler_name, **scheduler_options)
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    early_stopping = EarlyStopping(
        patience=early_raw.get("patience", 10),
        min_delta=early_raw.get("min_delta", 0.0),
    )
    history = {
        "train_loss": [], "val_loss": [], "learning_rate": [],
        "train_channel_loss": [], "val_channel_loss": [],
    }
    start_epoch = 1

    if resume_path is not None:
        try:
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
        except TypeError:  # Compatibility with older PyTorch versions.
            checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
        history = checkpoint.get("history", history)
        # Older checkpoints did not contain per-channel histories.
        history.setdefault("train_channel_loss", [])
        history.setdefault("val_channel_loss", [])
        restore_random_state(checkpoint.get("random_state"), loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.test_only:
            start_epoch = epochs + 1
        trim_log_after_epoch(log_path, int(checkpoint["epoch"]))
        print(f"resumed_from={resume_path} next_epoch={start_epoch}")

    print(f"run_dir={run_dir}")
    print(
        f"device={device} mixed_precision={amp_enabled} amp_dtype={amp_dtype_name} "
        f"train_bins={len(train_subset)} val_bins={len(val_subset)} "
        f"test_bins={len(test_indices)} seed={seed}"
    )
    train_ids = train_subset.indices
    val_ids = val_subset.indices
    print(
        "time_split: "
        f"train=[{np.datetime64(int(sample_times[train_ids[0]]), 'ns')}, "
        f"{np.datetime64(int(sample_times[train_ids[-1]]), 'ns')}], "
        f"val=[{np.datetime64(int(sample_times[val_ids[0]]), 'ns')}, "
        f"{np.datetime64(int(sample_times[val_ids[-1]]), 'ns')}], "
        f"test=[{np.datetime64(int(sample_times[test_indices[0]]), 'ns')}, "
        f"{np.datetime64(int(sample_times[test_indices[-1]]), 'ns')}]"
    )
    for epoch in range(start_epoch, epochs + 1):
        train_base.set_epoch(epoch)
        model.train()
        squared_error_sum = 0.0
        valid_count = 0
        channel_error_sum = torch.zeros(15, dtype=torch.float64)
        channel_valid_count = torch.zeros(15, dtype=torch.float64)
        progress = tqdm(
            train_loader, total=steps_per_epoch, desc=f"train epoch {epoch}",
            unit="batch", leave=True,
        )
        for step, batch in enumerate(progress, start=1):
            context = to_device(batch["context"], device)
            target = to_device(batch["target"], device)
            target_bt = batch["target_bt"].to(device)
            target_valid = batch["target_valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                pred, _, _ = model(context, target)
                loss = masked_mse_loss(pred, target_bt, target_valid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler_name.lower() == "warmup_cosine":
                step_scheduler(scheduler, scheduler_name)
            count = int(target_valid.sum().item())
            squared_error_sum += float(loss.item()) * count
            valid_count += count
            with torch.no_grad():
                error = (pred.float() - target_bt.float()).square()
                valid_float = target_valid.float()
                channel_error_sum += (
                    (error * valid_float).sum(dim=(0, 1)).cpu().double()
                )
                channel_valid_count += (
                    valid_float.sum(dim=(0, 1)).cpu().double()
                )
            progress.set_postfix(loss=f"{loss.item():.5f}")
            if max_steps is not None and step >= max_steps:
                break

        train_loss = squared_error_sum / max(valid_count, 1)
        train_channel_loss = (
            channel_error_sum / channel_valid_count.clamp_min(1.0)
        ).tolist()
        val_loss = None
        val_channel_loss = None
        improved = False
        should_stop = False
        if epoch % validate_every == 0 or epoch == epochs:
            val_base.set_epoch(0)  # Fixed validation sampling makes epochs comparable.
            reconstruction_path = reconstruction_dir / f"epoch_{epoch:04d}.pth"
            val_loss, val_channel_loss = run_validation(
                model, val_loader, device, reconstruction_path,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, epoch=epoch,
            )
            improved, should_stop = early_stopping.update(val_loss)
            if scheduler_name.lower() == "reduce_on_plateau":
                step_scheduler(scheduler, scheduler_name, val_loss)
        if scheduler_name.lower() == "step":
            step_scheduler(scheduler, scheduler_name)

        learning_rate = optimizer.param_groups[0]["lr"]
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})
        history["train_channel_loss"].append({
            "epoch": epoch, "loss": train_channel_loss,
        })
        if val_loss is not None:
            history["val_loss"].append({"epoch": epoch, "loss": val_loss})
            history["val_channel_loss"].append({
                "epoch": epoch, "loss": val_channel_loss,
            })
        history["learning_rate"].append({"epoch": epoch, "lr": learning_rate})
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "train_channel_loss": train_channel_loss,
            "val_loss": val_loss, "val_channel_loss": val_channel_loss,
            "learning_rate": learning_rate,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_loss_history(history_path, history)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'not_run'} "
            f"lr={learning_rate:.3e}"
        )
        print(f"train_channel_mse: {format_channel_losses(train_channel_loss)}")
        print(f"val_channel_mse:   {format_channel_losses(val_channel_loss)}")

        state = checkpoint_state(
            epoch, model, optimizer, scheduler, scaler, early_stopping,
            history, config, resolved, loader_generator,
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

    if test_enabled:
        best_path = run_dir / "best.pth"
        if best_path.exists():
            try:
                best_checkpoint = torch.load(
                    best_path, map_location=device, weights_only=False
                )
            except TypeError:
                best_checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(best_checkpoint["model"])
            print(f"full_test_model={best_path}")
        run_full_test(
            model=model,
            dataset=test_dataset,
            test_indices=test_indices,
            context_fractions=context_fractions,
            device=device,
            output_dir=test_output_dir,
            context_chunk_size=context_chunk_size,
            query_chunk_size=query_chunk_size,
            max_samples=test_max_samples,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            seed=seed,
        )


if __name__ == "__main__":
    main()
