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
from satellite.plotting import (
    decode_global_grid,
    save_global_reconstruction_plots,
    save_loss_plot,
    save_point_reconstruction_plots,
)
from satellite.scheduler import build_scheduler, step_scheduler


def to_device(mapping, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in mapping.items()
    }


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

def get_total_parameters(model,out_dir,name_save='model_parameters.json'):
    parameter_details = []

    for name, param in model.named_parameters():
        info = {
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "requires_grad": param.requires_grad,
        }
        parameter_details.append(info)

        if param.requires_grad:
            print(
                f"{name:60s} "
                f"shape={str(tuple(param.shape)):25s} "
                f"numel={param.numel():,}"
            )

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    model_parameter_log = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "parameters": parameter_details,
    }

    (out_dir / name_save).write_text(
        json.dumps(model_parameter_log, indent=2),
        encoding="utf-8",
    )
    print(f"Model parameters saved to: {out_dir / name_save}")

    
def restore_random_state(state, loader_generator):
    if not state:
        return

    def cpu_byte_tensor(value):
        if torch.is_tensor(value):
            return value.detach().to(device="cpu", dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") also moves RNG tensors to CUDA, while
    # these APIs specifically require CPU ByteTensor states.
    torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    loader_generator.set_state(cpu_byte_tensor(state["loader_generator"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([
            cpu_byte_tensor(cuda_state) for cuda_state in state["cuda"]
        ])


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


def load_bt_plot_stats(zarr_path, n_channels):
    """Read the transform needed to plot decoder outputs in physical BT units."""
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    description = str(root.attrs.get("brightness_temperature", "")).lower()
    standardized = any(
        token in description for token in ("standard", "z-score", "zscore")
    )
    if "channel_mean" not in root or "channel_std" not in root:
        if standardized:
            raise ValueError(
                "BT is standardized but channel_mean/channel_std are missing"
            )
        return False, None, None
    mean = np.asarray(root["channel_mean"][:], dtype=np.float32)
    std = np.asarray(root["channel_std"][:], dtype=np.float32)
    if mean.shape != (n_channels,) or std.shape != (n_channels,):
        raise ValueError(
            f"Expected BT stats [{n_channels}], got mean={mean.shape}, std={std.shape}"
        )
    return standardized, mean.tolist(), np.maximum(std, 1.0e-6).tolist()


def split_datasets_by_time(zarr_path, config, val_fraction, test_fraction,
                           sampling_seed):
    """Use early non-empty 6h bins for train and later bins for validation/test."""
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
    counts = np.asarray(root["sample_count"][:], dtype=np.int64)
    nonempty = np.flatnonzero(counts >= 2)
    skipped = len(counts) - len(nonempty)
    if skipped:
        print(f"skipping_empty_or_tiny_bins={skipped} min_required_count=2")
    indices = nonempty[np.argsort(times[nonempty], kind="stable")].tolist()
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


def save_reconstruction_figures(model, reconstruction, output_dir, prefix,
                                plot_options, device, amp_enabled, amp_dtype):
    """Save point-comparison plots and an optional global decoded grid."""
    if not plot_options.get("enabled", True):
        return
    channels = plot_options.get("channels", [1])
    save_point_reconstruction_plots(
        lon=reconstruction["lon"],
        lat=reconstruction["lat"],
        truth=reconstruction["target_bt"],
        prediction=reconstruction["pred"],
        valid=reconstruction["target_valid"],
        satellite_id=reconstruction["satellite_id"],
        channels=channels,
        output_dir=output_dir,
        prefix=prefix,
        max_points=int(plot_options.get("max_points", 100_000)),
        point_size=float(plot_options.get("point_size", 3.0)),
        bt_standardized=plot_options.get("bt_standardized", False),
        bt_mean=plot_options.get("bt_mean"),
        bt_std=plot_options.get("bt_std"),
        color_std_range=plot_options.get("color_std_range", 3.0),
    )
    if not plot_options.get("global_enabled", True):
        return
    configured_satellite_id = plot_options.get("global_satellite_id")
    if configured_satellite_id is None:
        query_satellite_ids = torch.unique(
            torch.as_tensor(reconstruction["satellite_id"]).reshape(-1)
        ).tolist()
    else:
        query_satellite_ids = [int(configured_satellite_id)]
    for query_satellite_id in query_satellite_ids:
        lon_grid, lat_grid, global_prediction = decode_global_grid(
            model=model,
            latent=reconstruction["latent"].to(device),
            sample_time=reconstruction["sample_time"],
            satellite_id=query_satellite_id,
            is_land=bool(plot_options.get("global_is_land", 0)),
            resolution_deg=float(plot_options.get("global_resolution_deg", 2.0)),
            chunk_size=int(plot_options.get("global_query_chunk_size", 16_384)),
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        save_global_reconstruction_plots(
            lon_grid, lat_grid, global_prediction, channels,
            output_dir=output_dir, prefix=prefix,
            satellite_id=query_satellite_id,
            bt_standardized=plot_options.get("bt_standardized", False),
            bt_mean=plot_options.get("bt_mean"),
            bt_std=plot_options.get("bt_std"),
            color_std_range=plot_options.get("color_std_range", 3.0),
        )


def full_test_sample(model, dataset, sample_index, context_fractions, device,
                     output_dir, context_chunk_size, query_chunk_size,
                     amp_enabled=False, amp_dtype=torch.float16, seed=0,
                     plot_options=None):
    """Reconstruct every query point using several context percentages."""
    item = dataset.get_full_sample(sample_index)
    observations = item["observations"]
    n_points = item["count"]
    if n_points < 1:
        print(f"skip_test_sample={sample_index} count={n_points}")
        return []
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
                query_lon = observations["lon"][start:end].unsqueeze(0).to(device)
                query_lat = observations["lat"][start:end].unsqueeze(0).to(device)
                query_satellite_id = observations["satellite_id"][
                    start:end
                ].unsqueeze(0).to(device)
                query_is_land = observations["is_land"][
                    start:end
                ].unsqueeze(0).to(device)
                query_sample_time = observations["sample_time"].unsqueeze(0).to(
                    device
                )
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
                ):
                    pred = model.decode(
                        latent=latent,
                        lon=query_lon,
                        lat=query_lat,
                        satellite_id=query_satellite_id,
                        is_land=query_is_land,
                        sample_time=query_sample_time,
                    )
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
            all_predictions = torch.cat(predictions)
            torch.save({
                "pred": all_predictions,
                "context_indices": context_indices,
                "metrics": metrics,
            }, sample_dir / f"context_{percent:03d}_percent.pth")
            if plot_options and plot_options.get("test_enabled", True):
                save_reconstruction_figures(
                    model=model,
                    reconstruction={
                        "pred": all_predictions,
                        "target_bt": observations["bt"],
                        "target_valid": observations["valid"],
                        "lon": observations["lon"],
                        "lat": observations["lat"],
                        "satellite_id": observations["satellite_id"],
                        "sample_time": observations["sample_time"],
                        "latent": latent,
                    },
                    output_dir=sample_dir,
                    prefix=f"context_{percent:03d}_percent",
                    plot_options=plot_options,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
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
                  amp_dtype=torch.float16, seed=0, plot_options=None):
    selected = test_indices if max_samples is None else test_indices[:max_samples]
    all_results = []
    for sample_index in selected:
        all_results.extend(full_test_sample(
            model, dataset, sample_index, context_fractions, device,
            output_dir, context_chunk_size, query_chunk_size,
            amp_enabled, amp_dtype, seed, plot_options,
        ))
    (output_dir / "all_metrics.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    return all_results


def run_validation(model, loader, device, reconstruction_path=None,
                   amp_enabled=False, amp_dtype=torch.float16,
                   epoch=None, plot_output_dir=None, plot_options=None):
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
            target_bt = batch["target_bt"].to(device, non_blocking=True)
            target_valid = batch["target_valid"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                pred, latent, _ = model(context, target)
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
                    "satellite_id": target["satellite_id"].cpu(),
                    "sample_time": target["sample_time"].cpu(),
                    "latent": latent[:1].float().cpu(),
                }
    if reconstruction_path is not None and first_reconstruction is not None:
        torch.save(first_reconstruction, reconstruction_path)
    if (
        first_reconstruction is not None
        and plot_output_dir is not None
        and plot_options
        and plot_options.get("validation_enabled", True)
    ):
        first = {
            key: value[:1] if torch.is_tensor(value) and value.ndim > 0 else value
            for key, value in first_reconstruction.items()
        }
        # Point plotting expects [N,...], whereas latent retains its batch axis.
        for key in ("pred", "target_bt", "target_valid", "lon", "lat",
                    "satellite_id"):
            first[key] = first[key][0]
        save_reconstruction_figures(
            model=model,
            reconstruction=first,
            output_dir=plot_output_dir,
            prefix=f"validation_epoch_{int(epoch):04d}",
            plot_options=plot_options,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
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


def load_model_weights(model, checkpoint):
    state = dict(checkpoint["model"])
    if "latent_projection.weight" in state:
        state["latent_processor.input_projection.weight"] = state.pop(
            "latent_projection.weight"
        )
    if "latent_projection.bias" in state:
        state["latent_processor.input_projection.bias"] = state.pop(
            "latent_projection.bias"
        )
    # Migrate either previous decoder layout:
    #   [latent, density, metadata] -> [latent, metadata]
    #   [latent]                    -> [latent, newly initialised metadata]
    decoder_key = "point_decoder.mlp.0.weight"
    if decoder_key in state:
        expected = model.state_dict()[decoder_key]
        old = state[decoder_key]
        if old.shape != expected.shape and old.shape[0] == expected.shape[0]:
            latent_dim = model.config.latent_dim
            metadata_dim = model.config.metadata_dim
            if old.shape[1] == latent_dim + 1 + metadata_dim:
                state[decoder_key] = torch.cat([
                    old[:, :latent_dim],
                    old[:, latent_dim + 1:],
                ], dim=1)
            elif old.shape[1] == latent_dim:
                migrated = expected.clone()
                migrated[:, :latent_dim] = old
                state[decoder_key] = migrated
        if state[decoder_key].shape != expected.shape:
            # Unknown historical layout: keep the newly initialised layer.
            state.pop(decoder_key)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "checkpoint_model_keys: "
            f"missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    return result


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
    plot_raw = raw.get("plot", {})
    output_raw = raw.get("output", {})
    # It can be enabled either from the command line or from YAML:
    # test_only: true
    test_only = bool(args.test_only or raw.get("test_only", False))

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
    pin_memory = bool(train_raw.get("pin_memory", True))
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
    test_enabled = bool(test_raw.get("enabled", True)) or test_only
    context_fractions = [float(value) for value in
                         test_raw.get("context_fractions", [1.0, 0.9, 0.8])]
    if any(value <= 0 or value > 1 for value in context_fractions):
        raise ValueError("test.context_fractions values must be in (0, 1]")
    context_chunk_size = int(test_raw.get("context_chunk_size", 16_384))
    query_chunk_size = int(test_raw.get("query_chunk_size", 16_384))
    test_max_samples = test_raw.get("max_samples")
    if test_max_samples is not None:
        test_max_samples = int(test_max_samples)
    plot_options = {
        "enabled": bool(plot_raw.get("enabled", True)),
        "loss_enabled": bool(plot_raw.get("loss_enabled", True)),
        "loss_log_scale": bool(plot_raw.get("loss_log_scale", True)),
        "validation_enabled": bool(plot_raw.get("validation_enabled", True)),
        "test_enabled": bool(plot_raw.get("test_enabled", True)),
        "channels": [int(value) for value in plot_raw.get("channels", [1])],
        "max_points": int(plot_raw.get("max_points", 100_000)),
        "point_size": float(plot_raw.get("point_size", 3.0)),
        "global_enabled": bool(plot_raw.get("global_enabled", True)),
        "global_resolution_deg": float(
            plot_raw.get("global_resolution_deg", 2.0)
        ),
        "global_query_chunk_size": int(
            plot_raw.get("global_query_chunk_size", query_chunk_size)
        ),
        "global_satellite_id": plot_raw.get("global_satellite_id"),
        "global_is_land": int(plot_raw.get("global_is_land", 0)),
        "color_std_range": float(plot_raw.get("color_std_range", 3.0)),
    }
    bt_standardized, bt_mean, bt_std = load_bt_plot_stats(
        zarr_path, config.n_channels
    )
    plot_options.update({
        "bt_standardized": bt_standardized,
        "bt_mean": bt_mean,
        "bt_std": bt_std,
    })
    if not plot_options["channels"] or any(
        channel < 1 or channel > config.n_channels
        for channel in plot_options["channels"]
    ):
        raise ValueError(
            f"plot.channels must contain 1-based values in [1, {config.n_channels}]"
        )
    if plot_options["max_points"] < 0:
        raise ValueError("plot.max_points must be >= 0; use 0 for all points")
    if plot_options["global_resolution_deg"] <= 0:
        raise ValueError("plot.global_resolution_deg must be positive")
    if plot_options["color_std_range"] <= 0:
        raise ValueError("plot.color_std_range must be positive")
    run_name = pick(args.run_name, output_raw.get("run_name"), "bamua_autoencoder")
    runs_dir = Path(pick(args.runs_dir, output_raw.get("runs_dir"), "runs"))

    optimizer_name = optimizer_raw.get("name", "adamw")
    lr = pick(args.lr, optimizer_raw.get("lr", train_raw.get("lr")), 1.0e-3)
    weight_decay = float(optimizer_raw.get("weight_decay", 1.0e-4))
    scheduler_name = scheduler_raw.get("name", "reduce_on_plateau")
    resume = pick(args.resume, train_raw.get("resume"))
    if test_only and not resume:
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
    if test_only and resume_path is None:
        raise FileNotFoundError(
            "test_only requires an existing checkpoint. Set train.resume to a "
            "checkpoint path, or make sure the run contains latest.pth."
        )

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
    plot_dir = run_dir / "plots"
    loss_plot_dir = plot_dir / "loss"
    validation_plot_dir = plot_dir / "validation"
    if plot_options["enabled"]:
        loss_plot_dir.mkdir(parents=True, exist_ok=True)
        validation_plot_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "loss_log.jsonl"
    history_path = run_dir / "loss_history.json"

    resolved = {
        "test_only": test_only,
        "data": {**data_raw, "zarr": zarr_path},
        "model": vars(config),
        "train": {
            "epochs": epochs, "batch_size": batch_size,
            "num_workers": num_workers, "pin_memory": pin_memory,
            "max_steps": max_steps,
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
        "plot": plot_options,
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
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = BAMUAAutoEncoder(config).to(device)
    get_total_parameters(model,run_dir)

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
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    except (AttributeError, TypeError):  # Compatibility with older PyTorch.
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
        load_model_weights(model, checkpoint)
        if test_only:
            # Testing only needs model parameters. Do not alter optimizer/RNG/history
            # state or trim the existing training log.
            start_epoch = epochs + 1
            print(f"test_only_checkpoint={resume_path}")
        else:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as error:
                print(f"optimizer_state_not_loaded={error}")
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except ValueError as error:
                    print(f"scheduler_state_not_loaded={error}")
            if checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
            history = checkpoint.get("history", history)
            # Older checkpoints did not contain per-channel histories.
            history.setdefault("train_channel_loss", [])
            history.setdefault("val_channel_loss", [])
            restore_random_state(checkpoint.get("random_state"), loader_generator)
            start_epoch = int(checkpoint["epoch"]) + 1
            trim_log_after_epoch(log_path, int(checkpoint["epoch"]))
            print(f"resumed_from={resume_path} next_epoch={start_epoch}")

    print(f"run_dir={run_dir}")
    print(f"test_only={test_only}")
    print(
        f"device={device} mixed_precision={amp_enabled} amp_dtype={amp_dtype_name} "
        f"pin_memory={pin_memory} "
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
            target_bt = batch["target_bt"].to(device, non_blocking=True)
            target_valid = batch["target_valid"].to(device, non_blocking=True)
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
                plot_output_dir=validation_plot_dir / f"epoch_{epoch:04d}",
                plot_options=plot_options,
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
        if plot_options["enabled"] and plot_options["loss_enabled"]:
            loss_plot_path = save_loss_plot(
                history, loss_plot_dir, epoch,
                log_scale=plot_options["loss_log_scale"],
            )
            print(f"loss_plot={loss_plot_path}")
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
        if test_only:
            # The requested checkpoint was already loaded above. Keep it instead
            # of silently replacing it with best.pth from the same run folder.
            print(f"full_test_model={resume_path}")
        elif best_path.exists():
            try:
                best_checkpoint = torch.load(
                    best_path, map_location=device, weights_only=False
                )
            except TypeError:
                best_checkpoint = torch.load(best_path, map_location=device)
            load_model_weights(model, best_checkpoint)
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
            plot_options=plot_options,
        )


if __name__ == "__main__":
    main()
