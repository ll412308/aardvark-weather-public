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
from atmosphere.plotting import save_latent_comparison, save_loss_plot
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


def configure_training_mode(model, mode):
    """Choose joint training or one of the two separate training stages."""
    mode = str(mode).lower()
    valid_modes = {"joint", "fusion_reconstruction", "forecast"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown train.training_mode={mode!r}; use one of "
            f"{sorted(valid_modes)}"
        )

    # Start from a known state, which is important when this function is reused.
    model.requires_grad_(True)
    if mode == "fusion_reconstruction":
        # Train instrument adapters + cross-instrument fusion + instrument heads.
        # The forecast processor is present for checkpoint compatibility but frozen.
        model.processor.requires_grad_(False)
    elif mode == "forecast":
        # The fusion reconstruction system has already been trained. Its frozen
        # heads still transmit gradients with respect to the forecast state, so
        # the processor can learn through the latent forecast loss.
        model.requires_grad_(False)
        model.processor.requires_grad_(True)
    return mode


def save_model_parameter_report(model, output_path):
    """Print model size and save a detailed parameter report as JSON."""
    mib = 1024 ** 2
    parameters = []
    buffers = []
    modules = {}
    dtypes = {}

    def module_stats(name):
        # Group parameters by the first module name, for example adapters/fusion.
        group = name.split(".", 1)[0]
        return modules.setdefault(group, {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "parameter_size_bytes": 0,
            "buffer_elements": 0,
            "buffer_size_bytes": 0,
        })

    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        size_bytes = count * int(parameter.element_size())
        trainable = bool(parameter.requires_grad)
        dtype = str(parameter.dtype).replace("torch.", "")
        parameters.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": dtype,
            "numel": count,
            "trainable": trainable,
            "size_bytes": size_bytes,
        })
        stats = module_stats(name)
        stats["total_parameters"] += count
        stats["parameter_size_bytes"] += size_bytes
        key = "trainable_parameters" if trainable else "frozen_parameters"
        stats[key] += count
        dtype_stats = dtypes.setdefault(dtype, {
            "parameter_count": 0, "size_bytes": 0,
        })
        dtype_stats["parameter_count"] += count
        dtype_stats["size_bytes"] += size_bytes

    for name, buffer in model.named_buffers():
        count = int(buffer.numel())
        size_bytes = count * int(buffer.element_size())
        buffers.append({
            "name": name,
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype).replace("torch.", ""),
            "numel": count,
            "size_bytes": size_bytes,
        })
        stats = module_stats(name)
        stats["buffer_elements"] += count
        stats["buffer_size_bytes"] += size_bytes

    total = sum(item["numel"] for item in parameters)
    trainable = sum(
        item["numel"] for item in parameters if item["trainable"]
    )
    frozen = total - trainable
    parameter_bytes = sum(item["size_bytes"] for item in parameters)
    trainable_bytes = sum(
        item["size_bytes"] for item in parameters if item["trainable"]
    )
    buffer_elements = sum(item["numel"] for item in buffers)
    buffer_bytes = sum(item["size_bytes"] for item in buffers)

    for stats in modules.values():
        stats["parameter_size_mib"] = stats["parameter_size_bytes"] / mib
        stats["buffer_size_mib"] = stats["buffer_size_bytes"] / mib
    for stats in dtypes.values():
        stats["size_mib"] = stats["size_bytes"] / mib

    # Adam/AdamW usually stores a gradient and two moment tensors per trainable
    # parameter. This estimate deliberately excludes activations and batches.
    estimated_adam_bytes = parameter_bytes + buffer_bytes + 3 * trainable_bytes
    report = {
        "model_class": model.__class__.__name__,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_fraction": trainable / max(total, 1),
        "parameter_size_bytes": parameter_bytes,
        "parameter_size_mib": parameter_bytes / mib,
        "trainable_parameter_size_mib": trainable_bytes / mib,
        "buffer_elements": buffer_elements,
        "buffer_size_bytes": buffer_bytes,
        "buffer_size_mib": buffer_bytes / mib,
        "model_state_size_mib": (parameter_bytes + buffer_bytes) / mib,
        "estimated_adam_model_gradient_optimizer_mib": estimated_adam_bytes / mib,
        "size_note": (
            "model_state_size_mib contains parameters and buffers only. The Adam "
            "estimate adds gradients and two optimizer moments, but excludes "
            "activations, input batches, CUDA workspace and AMP overhead."
        ),
        "modules": modules,
        "dtypes": dtypes,
        "parameters": parameters,
        "buffers": buffers,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("model_parameter_summary:")
    print(
        f"  total={total:,} trainable={trainable:,} frozen={frozen:,} "
        f"trainable_fraction={report['trainable_fraction']:.2%}"
    )
    print(
        f"  parameters={report['parameter_size_mib']:.2f} MiB "
        f"buffers={report['buffer_size_mib']:.2f} MiB "
        f"model_state={report['model_state_size_mib']:.2f} MiB"
    )
    print(
        "  estimated_adam_model_gradient_optimizer="
        f"{report['estimated_adam_model_gradient_optimizer_mib']:.2f} MiB "
        "(activations not included)"
    )
    for name, stats in modules.items():
        print(
            f"  [{name}] trainable={stats['trainable_parameters']:,} / "
            f"total={stats['total_parameters']:,} "
            f"size={stats['parameter_size_mib']:.2f} MiB"
        )
    print(f"model_parameter_report={output_path}")
    return report


def step_loss(model, batch, cfg, training=True):
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})
    reconstruction_weight = float(
        loss_cfg.get("current_reconstruction_weight", 0.25)
    )
    forecast_weight = float(loss_cfg.get("forecast_weight", 1.0))
    training_mode = str(train_cfg.get("training_mode", "joint")).lower()
    train_reconstruction = training_mode in {"joint", "fusion_reconstruction"}
    train_forecast = training_mode in {"joint", "forecast"}
    use_density_mask = bool(loss_cfg.get("use_density_mask", False))
    density_threshold = float(loss_cfg.get("density_threshold", 1.0e-6))
    dropout = (
        float(train_cfg.get("instrument_dropout", 0.0)) if training else 0.0
    )

    input_available = {
        name: batch["available"][name][:, 0]   # 取batch的第一个时间步的可用性
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
    if train_reconstruction and reconstruction_weight > 0:
        current = model.decode_state(state, output_shapes)
        current_instrument_count = 0
        for name in model.instrument_names:
            target_available = batch["available"][name][:, 0]
            # An instrument may be completely absent at this time in the whole
            # batch. latent_mse would correctly return zero, but dividing that
            # zero by the fixed total number of configured instruments would
            # artificially shrink the losses (and gradients) of present ones.
            if not target_available.any():
                continue
            current_latent_loss = current_latent_loss + latent_mse(
                current[name]["latent"], batch["latents"][name][:, 0],
                target_available, density=batch["densities"][name][:, 0],
                use_density_mask=use_density_mask,
                density_threshold=density_threshold,
            )
            current_instrument_count += 1
        # Give equal weight only to instruments that really have a target.
        # If all instruments are absent, this term remains zero and is skipped.
        if current_instrument_count > 0:
            current_latent_loss = (
                current_latent_loss / current_instrument_count
            )
            total = total + reconstruction_weight * current_latent_loss

    future_latent_loss = state.new_zeros(())
    rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
    valid_rollout_count = 0
    forecast_range = (
        range(1, rollout_steps + 1) if train_forecast else range(0)
    )
    for rollout_step in forecast_range:
        state = model.forecast_state(state)
        prediction = model.decode_state(state, output_shapes)
        step_latent_loss = state.new_zeros(())
        step_instrument_count = 0
        for name in model.instrument_names:
            target_available = batch["available"][name][:, rollout_step]
            # Skip an absent target instead of counting it in the instrument
            # average. A partially available batch is still handled inside
            # latent_mse by its per-sample available mask.
            if not target_available.any():
                continue
            step_latent_loss = step_latent_loss + latent_mse(
                prediction[name]["latent"],
                batch["latents"][name][:, rollout_step],
                target_available,
                density=batch["densities"][name][:, rollout_step],
                use_density_mask=use_density_mask,
                density_threshold=density_threshold,
            )
            step_instrument_count += 1
        if step_instrument_count > 0:
            future_latent_loss = (
                future_latent_loss
                + step_latent_loss / step_instrument_count
            )
            valid_rollout_count += 1

    # Average only over future times with at least one valid target. Otherwise
    # a completely missing future time would silently reduce the forecast loss.
    if train_forecast and valid_rollout_count > 0:
        future_latent_loss = future_latent_loss / valid_rollout_count  # 可以调整rollout的平均方式
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
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {desc} loss at step={step}: {terms}. "
                "Check latent Zarr values/statistics and mixed-precision settings."
            )
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


@torch.no_grad()
def evaluate_loss_by_lead(model, loader, device, cfg, max_steps=None,
                          amp_enabled=False, amp_dtype=torch.float16):
    """Evaluate each autoregressive lead separately, including lead 0."""
    model.eval()
    loss_cfg = cfg.get("loss", {})
    use_density_mask = bool(loss_cfg.get("use_density_mask", False))
    density_threshold = float(loss_cfg.get("density_threshold", 1.0e-6))
    interval_hours = int(cfg.get("data", {}).get("interval_hours", 6))
    sums, counts = {}, {}

    progress = tqdm(loader, desc="test by lead", unit="batch")
    for step, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        initial_latents = {
            name: batch["latents"][name][:, 0]
            for name in model.instrument_names
        }
        initial_densities = {
            name: batch["densities"][name][:, 0]
            for name in model.instrument_names
        }
        initial_available = {
            name: batch["available"][name][:, 0]
            for name in model.instrument_names
        }
        output_shapes = model.spatial_shapes(initial_latents)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            state, _ = model.fuse(
                initial_latents, initial_densities, initial_available
            )
            rollout_steps = (
                next(iter(batch["latents"].values())).shape[1] - 1
            )
            for lead in range(rollout_steps + 1):
                if lead > 0:
                    # Reuse the model's own previous state: no teacher forcing.
                    state = model.forecast_state(state)
                prediction = model.decode_state(state, output_shapes)
                lead_loss = state.new_zeros(())
                instrument_count = 0
                for name in model.instrument_names:
                    available = batch["available"][name][:, lead]
                    if not available.any():
                        continue
                    lead_loss = lead_loss + latent_mse(
                        prediction[name]["latent"],
                        batch["latents"][name][:, lead],
                        available,
                        density=batch["densities"][name][:, lead],
                        use_density_mask=use_density_mask,
                        density_threshold=density_threshold,
                    )
                    instrument_count += 1
                if instrument_count > 0:
                    value = float(
                        (lead_loss / instrument_count).detach().item()
                    )
                    if not np.isfinite(value):
                        raise FloatingPointError(
                            f"Non-finite test loss at lead={lead} step={step}"
                        )
                    sums[lead] = sums.get(lead, 0.0) + value
                    counts[lead] = counts.get(lead, 0) + 1
        if max_steps is not None and step >= max_steps:
            break

    return {
        f"lead_{lead:02d}": {
            "lead_steps": lead,
            "lead_hours": lead * interval_hours,
            "latent_mse": sums[lead] / max(counts[lead], 1),
            "evaluated_batches": counts[lead],
        }
        for lead in sorted(sums)
    }


@torch.no_grad()
def save_sequence_comparisons(model, loader, dataset, device, epoch,
                              output_dir, plot_cfg, amp_enabled=False,
                              amp_dtype=torch.float16):
    """Plot lead 0 and forecast leads for the first validation/test batch."""
    model.eval()
    try:
        batch = move_batch(next(iter(loader)), device)
    except StopIteration:
        return []
    requested_names = plot_cfg.get("instruments")
    names = (
        model.instrument_names if requested_names is None
        else [name for name in requested_names if name in model.instrument_names]
    )
    channels = [int(value) - 1 for value in plot_cfg.get("channels", [1])]
    max_future_steps = plot_cfg.get("max_future_steps")
    max_future_steps = (
        None if max_future_steps is None else int(max_future_steps)
    )

    initial_latents = {
        name: batch["latents"][name][:, 0] for name in model.instrument_names
    }
    initial_densities = {
        name: batch["densities"][name][:, 0] for name in model.instrument_names
    }
    initial_available = {
        name: batch["available"][name][:, 0] for name in model.instrument_names
    }
    output_shapes = model.spatial_shapes(initial_latents)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        state, _ = model.fuse(
            initial_latents, initial_densities, initial_available
        )
        # Lead 0 uses only instrument adapters -> fusion -> instrument heads.
        # It does not pass through forecast_state/Swin.
        predictions = [
            ("forecast_lead_00", model.decode_state(state, output_shapes), 0)
        ]
        rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
        if max_future_steps is not None:
            rollout_steps = min(rollout_steps, max_future_steps)
        for lead in range(1, rollout_steps + 1):
            state = model.forecast_state(state)
            predictions.append(
                (f"forecast_lead_{lead:02d}",
                 model.decode_state(state, output_shapes), lead)
            )

    saved = []
    for phase, prediction, target_step in predictions:
        for name in names:
            available = batch["available"][name][:, target_step]
            available_indices = torch.nonzero(available, as_tuple=False).flatten()
            if available_indices.numel() == 0:
                continue
            batch_index = int(available_indices[0])
            pred = dataset.denormalize(name, prediction[name]["latent"])
            target = dataset.denormalize(
                name, batch["latents"][name][:, target_step]
            )
            for channel in channels:
                if channel < 0 or channel >= pred.shape[1]:
                    print(
                        f"plot_skip={name} latent channel {channel + 1} "
                        f"is outside [1, {pred.shape[1]}]"
                    )
                    continue
                saved.append(save_latent_comparison(
                    target=target[batch_index, channel],
                    prediction=pred[batch_index, channel],
                    output_dir=output_dir,
                    epoch=epoch,
                    instrument=name,
                    phase=phase,
                    channel=channel,
                ))
    return saved


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

    def cpu_byte_tensor(value):
        if torch.is_tensor(value):
            return value.detach().to(device="cpu", dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    loader_generator.set_state(cpu_byte_tensor(state["loader_generator"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([
            cpu_byte_tensor(cuda_state) for cuda_state in state["cuda"]
        ])


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
        "--test-only", action="store_true",
        help="Load a checkpoint, skip training/validation, and run test only",
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto",
        help="Checkpoint path, or omit the path to find the latest checkpoint",
    )
    parser.add_argument(
        "--finetune-from",
        help=(
            "Load model weights from this checkpoint, but start a new optimizer, "
            "scheduler, epoch counter and run directory"
        ),
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    early_cfg = cfg.get("early_stopping", {})
    test_cfg = cfg.get("test", {})
    plot_cfg = cfg.get("plot", {})
    output_cfg = cfg.get("output", {})
    test_only = bool(args.test_only or cfg.get("test_only", False))
    test_enabled = bool(test_cfg.get("enabled", True)) or test_only
    pin_memory = bool(train_cfg.get("pin_memory", True))

    seed = int(train_cfg.get("seed", 42))
    deterministic = bool(train_cfg.get("deterministic", False))
    seed_everything(seed, deterministic)
    train_rollout_steps = int(data_cfg.get("rollout_steps", 1))
    training_mode = str(train_cfg.get("training_mode", "joint")).lower()
    if training_mode == "forecast" and train_rollout_steps < 1:
        raise ValueError(
            "Forecast-only training requires data.rollout_steps >= 1"
        )
    reconstruction_weight = float(
        cfg.get("loss", {}).get("current_reconstruction_weight", 0.25)
    )
    forecast_weight = float(cfg.get("loss", {}).get("forecast_weight", 1.0))
    if training_mode == "fusion_reconstruction" and reconstruction_weight <= 0:
        raise ValueError(
            "fusion_reconstruction requires current_reconstruction_weight > 0"
        )
    if training_mode == "forecast" and forecast_weight <= 0:
        raise ValueError("forecast mode requires forecast_weight > 0")
    if training_mode == "joint" and reconstruction_weight <= 0 and forecast_weight <= 0:
        raise ValueError("joint mode requires at least one positive loss weight")
    test_rollout_steps = int(
        test_cfg.get("rollout_steps", train_rollout_steps)
    )
    if test_rollout_steps < 0:
        raise ValueError("test.rollout_steps must be at least zero")
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"],
        rollout_steps=train_rollout_steps,
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    train_indices, val_indices, training_test_indices = dataset.split_chronological(
        float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
    )
    if test_rollout_steps == train_rollout_steps:
        test_dataset = dataset
        test_indices = training_test_indices
    else:
        # Testing may use a longer target sequence than training. The model can
        # recursively reuse forecast_state without any architectural change.
        test_dataset = MultiInstrumentLatentSequenceDataset(
            stores=data_cfg["instruments"],
            rollout_steps=test_rollout_steps,
            interval_hours=int(data_cfg.get("interval_hours", 6)),
            normalize_latents=bool(data_cfg.get("normalize_latents", True)),
        )
        _, _, test_indices = test_dataset.split_chronological(
            float(data_cfg.get("val_fraction", 0.1)),
            float(data_cfg.get("test_fraction", 0.1)),
        )
    if not train_indices or not val_indices or (test_enabled and not test_indices):
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
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True,
        generator=loader_generator, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=int(test_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(test_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    model = build_model(cfg, dataset).to(device)
    training_mode = configure_training_mode(model, training_mode)
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
    max_test_steps = test_cfg.get("max_steps")
    max_test_steps = int(max_test_steps) if max_test_steps is not None else None
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
    finetune_from = (
        args.finetune_from
        if args.finetune_from is not None
        else train_cfg.get("finetune_from")
    )
    if resume and finetune_from:
        raise ValueError(
            "Use only one of train.resume and train.finetune_from. "
            "resume restores the complete training state; finetune_from loads "
            "model weights only and starts a new run."
        )
    finetune_path = Path(finetune_from) if finetune_from else None
    if finetune_path is not None and not finetune_path.exists():
        raise FileNotFoundError(
            f"Fine-tuning checkpoint not found: {finetune_path}"
        )
    if test_only and not resume:
        resume = "auto"
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
    if test_only and resume_path is None:
        raise FileNotFoundError(
            "test_only requires an existing checkpoint. Set train.resume to "
            "best.pth/epoch_XXXX.pth, or make sure latest.pth can be found."
        )
    if (
        training_mode == "forecast"
        and not test_only
        and finetune_path is None
        and resume_path is None
    ):
        raise ValueError(
            "Separate forecast training requires train.finetune_from to point "
            "to the fusion-reconstruction best.pth (or train.resume for an "
            "already-started forecast run)."
        )

    if resume_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_path.parent
    log_path = run_dir / "loss_log.jsonl"
    history_path = run_dir / "loss_history.json"
    plot_dir = run_dir / "plots"
    loss_plot_dir = plot_dir / "loss"
    validation_plot_dir = plot_dir / "validation"
    test_plot_dir = plot_dir / "test"
    test_output_dir = run_dir / "test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    plot_enabled = bool(plot_cfg.get("enabled", True))
    if plot_enabled:
        loss_plot_dir.mkdir(parents=True, exist_ok=True)
        validation_plot_dir.mkdir(parents=True, exist_ok=True)
        test_plot_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / (
        "resolved_config_resume.json" if resume_path else "resolved_config.json"
    )
    resolved_cfg = dict(cfg)
    resolved_cfg["test_only"] = test_only
    resolved_path.write_text(
        json.dumps(resolved_cfg, indent=2), encoding="utf-8"
    )

    history = {"train": [], "val": [], "learning_rate": []}
    start_epoch = 1
    model_epoch = 0
    if finetune_path is not None:
        # Fine-tuning intentionally starts a new optimizer, scheduler, epoch
        # counter, early-stopping state and loss history from the YAML settings.
        checkpoint = load_checkpoint(finetune_path, device)
        model.load_state_dict(checkpoint["model"])
        print(
            f"finetune_from={finetune_path} source_epoch="
            f"{checkpoint.get('epoch', 'unknown')} start_epoch=1"
        )
    elif resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"])
        model_epoch = int(checkpoint["epoch"])
        if test_only:
            # Test only needs model weights. Do not alter optimizer, scheduler,
            # RNG, history or the existing training log.
            start_epoch = epochs + 1
            print(f"test_only_checkpoint={resume_path}")
        else:
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

    save_model_parameter_report(model, run_dir / "model_parameters.json")
    print(f"run_dir={run_dir}")
    print(f"test_only={test_only} test_enabled={test_enabled}")
    print(f"training_mode={training_mode}")
    print(
        f"train_rollout_steps={train_rollout_steps} "
        f"test_rollout_steps={test_rollout_steps}"
    )
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
        if plot_enabled and bool(plot_cfg.get("loss_enabled", True)):
            loss_plot = save_loss_plot(
                history, loss_plot_dir, epoch,
                log_scale=bool(plot_cfg.get("loss_log_scale", True)),
            )
            print(f"loss_plot={loss_plot}")
        if (
            val_metrics is not None
            and plot_enabled
            and bool(plot_cfg.get("validation_enabled", True))
        ):
            comparison_paths = save_sequence_comparisons(
                model=model,
                loader=val_loader,
                dataset=dataset,
                device=device,
                epoch=epoch,
                output_dir=validation_plot_dir / f"epoch_{epoch:04d}",
                plot_cfg=plot_cfg,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            print(f"validation_plots={len(comparison_paths)}")
        print(json.dumps(record))

        state = checkpoint_state(
            epoch, model, optimizer, scheduler, scaler,
            early_stopping, history, cfg, loader_generator,
        )
        if improved:
            torch.save(state, run_dir / "best.pth")
        model_epoch = epoch
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
        # In test-only mode, keep the exact resume checkpoint already loaded.
        # After normal training, evaluate the checkpoint selected by validation.
        test_model_path = resume_path if test_only else None
        best_path = run_dir / "best.pth"
        if not test_only and best_path.exists():
            best_checkpoint = load_checkpoint(best_path, device)
            model.load_state_dict(best_checkpoint["model"])
            model_epoch = int(best_checkpoint["epoch"])
            test_model_path = best_path
        print(f"test_model={test_model_path or 'current_in_memory_model'}")

        with torch.no_grad():
            test_metrics = run_epoch(
                model, test_loader, device, cfg,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_steps=max_test_steps,
                desc="test",
            )
        lead_metrics = evaluate_loss_by_lead(
            model=model,
            loader=test_loader,
            device=device,
            cfg=cfg,
            max_steps=max_test_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        test_record = {
            "checkpoint": (
                str(test_model_path) if test_model_path is not None else None
            ),
            "checkpoint_epoch": model_epoch,
            "sequence_count": len(test_indices),
            "evaluated_batches": (
                len(test_loader) if max_test_steps is None
                else min(len(test_loader), max_test_steps)
            ),
            "metrics": test_metrics,
            "loss_by_lead": lead_metrics,
        }
        test_metrics_path = test_output_dir / "test_metrics.json"
        test_metrics_path.write_text(
            json.dumps(test_record, indent=2), encoding="utf-8"
        )
        print(f"test_metrics={json.dumps(test_metrics)}")
        print(f"test_loss_by_lead={json.dumps(lead_metrics)}")
        print(f"test_metrics_path={test_metrics_path}")

        if (
            plot_enabled
            and bool(plot_cfg.get("test_enabled", True))
        ):
            test_comparison_paths = save_sequence_comparisons(
                model=model,
                loader=test_loader,
                dataset=test_dataset,
                device=device,
                epoch=model_epoch,
                output_dir=test_plot_dir,
                plot_cfg=plot_cfg,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            print(f"test_plots={len(test_comparison_paths)}")


if __name__ == "__main__":
    main()
