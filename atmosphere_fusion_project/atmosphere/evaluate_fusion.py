from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.losses import latent_mse
from atmosphere.train_fusion import (
    build_model, choose_device, load_checkpoint, move_batch
)
from atmosphere.utils import amp_dtype_from_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    test_cfg = cfg.get("test", {})
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"],
        rollout_steps=int(data_cfg.get("rollout_steps", 1)),
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    _, _, test_idx = dataset.split_chronological(
        float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
    )
    if not test_idx:
        raise ValueError("No test sequences")
    device = choose_device(train_cfg.get("device", "auto"))
    model = build_model(cfg, dataset).to(device)
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=int(test_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(test_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    loss_cfg = cfg.get("loss", {})
    use_density_mask = bool(loss_cfg.get("use_density_mask", False))
    threshold = float(loss_cfg.get("density_threshold", 1.0e-6))
    max_steps = test_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(train_cfg.get("amp_dtype", "bfloat16"))
    sums = {}
    counts = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="test"), 1):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                initial_latents = {n: batch["latents"][n][:, 0] for n in model.instrument_names}
                initial_density = {n: batch["densities"][n][:, 0] for n in model.instrument_names}
                initial_available = {n: batch["available"][n][:, 0] for n in model.instrument_names}
                output_shapes = model.spatial_shapes(initial_latents)
                state, _ = model.fuse(initial_latents, initial_density, initial_available)
                rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
                for lead in range(1, rollout_steps + 1):
                    state = model.forecast_state(state)
                    pred = model.decode_state(state, output_shapes)
                    for name in model.instrument_names:
                        loss = latent_mse(
                            pred[name]["latent"], batch["latents"][name][:, lead],
                            batch["available"][name][:, lead],
                            density=batch["densities"][name][:, lead],
                            use_density_mask=use_density_mask,
                            density_threshold=threshold,
                        )
                        key = f"{name}/lead_{lead}"
                        sums[key] = sums.get(key, 0.0) + float(loss)
                        counts[key] = counts.get(key, 0) + 1
            if max_steps is not None and batch_index >= max_steps:
                break
    metrics = {k: sums[k] / max(counts[k], 1) for k in sorted(sums)}
    output_path = Path(
        args.output or test_cfg.get("output", "fusion_test_metrics.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
