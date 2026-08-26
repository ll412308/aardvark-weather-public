from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.losses import latent_coverage_mse
from atmosphere.train_fusion import build_model, move_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="fusion_test_metrics.json")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
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
    device = torch.device(cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, dataset).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False, num_workers=0)
    threshold = float(cfg.get("loss", {}).get("density_threshold", 1.0e-6))
    sums = {}
    counts = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            batch = move_batch(batch, device)
            initial_latents = {n: batch["latents"][n][:, 0] for n in model.instrument_names}
            initial_density = {n: batch["densities"][n][:, 0] for n in model.instrument_names}
            initial_available = {n: batch["available"][n][:, 0] for n in model.instrument_names}
            state, _ = model.fuse(initial_latents, initial_density, initial_available)
            rollout_steps = next(iter(batch["latents"].values())).shape[1] - 1
            for lead in range(1, rollout_steps + 1):
                state = model.forecast_state(state)
                pred = model.decode_state(state)
                for name in model.instrument_names:
                    loss = latent_coverage_mse(
                        pred[name]["latent"], batch["latents"][name][:, lead],
                        batch["densities"][name][:, lead], batch["available"][name][:, lead], threshold
                    )
                    key = f"{name}/lead_{lead}"
                    sums[key] = sums.get(key, 0.0) + float(loss)
                    counts[key] = counts.get(key, 0) + 1
    metrics = {k: sums[k] / max(counts[k], 1) for k in sorted(sums)}
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
