from __future__ import annotations

import argparse
from pathlib import Path

import torch

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.train_fusion import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", default="latent_forecast.pth")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    # Only the initial time is needed here; rollout length is controlled by --steps.
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"], rollout_steps=0,
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    item = dataset[args.sample_index]
    device = torch.device(cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, dataset).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    latents = {n: item["latents"][n][:1].to(device) for n in model.instrument_names}
    densities = {n: item["densities"][n][:1].to(device) for n in model.instrument_names}
    available = {n: item["available"][n][:1].to(device) for n in model.instrument_names}
    interval_ns = dataset.interval_ns
    start_time = int(item["time"][0])
    output = {"start_time": start_time, "steps": []}
    with torch.no_grad():
        state, weights = model.fuse(latents, densities, available)
        for lead in range(1, args.steps + 1):
            state = model.forecast_state(state)
            decoded = model.decode_state(state)
            record = {"time": start_time + lead * interval_ns, "instruments": {}}
            for name in model.instrument_names:
                latent_norm = decoded[name]["latent"]
                latent_raw = dataset.denormalize(name, latent_norm).squeeze(0).cpu()
                density = torch.expm1(decoded[name]["log_density"].clamp(max=20)).squeeze(0).cpu()
                record["instruments"][name] = {
                    "latent": latent_raw,
                    "density": density,
                }
            output["steps"].append(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
