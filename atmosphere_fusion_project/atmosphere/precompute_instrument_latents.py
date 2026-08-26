from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from atmosphere.config import load_yaml
from atmosphere.utils import amp_dtype_from_name, import_object


def _filter_dataclass_kwargs(cls, values):
    try:
        allowed = {f.name for f in fields(cls)}
    except TypeError:
        return values
    return {k: v for k, v in values.items() if k in allowed}


def load_model(cfg, device):
    model_cls = import_object(cfg["model_class"])
    config_cls = import_object(cfg["config_class"])
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu")
    model_cfg_raw = dict(checkpoint.get("config", {}))
    model_cfg_raw.update(cfg.get("model_config_override", {}))
    model_cfg = config_cls(**_filter_dataclass_kwargs(config_cls, model_cfg_raw))
    model = model_cls(model_cfg)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, model_cfg = load_model(cfg, device)

    dataset_cls = import_object(cfg["dataset_class"])
    dataset = dataset_cls(
        cfg["raw_zarr"], n_context=1, n_target=1, target_overlap=1.0, seed=0
    )
    raw_root = dataset._open()
    times = dataset._int64_time(raw_root["time_series"][:])
    t = len(times)
    d = int(model_cfg.latent_dim)
    h = int(model_cfg.grid_height)
    w = int(model_cfg.grid_width)

    import zarr
    from numcodecs import Blosc

    output = Path(cfg["output_zarr"])
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root.create_dataset("time", shape=(t,), chunks=(min(t, 1024),), dtype="i8", compressor=compressor)
    root.create_dataset(
        "latent", shape=(t, d, h, w), chunks=(1, d, min(h, 32), min(w, 64)),
        dtype="f4", compressor=compressor
    )
    root.create_dataset(
        "density", shape=(t, 1, h, w), chunks=(1, 1, min(h, 32), min(w, 64)),
        dtype="f4", compressor=compressor
    )
    root.create_dataset("available", shape=(t,), chunks=(min(t, 1024),), dtype="bool", compressor=compressor)
    root["time"][:] = times
    root.attrs.update({
        "instrument": cfg["instrument"],
        "source_raw_zarr": str(cfg["raw_zarr"]),
        "checkpoint": str(cfg["checkpoint"]),
        "latent_dim": d,
        "grid_height": h,
        "grid_width": w,
        "grid_resolution_deg": float(model_cfg.grid_resolution_deg),
    })

    chunk_size = int(cfg.get("encode_chunk_size", 65536))
    amp_enabled = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(cfg.get("amp_dtype", "float16"))
    channel_sum = torch.zeros(d, dtype=torch.float64)
    channel_sumsq = torch.zeros(d, dtype=torch.float64)
    scalar_count = 0

    with torch.no_grad():
        for i in tqdm(range(t), desc=f"precompute {cfg['instrument']}"):
            item = dataset.get_full_sample(i)
            obs = item["observations"]
            if item["count"] == 0:
                root["latent"][i] = 0
                root["density"][i] = 0
                root["available"][i] = False
                continue
            batch = {
                k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                for k, v in obs.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                latent, density = model.encode_chunked(
                    **batch, chunk_size=chunk_size
                )
            latent = latent.squeeze(0).float().cpu()
            density = density.squeeze(0).float().cpu()
            root["latent"][i] = latent.numpy()
            root["density"][i] = density.numpy()
            root["available"][i] = True
            channel_sum += latent.double().sum(dim=(1, 2))
            channel_sumsq += latent.double().square().sum(dim=(1, 2))
            scalar_count += h * w

    if scalar_count == 0:
        mean = torch.zeros(d, dtype=torch.float64)
        std = torch.ones(d, dtype=torch.float64)
    else:
        mean = channel_sum / scalar_count
        var = channel_sumsq / scalar_count - mean.square()
        std = var.clamp_min(1.0e-12).sqrt()
    root.create_dataset("latent_mean", data=mean.numpy(), dtype="f8", compressor=compressor)
    root.create_dataset("latent_std", data=std.numpy(), dtype="f8", compressor=compressor)
    print(f"wrote {output}")
    print(f"shape latent={(t, d, h, w)}")


if __name__ == "__main__":
    main()
