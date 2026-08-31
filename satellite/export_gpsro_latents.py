"""Export one GPSRO common 2-D latent for every non-empty six-hour bin."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from satellite.datasets import GPSROZarrDataset
from satellite.models import GPSROAutoEncoder
from satellite.train_gpsro import load_yaml, make_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode complete GPSRO 6h bins and save them to Zarr."
    )
    parser.add_argument("--config", required=True, help="GPSRO training YAML")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth file")
    parser.add_argument(
        "--input-zarr", help="Defaults to data.zarr in the training YAML"
    )
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--calculate-stats", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--standardize-latents", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-density-3d", action=argparse.BooleanOptionalAction, default=None,
        help="Also retain [T,1,Z,H,W] vertical coverage density",
    )
    parser.add_argument(
        "--output-resolution-deg", type=float,
        help=(
            "Optional output lon/lat resolution. For example 2.0 writes "
            "[91,180] so it can align with a 2-degree instrument store."
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_from_checkpoint(yaml_raw, checkpoint):
    """Use checkpoint model settings because they define every tensor shape."""
    saved = checkpoint.get("config")
    if not saved:
        return make_config(yaml_raw), yaml_raw
    if "model" in saved:
        saved_model = saved["model"]
    else:
        # Compatibility with checkpoints that stored only the model section.
        saved_model = saved
    resolved = dict(yaml_raw)
    resolved["model"] = dict(saved_model)
    return make_config(resolved), resolved


def load_model_state(model, checkpoint):
    state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state is None:
        state = checkpoint
    model.load_state_dict(state)


def _batch_to_device(points, device):
    return {
        name: value.unsqueeze(0).to(device, non_blocking=True)
        for name, value in points.items()
    }


def encode_one_sample(model, dataset, source_index, chunk_size, device,
                      amp_enabled, amp_dtype):
    """Aggregate a complete 6h bin without placing all observations on GPU."""
    root = dataset._open()
    sample_start = int(root["sample_start"][source_index])
    sample_count = int(root["sample_count"][source_index])
    sample_time = int(dataset._time_int64(root["time_series"][source_index]))
    latent_sum = density_sum = None
    satellite_counts = Counter()

    for offset in range(0, sample_count, chunk_size):
        end = min(offset + chunk_size, sample_count)
        indices = slice(sample_start + offset, sample_start + end)
        points = dataset._points(
            root, indices, sample_time, include_value=True
        )
        ids, counts = np.unique(
            points["satellite_id"].numpy(), return_counts=True
        )
        satellite_counts.update({
            int(identifier): int(count)
            for identifier, count in zip(ids, counts)
        })
        points = _batch_to_device(points, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            feature = model.point_encoder(
                points["refractivity"], points["valid"],
                points["satellite_id"], points["is_land"],
                points["obs_time"], points["sample_time"],
            )
            chunk_sum, chunk_density = model.point_to_grid.aggregate(
                feature, points["lon"], points["lat"], points["height"],
                vertical_type="altitude", vertical_unit="m",
                point_mask=points["valid"][:, :, 0],
            )
        latent_sum = chunk_sum if latent_sum is None else latent_sum + chunk_sum
        density_sum = (
            chunk_density if density_sum is None
            else density_sum + chunk_density
        )

    latent_3d = latent_sum / density_sum.clamp_min(model.config.eps)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        latent_2d = model.latent_processor(latent_3d)
    ids = np.asarray(sorted(satellite_counts), dtype=np.int64)
    counts = np.asarray([satellite_counts[int(i)] for i in ids], dtype=np.int64)
    return (
        latent_2d[0].float().cpu().numpy(),
        density_sum[0].float().cpu().numpy(),
        ids,
        counts,
    )


def resize_spatial(latent, density_3d, output_size):
    """Optionally align GPSRO H/W with another instrument before fusion."""
    if tuple(latent.shape[-2:]) == tuple(output_size):
        return latent, density_3d
    latent_tensor = torch.from_numpy(latent).unsqueeze(0)
    latent = F.interpolate(
        latent_tensor, size=output_size, mode="bilinear", align_corners=False
    )[0].numpy()
    channels = density_3d.shape[0] * density_3d.shape[1]
    density_tensor = torch.from_numpy(density_3d).reshape(
        1, channels, *density_3d.shape[-2:]
    )
    density_3d = F.interpolate(
        density_tensor, size=output_size, mode="bilinear", align_corners=False
    )[0].reshape(*density_3d.shape[:2], *output_size).numpy()
    return latent, density_3d


def _compressor():
    try:
        from numcodecs import Blosc
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        return None


def create_output(path, n_samples, config, output_height, output_width,
                  times, source_indices, sample_counts, checkpoint_path,
                  input_path, overwrite, save_density_3d):
    import zarr

    output_path = Path(path).resolve()
    input_path = Path(input_path).resolve()
    if output_path == input_path:
        raise ValueError("output-zarr must differ from input-zarr")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    root = zarr.open_group(str(output_path), mode="w")
    compressor = _compressor()
    common = {"compressor": compressor} if compressor is not None else {}
    time_chunk = min(max(n_samples, 1), 1024)
    root.create_dataset(
        "latent",
        shape=(n_samples, config.latent_dim, output_height, output_width),
        chunks=(1, config.latent_dim, output_height, output_width),
        dtype="f4", **common,
    )
    root.create_dataset(
        "density", shape=(n_samples, 1, output_height, output_width),
        chunks=(1, 1, output_height, output_width), dtype="f4", **common,
    )
    if save_density_3d:
        root.create_dataset(
            "density_3d",
            shape=(n_samples, 1, config.grid_depth, output_height, output_width),
            chunks=(1, 1, config.grid_depth, output_height, output_width),
            dtype="f4", **common,
        )
    for name in ("time", "time_series", "source_sample_index", "sample_count"):
        root.create_dataset(name, shape=(n_samples,), chunks=(time_chunk,), dtype="i8")
    root.create_dataset(
        "available", shape=(n_samples,), chunks=(time_chunk,), dtype="bool"
    )
    root.create_dataset(
        "satellite_id", shape=(n_samples,), chunks=(time_chunk,), dtype="i8"
    )
    root.create_dataset(
        "satellite_id_start", shape=(n_samples,), chunks=(time_chunk,), dtype="i8"
    )
    root.create_dataset(
        "satellite_id_unique_count", shape=(n_samples,),
        chunks=(time_chunk,), dtype="i8",
    )
    root.create_dataset(
        "satellite_id_values", shape=(0,), chunks=(1024,), dtype="i8"
    )
    root.create_dataset(
        "satellite_id_observation_count", shape=(0,), chunks=(1024,), dtype="i8"
    )
    root.create_dataset(
        "latent_mean", shape=(config.latent_dim,),
        chunks=(config.latent_dim,), dtype="f4",
    )
    root.create_dataset(
        "latent_std", shape=(config.latent_dim,),
        chunks=(config.latent_dim,), dtype="f4",
    )
    root.create_dataset(
        "latitude", shape=(output_height,), chunks=(output_height,), dtype="f4",
        data=np.linspace(-90.0, 90.0, output_height, dtype=np.float32),
    )
    root.create_dataset(
        "longitude", shape=(output_width,), chunks=(output_width,), dtype="f4",
        data=np.linspace(-180.0, 180.0, output_width + 1, dtype=np.float32)[:-1],
    )
    root.create_dataset(
        "vertical_geopotential_height_m",
        shape=(config.grid_depth,), chunks=(config.grid_depth,), dtype="f4",
        data=np.linspace(
            config.vertical_min_m, config.vertical_max_m,
            config.grid_depth, dtype=np.float32,
        ),
    )
    root["time"][:] = times
    root["time_series"][:] = times
    root["source_sample_index"][:] = source_indices
    root["sample_count"][:] = sample_counts
    root["available"][:] = True
    root["satellite_id"][:] = -1
    root["satellite_id_start"][:] = 0
    root["satellite_id_unique_count"][:] = 0
    root["latent_mean"][:] = 0.0
    root["latent_std"][:] = 1.0
    root.attrs.update({
        "instrument": "gpsro",
        "time_units": "nanoseconds since Unix epoch",
        "latent_definition": (
            "GPSRO SetConv3D latent with feature/height folded into channels, "
            "then processed by LatentGridProcessor"
        ),
        "density_definition": "sum of density_3d over the vertical grid axis",
        "density_3d_saved": bool(save_density_3d),
        "satellite_id_definition": (
            "dominant observation satellite_id in each 6h bin; complete "
            "per-bin IDs/counts use the satellite_id_start ragged-array schema"
        ),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source_zarr": str(input_path),
        "model_config": json.dumps(vars(config), sort_keys=True),
        "native_grid_height": int(config.grid_height),
        "native_grid_width": int(config.grid_width),
        "output_grid_height": int(output_height),
        "output_grid_width": int(output_width),
        "output_latitude_resolution_deg": 180.0 / max(output_height - 1, 1),
        "output_longitude_resolution_deg": 360.0 / output_width,
        "latent_stats_calculated": False,
        "latents_standardized": False,
        "export_complete": False,
    })
    return root


def append_satellite_ids(root, output_index, ids, counts, current_size):
    new_size = current_size + len(ids)
    root["satellite_id_values"].resize((new_size,))
    root["satellite_id_observation_count"].resize((new_size,))
    root["satellite_id_values"][current_size:new_size] = ids
    root["satellite_id_observation_count"][current_size:new_size] = counts
    root["satellite_id_start"][output_index] = current_size
    root["satellite_id_unique_count"][output_index] = len(ids)
    if len(ids):
        root["satellite_id"][output_index] = int(ids[np.argmax(counts)])
    return new_size


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    yaml_raw = load_yaml(args.config)
    export_raw = yaml_raw.get("export", {})
    calculate_stats = args.calculate_stats
    if calculate_stats is None:
        calculate_stats = bool(export_raw.get("calculate_stats", False))
    standardize = args.standardize_latents
    if standardize is None:
        standardize = bool(export_raw.get("standardize_latents", True))
    if standardize:
        calculate_stats = True
    save_density_3d = args.save_density_3d
    if save_density_3d is None:
        save_density_3d = bool(export_raw.get("save_density_3d", True))
    output_resolution = args.output_resolution_deg
    if output_resolution is None:
        output_resolution = export_raw.get("output_resolution_deg")
    if output_resolution is not None and float(output_resolution) <= 0:
        raise ValueError("output-resolution-deg must be positive")
    input_zarr = args.input_zarr or yaml_raw.get("data", {}).get("zarr")
    if not input_zarr:
        raise ValueError("Provide --input-zarr or data.zarr in the YAML")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config, resolved = config_from_checkpoint(yaml_raw, checkpoint)
    print("Loaded checkpoint model config:", resolved.get("model", {}))
    model = GPSROAutoEncoder(config).to(device)
    load_model_state(model, checkpoint)
    model.eval()

    dataset = GPSROZarrDataset(input_zarr)
    source = dataset._open()
    all_counts = np.asarray(source["sample_count"][:], dtype=np.int64)
    all_times = dataset._time_int64(source["time_series"][:])
    source_indices = np.flatnonzero(all_counts >= 2)
    source_indices = source_indices[
        np.argsort(all_times[source_indices], kind="stable")
    ]
    if args.max_samples is not None:
        source_indices = source_indices[:max(int(args.max_samples), 0)]
    if not len(source_indices):
        raise ValueError("No non-empty GPSRO 6h bins were selected")

    if output_resolution is None:
        output_height, output_width = config.grid_height, config.grid_width
    else:
        output_height = round(180.0 / float(output_resolution)) + 1
        output_width = round(360.0 / float(output_resolution))
    output = create_output(
        args.output_zarr, len(source_indices), config,
        output_height, output_width,
        all_times[source_indices].astype(np.int64),
        source_indices.astype(np.int64), all_counts[source_indices],
        args.checkpoint, input_zarr, args.overwrite, save_density_3d,
    )
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    print(
        f"device={device} samples={len(source_indices)} "
        f"latent_shape={output['latent'].shape} chunk_size={args.chunk_size} "
        f"calculate_stats={calculate_stats} standardize={standardize} "
        f"save_density_3d={save_density_3d}"
    )

    channel_sum = np.zeros(config.latent_dim, dtype=np.float64)
    channel_square_sum = np.zeros(config.latent_dim, dtype=np.float64)
    values_per_channel = 0
    satellite_value_count = 0
    with torch.inference_mode():
        for output_index, source_index in enumerate(tqdm(
            source_indices, desc="export GPSRO latents", unit="bin"
        )):
            latent, density_3d, satellite_ids, satellite_counts = encode_one_sample(
                model, dataset, int(source_index), args.chunk_size,
                device, amp_enabled, amp_dtype,
            )
            latent, density_3d = resize_spatial(
                latent, density_3d, (output_height, output_width)
            )
            density_2d = density_3d.sum(axis=1)
            for name, value in (
                ("latent", latent), ("density", density_2d),
                ("density_3d", density_3d),
            ):
                if not np.isfinite(value).all():
                    count = int((~np.isfinite(value)).sum())
                    raise FloatingPointError(
                        f"Non-finite {name} at source sample {source_index}: "
                        f"count={count}. "
                        + ("Retry without AMP." if amp_enabled else "Inspect inputs/model.")
                    )
            output["latent"][output_index] = latent
            output["density"][output_index] = density_2d
            if save_density_3d:
                output["density_3d"][output_index] = density_3d
            satellite_value_count = append_satellite_ids(
                output, output_index, satellite_ids, satellite_counts,
                satellite_value_count,
            )
            if calculate_stats:
                latent64 = latent.astype(np.float64, copy=False)
                channel_sum += latent64.sum(axis=(1, 2))
                channel_square_sum += np.square(latent64).sum(axis=(1, 2))
                values_per_channel += latent.shape[1] * latent.shape[2]

    if calculate_stats:
        mean = channel_sum / values_per_channel
        variance = channel_square_sum / values_per_channel - np.square(mean)
        std = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1.0e-6)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise FloatingPointError("Calculated latent statistics contain NaN/Inf")
        output["latent_mean"][:] = mean.astype(np.float32)
        output["latent_std"][:] = std.astype(np.float32)
        output.attrs["latent_stats_calculated"] = True
        output.attrs["latent_stats_element_count_per_channel"] = int(
            values_per_channel
        )
        if standardize:
            mean_grid = mean.astype(np.float32)[:, None, None]
            std_grid = std.astype(np.float32)[:, None, None]
            for index in tqdm(
                range(len(source_indices)), desc="standardize GPSRO latents",
                unit="bin",
            ):
                latent = np.asarray(output["latent"][index], dtype=np.float32)
                output["latent"][index] = (latent - mean_grid) / std_grid
            output.attrs["latents_standardized"] = True
            output.attrs["latent_standardization"] = (
                "stored_latent=(raw_latent-latent_mean)/latent_std"
            )
    output.attrs["export_complete"] = True
    print(f"latent={output['latent'].shape} dtype={output['latent'].dtype}")
    print(f"density={output['density'].shape} dtype={output['density'].dtype}")
    if save_density_3d:
        print(
            f"density_3d={output['density_3d'].shape} "
            f"dtype={output['density_3d'].dtype}"
        )
    print(
        f"time={output['time'].shape} available_true="
        f"{int(np.asarray(output['available'][:], dtype=bool).sum())}"
    )
    print(f"saved={Path(args.output_zarr).resolve()}")


if __name__ == "__main__":
    main()


# python -m satellite.export_gpsro_latents --config satellite/configs/gpsro_train.yaml --checkpoint "C:/Users/Lenovo/code/local_code/aardvark-weather-public/runs/gpsro_autoencoder/20260831_170515/best.pth" --output-zarr "F:/lyh_data/data_latent/gpsro_latents.zarr"  --calculate-stats --standardize-latents --overwrite