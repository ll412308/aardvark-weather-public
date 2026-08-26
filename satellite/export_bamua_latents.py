"""Export one final 1BAMUA latent grid for every non-empty 6-hour bin."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from satellite.datasets import BAMUAZarrDataset
from satellite.models import BAMUAAutoEncoder
from satellite.train_bamua import (
    load_model_weights,
    load_yaml_config,
    make_bamua_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode complete 1BAMUA 6h bins and save them to Zarr."
    )
    parser.add_argument("--config", required=True, help="Training YAML file")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth file")
    parser.add_argument(
        "--input-zarr",
        help="Source 1bamua.zarr; defaults to data.zarr in the YAML file",
    )
    parser.add_argument("--output-zarr", required=True, help="Output Zarr path")
    parser.add_argument("--chunk-size", type=int, default=16_384)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--calculate-stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Calculate channel-wise latent mean/std over all exported times "
            "and grid cells. Defaults to export.calculate_stats in YAML, "
            "or false."
        ),
    )
    parser.add_argument(
        "--standardize-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write channel-wise standardized latent values. Defaults to "
            "export.standardize_latents in YAML, or true."
        ),
    )
    parser.add_argument(
        "--amp-dtype", default="float16", choices=("float16", "bfloat16")
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Only export the first N non-empty bins (useful for a smoke test)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace output-zarr if it already exists",
    )
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def add_batch_and_move(mapping, device):
    return {name: value.unsqueeze(0).to(device) for name, value in mapping.items()}


def encode_one_sample(model, dataset, sample_index, chunk_size, device,
                      amp_enabled, amp_dtype):
    """Read and aggregate one 6h bin without putting all points on the GPU."""
    root = dataset._open()
    sample_start = int(root["sample_start"][sample_index])
    sample_count = int(root["sample_count"][sample_index])
    sample_time = dataset._int64_time(root["time_series"][sample_index]).item()
    latent_sum = None
    density_sum = None

    for offset in range(0, sample_count, chunk_size):
        end = min(offset + chunk_size, sample_count)
        indices = slice(sample_start + offset, sample_start + end)
        points = dataset._common(root, indices, sample_time)
        points["bt"] = torch.from_numpy(
            dataset._read(root, "brightness_temperature", indices)
        ).float()
        points["valid"] = torch.from_numpy(
            dataset._read(root, "brightness_temperature_valid", indices)
        ).bool()
        points = add_batch_and_move(points, device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            feature = model.point_encoder(
                points["bt"],
                points["valid"],
                **model._metadata(
                    satellite_id=points["satellite_id"],
                    is_land=points["is_land"],
                    sample_time=points["sample_time"],
                    obs_time=points["obs_time"],
                    zenith=points["zenith"],
                    azimuth=points["azimuth"],
                ),
            )
            chunk_sum, chunk_density = model.point_to_grid.aggregate(
                feature, points["lon"], points["lat"]
            )
        latent_sum = chunk_sum if latent_sum is None else latent_sum + chunk_sum
        density_sum = (
            chunk_density
            if density_sum is None
            else density_sum + chunk_density
        )

    latent = latent_sum / density_sum.clamp_min(model.config.eps)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        latent = model.latent_processor(latent)
    return latent[0].float().cpu().numpy(), density_sum[0].float().cpu().numpy()


def create_output(path, n_samples, config, times, source_indices, counts,
                  checkpoint_path, input_path, overwrite):
    import zarr

    output_path = Path(path).resolve()
    input_path_resolved = Path(input_path).resolve()
    if output_path == input_path_resolved:
        raise ValueError("output-zarr must be different from input-zarr")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    root = zarr.open_group(str(output_path), mode="w")
    latent_shape = (
        n_samples, config.latent_dim, config.grid_height, config.grid_width
    )
    root.create_dataset(
        "latent", shape=latent_shape,
        chunks=(1, config.latent_dim, config.grid_height, config.grid_width),
        dtype="f4",
    )
    root.create_dataset(
        "density",
        shape=(n_samples, 1, config.grid_height, config.grid_width),
        chunks=(1, 1, config.grid_height, config.grid_width),
        dtype="f4",
    )
    root.create_dataset(
        "time_series", shape=(n_samples,), chunks=(min(n_samples, 1024),),
        dtype="i8",
    )
    root.create_dataset(
        "source_sample_index", shape=(n_samples,),
        chunks=(min(n_samples, 1024),), dtype="i8",
    )
    root.create_dataset(
        "sample_count", shape=(n_samples,), chunks=(min(n_samples, 1024),),
        dtype="i8",
    )
    root.create_dataset(
        "latent_mean", shape=(config.latent_dim,),
        chunks=(config.latent_dim,), dtype="f4",
    )
    root.create_dataset(
        "latent_std", shape=(config.latent_dim,),
        chunks=(config.latent_dim,), dtype="f4",
    )
    root["time_series"][:] = times
    root["source_sample_index"][:] = source_indices
    root["sample_count"][:] = counts
    # Identity normalisation is the default when statistics are not requested.
    root["latent_mean"][:] = np.zeros(config.latent_dim, dtype=np.float32)
    root["latent_std"][:] = np.ones(config.latent_dim, dtype=np.float32)
    root.attrs.update({
        "instrument": "1bamua",
        "time_units": "nanoseconds since Unix epoch",
        "latent_definition": "output of BAMUAAutoEncoder.latent_processor",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source_zarr": str(input_path_resolved),
        "model_config": json.dumps(vars(config), sort_keys=True),
        "latent_stats_calculated": False,
        "latents_standardized": False,
        "export_complete": False,
    })
    return root


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    raw = load_yaml_config(args.config)
    calculate_stats = args.calculate_stats
    if calculate_stats is None:
        calculate_stats = bool(raw.get("export", {}).get("calculate_stats", False))
    standardize_latents = args.standardize_latents
    if standardize_latents is None:
        standardize_latents = bool(
            raw.get("export", {}).get("standardize_latents", True)
        )
    # Standardization needs the statistics of the raw exported latent values.
    if standardize_latents:
        calculate_stats = True
    input_zarr = args.input_zarr or raw.get("data", {}).get("zarr")
    if not input_zarr:
        raise ValueError("Provide --input-zarr or data.zarr in the YAML file")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    # The checkpoint config is authoritative because it defines tensor shapes.
    checkpoint_config = checkpoint.get("config")
    print('Loaded checkpoint config:', checkpoint_config)

    if checkpoint_config:
        raw = dict(raw)
        raw["model"] = dict(checkpoint_config)
    config = make_bamua_config(raw)
    model = BAMUAAutoEncoder(config).to(device)
    load_model_weights(model, checkpoint)
    model.eval()

    dataset = BAMUAZarrDataset(input_zarr)
    source = dataset._open()
    all_counts = np.asarray(source["sample_count"][:], dtype=np.int64)
    all_times = dataset._int64_time(source["time_series"][:])

    source_indices = np.flatnonzero(all_counts > 0)
    source_indices = source_indices[np.argsort(all_times[source_indices], kind="stable")]
    if args.max_samples is not None:
        source_indices = source_indices[:max(args.max_samples, 0)]
    if len(source_indices) == 0:
        raise ValueError("No non-empty 6h bins were selected")

    output = create_output(
        path=args.output_zarr,
        n_samples=len(source_indices),
        config=config,
        times=all_times[source_indices].astype(np.int64),
        source_indices=source_indices.astype(np.int64),
        counts=all_counts[source_indices],
        checkpoint_path=args.checkpoint,
        input_path=input_zarr,
        overwrite=args.overwrite,
    )
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    print(
        f"device={device} samples={len(source_indices)} "
        f"latent_shape={output['latent'].shape} chunk_size={args.chunk_size} "
        f"calculate_stats={calculate_stats} "
        f"standardize_latents={standardize_latents}"
    )

    channel_sum = np.zeros(config.latent_dim, dtype=np.float64)
    channel_square_sum = np.zeros(config.latent_dim, dtype=np.float64)
    values_per_channel = 0
    with torch.inference_mode():
        for output_index, sample_index in enumerate(
            tqdm(source_indices, desc="export 1BAMUA latents", unit="bin")
        ):
            latent, density = encode_one_sample(
                model=model,
                dataset=dataset,
                sample_index=int(sample_index),
                chunk_size=args.chunk_size,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            output["latent"][output_index] = latent
            output["density"][output_index] = density
            if calculate_stats:
                latent64 = latent.astype(np.float64, copy=False)
                channel_sum += latent64.sum(axis=(1, 2))
                channel_square_sum += np.square(latent64).sum(axis=(1, 2))
                values_per_channel += latent.shape[1] * latent.shape[2]

    if calculate_stats:
        mean = channel_sum / values_per_channel
        variance = channel_square_sum / values_per_channel - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.maximum(std, 1.0e-6)
        output["latent_mean"][:] = mean.astype(np.float32)
        output["latent_std"][:] = std.astype(np.float32)
        output.attrs["latent_stats_calculated"] = True
        output.attrs["latent_stats_element_count_per_channel"] = int(
            values_per_channel
        )
        if standardize_latents:
            mean_grid = mean.astype(np.float32)[:, None, None]
            std_grid = std.astype(np.float32)[:, None, None]
            for output_index in tqdm(
                range(len(source_indices)),
                desc="standardize latent Zarr",
                unit="bin",
            ):
                latent = np.asarray(
                    output["latent"][output_index], dtype=np.float32
                )
                output["latent"][output_index] = (
                    latent - mean_grid
                ) / std_grid
            output.attrs["latents_standardized"] = True
            output.attrs["latent_standardization"] = (
                "stored_latent=(raw_latent-latent_mean)/latent_std"
            )

    output.attrs["export_complete"] = True
    print(f"saved={Path(args.output_zarr).resolve()}")


if __name__ == "__main__":
    main()

#  python -m satellite.export_bamua_latents --config satellite/configs/bamua_smoke.yaml --checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\runs\bamua_smoke\20260826_085312\epoch_0020.pth" --output-zarr "F:\lyh_data\data_latent\1bamua_latents.zarr"  --calculate-stats
