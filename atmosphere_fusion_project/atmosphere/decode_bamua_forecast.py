"""Forecast a BAMUA latent and decode it to query-point brightness temperatures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# This module is normally run from atmosphere_fusion_project, while satellite is
# a sibling package one directory above it.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.train_fusion import build_model, choose_device, load_checkpoint
from atmosphere.utils import amp_dtype_from_name
from satellite.models import BAMUAAutoEncoder
from satellite.train_bamua import (
    load_model_weights,
    load_yaml_config as load_bamua_yaml,
    make_bamua_config,
)


def _time_int64(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    return values.astype(np.int64)


def _read(array, indices):
    if isinstance(indices, slice):
        return np.asarray(array[indices])
    try:
        return np.asarray(array.oindex[indices])
    except AttributeError:
        return np.asarray(array.get_orthogonal_selection((indices,)))


def _create_array(group, name, shape, chunks, dtype):
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def _query_indices(start, count, max_query_points):
    if max_query_points is None or max_query_points <= 0 or count <= max_query_points:
        return np.arange(start, start + count, dtype=np.int64)
    # Spread a small test sample across the entire 6-hour bin instead of taking
    # only the beginning of one orbital swath.
    local = np.linspace(0, count - 1, max_query_points, dtype=np.int64)
    return local + start


def _bt_stats(root, channels):
    if "channel_mean" not in root or "channel_std" not in root:
        return None, None
    mean = np.asarray(root["channel_mean"][:], dtype=np.float32)
    std = np.asarray(root["channel_std"][:], dtype=np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(
            f"Expected BT stats [{channels}], got mean={mean.shape}, std={std.shape}"
        )
    return mean, np.maximum(std, 1.0e-6)


def _bt_is_standardized(root):
    description = str(root.attrs.get("brightness_temperature", "")).lower()
    if "standard" in description or "z-score" in description or "zscore" in description:
        return True
    if "physical" in description or "kelvin" in description or description == "raw":
        return False
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--bamua-config", required=True)
    parser.add_argument("--bamua-checkpoint", required=True)
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int, default=16_384)
    parser.add_argument(
        "--max-query-points", type=int, default=16_384,
        help="Maximum actual observations per lead; use 0 to decode the full 6h bin.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_zarr)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Add --overwrite to replace it."
        )
    if args.steps < 1 or args.query_chunk_size < 1:
        raise ValueError("steps and query_chunk_size must be positive")

    fusion_cfg = load_yaml(args.fusion_config)
    data_cfg = fusion_cfg["data"]
    train_cfg = fusion_cfg.get("train", {})
    instrument_name = "1bamua"
    if instrument_name not in data_cfg["instruments"]:
        raise KeyError("Fusion config must contain data.instruments.1bamua")

    # Only the initial latent is read. Future latents are produced autoregressively.
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"],
        rollout_steps=0,
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    item = dataset[args.sample_index]
    device = choose_device(train_cfg.get("device", "auto"))
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(train_cfg.get("amp_dtype", "bfloat16"))

    fusion_model = build_model(fusion_cfg, dataset).to(device)
    fusion_checkpoint = load_checkpoint(args.fusion_checkpoint, "cpu")
    checkpoint_cfg = fusion_checkpoint.get("config", {})
    checkpoint_data_cfg = checkpoint_cfg.get("data", {}) if checkpoint_cfg else {}
    if "normalize_latents" in checkpoint_data_cfg:
        trained_normalize = bool(checkpoint_data_cfg["normalize_latents"])
        if trained_normalize != dataset.normalize_latents:
            raise ValueError(
                "Fusion config normalize_latents does not match the checkpoint: "
                f"config={dataset.normalize_latents} checkpoint={trained_normalize}"
            )
    fusion_model.load_state_dict(fusion_checkpoint["model"])
    fusion_model.eval()

    bamua_raw = load_bamua_yaml(args.bamua_config)
    bamua_config = make_bamua_config(bamua_raw)
    bamua_model = BAMUAAutoEncoder(bamua_config).to(device)
    bamua_checkpoint = load_checkpoint(args.bamua_checkpoint, "cpu")
    load_model_weights(bamua_model, bamua_checkpoint)
    bamua_model.eval()

    if dataset.specs[instrument_name].latent_dim != bamua_config.latent_dim:
        raise ValueError(
            "Fusion BAMUA latent_dim does not match the BAMUA decoder: "
            f"{dataset.specs[instrument_name].latent_dim} != {bamua_config.latent_dim}"
        )

    observation_path = bamua_raw.get("data", {}).get("zarr")
    if not observation_path:
        raise ValueError("BAMUA config must contain data.zarr for query metadata and BT stats")
    import zarr

    observations = zarr.open_group(str(observation_path), mode="r")
    observation_times = _time_int64(observations["time_series"][:])
    time_to_index = {int(value): i for i, value in enumerate(observation_times)}
    bt_is_standardized = _bt_is_standardized(observations)
    if bt_is_standardized is None:
        raise ValueError(
            "Cannot determine the BAMUA BT scale from "
            "attrs['brightness_temperature']; expected standardized/z-score "
            "or physical/kelvin/raw"
        )
    if bt_is_standardized:
        bt_mean, bt_std = _bt_stats(observations, bamua_config.n_channels)
        if bt_mean is None:
            raise ValueError(
                "BT is marked standardized, but channel_mean/channel_std are missing"
            )
    else:
        bt_mean = bt_std = None

    latents = {
        name: item["latents"][name][:1].to(device)
        for name in fusion_model.instrument_names
    }
    densities = {
        name: item["densities"][name][:1].to(device)
        for name in fusion_model.instrument_names
    }
    available = {
        name: item["available"][name][:1].to(device)
        for name in fusion_model.instrument_names
    }
    output_shapes = fusion_model.spatial_shapes(latents)
    start_time = int(item["time"][0])
    interval_ns = int(dataset.interval_ns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = zarr.open_group(str(output_path), mode="w")
    output.attrs.update({
        "description": "BAMUA brightness temperatures decoded from atmosphere forecast latents",
        "fusion_checkpoint": str(Path(args.fusion_checkpoint).resolve()),
        "bamua_checkpoint": str(Path(args.bamua_checkpoint).resolve()),
        "source_sample_index": int(args.sample_index),
        "start_time_ns": start_time,
        "interval_hours": int(data_cfg.get("interval_hours", 6)),
        "latent_decoder_input": "raw BAMUA AE latent after inverse latent standardization",
        "decoder_output_standardized": bool(bt_is_standardized),
        "decoder_output_scale": "z-score" if bt_is_standardized else "physical",
        "bt_physical_available": True,
    })
    if bt_mean is not None:
        output.create_dataset("channel_mean", data=bt_mean, dtype="f4")
        output.create_dataset("channel_std", data=bt_std, dtype="f4")

    spec = dataset.specs[instrument_name]
    print(
        f"fusion_latent_scale: normalize_latents={dataset.normalize_latents} "
        f"stored_standardized={spec.stored_standardized}"
    )
    print(
        "decoder_latent_scale: dataset.denormalize() converts the fusion output "
        "back to the raw BAMUA AE latent"
    )
    print(
        "brightness_temperature_scale: "
        + ("decoder output is z-score; channel_mean/channel_std will restore physical BT"
           if bt_is_standardized else
           "decoder output is already physical; no second inverse transform will be applied")
    )

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            state, _ = fusion_model.fuse(latents, densities, available)

        for lead in range(1, args.steps + 1):
            future_time = start_time + lead * interval_ns
            if future_time not in time_to_index:
                raise KeyError(
                    f"No BAMUA observation 6h bin for forecast time {future_time}"
                )
            source_index = time_to_index[future_time]
            sample_start = int(observations["sample_start"][source_index])
            sample_count = int(observations["sample_count"][source_index])
            if sample_count < 1:
                raise ValueError(f"Future BAMUA bin {source_index} is empty")
            indices = _query_indices(
                sample_start, sample_count, args.max_query_points
            )
            count = len(indices)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                state = fusion_model.forecast_state(state)
                decoded = fusion_model.decode_state(state, output_shapes)
            latent_norm = decoded[instrument_name]["latent"]
            latent_raw = dataset.denormalize(instrument_name, latent_norm).float()

            group = output.create_group(f"lead_{lead:03d}")
            group.attrs.update({
                "lead": lead,
                "time_ns": int(future_time),
                "source_observation_sample_index": int(source_index),
                "source_observation_count": int(sample_count),
                "decoded_query_count": int(count),
            })
            point_chunk = min(args.query_chunk_size, count)
            matrix_chunk = (point_chunk, bamua_config.n_channels)
            lon_out = _create_array(group, "longitude", (count,), (point_chunk,), "f4")
            lat_out = _create_array(group, "latitude", (count,), (point_chunk,), "f4")
            sat_out = _create_array(group, "satellite_id", (count,), (point_chunk,), "i8")
            land_out = _create_array(group, "is_land", (count,), (point_chunk,), "bool")
            valid_out = _create_array(
                group, "target_valid", (count, bamua_config.n_channels), matrix_chunk, "bool"
            )
            pred_decoder_out = _create_array(
                group, "pred_bt_decoder_scale",
                (count, bamua_config.n_channels), matrix_chunk, "f4",
            )
            target_decoder_out = _create_array(
                group, "target_bt_decoder_scale",
                (count, bamua_config.n_channels), matrix_chunk, "f4",
            )
            pred_phys_out = _create_array(
                group, "pred_bt_physical",
                (count, bamua_config.n_channels), matrix_chunk, "f4",
            )
            target_phys_out = _create_array(
                group, "target_bt_physical",
                (count, bamua_config.n_channels), matrix_chunk, "f4",
            )

            squared_error_sum = np.zeros(bamua_config.n_channels, dtype=np.float64)
            physical_squared_error_sum = np.zeros(
                bamua_config.n_channels, dtype=np.float64
            )
            valid_count = np.zeros(bamua_config.n_channels, dtype=np.int64)
            for offset in range(0, count, args.query_chunk_size):
                end = min(offset + args.query_chunk_size, count)
                chunk_indices = indices[offset:end]
                lon = _read(observations["longitude"], chunk_indices).astype(np.float32)
                lat = _read(observations["latitude"], chunk_indices).astype(np.float32)
                satellite_id = _read(observations["satellite_id"], chunk_indices).astype(np.int64)
                is_land = _read(observations["is_land"], chunk_indices).astype(bool)
                target_bt = _read(
                    observations["brightness_temperature"], chunk_indices
                ).astype(np.float32)
                target_valid = _read(
                    observations["brightness_temperature_valid"], chunk_indices
                ).astype(bool)

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    pred_std = bamua_model.decode(
                        latent=latent_raw,
                        lon=torch.from_numpy(lon).to(device).unsqueeze(0),
                        lat=torch.from_numpy(lat).to(device).unsqueeze(0),
                        satellite_id=torch.from_numpy(satellite_id).to(device).unsqueeze(0),
                        is_land=torch.from_numpy(is_land).to(device).unsqueeze(0),
                        sample_time=torch.tensor([future_time], dtype=torch.long, device=device),
                    )
                pred_std = pred_std[0].float().cpu().numpy()

                lon_out[offset:end] = lon
                lat_out[offset:end] = lat
                sat_out[offset:end] = satellite_id
                land_out[offset:end] = is_land
                valid_out[offset:end] = target_valid
                pred_decoder_out[offset:end] = pred_std
                target_decoder_out[offset:end] = target_bt
                error = (pred_std - target_bt) ** 2
                squared_error_sum += (error * target_valid).sum(axis=0)
                valid_count += target_valid.sum(axis=0)

                if bt_is_standardized:
                    pred_phys = pred_std * bt_std[None, :] + bt_mean[None, :]
                    target_phys = target_bt * bt_std[None, :] + bt_mean[None, :]
                else:
                    pred_phys = pred_std
                    target_phys = target_bt
                physical_error = (pred_phys - target_phys) ** 2
                physical_squared_error_sum += (
                    physical_error * target_valid
                ).sum(axis=0)
                target_phys = np.where(target_valid, target_phys, np.nan)
                pred_phys_out[offset:end] = pred_phys.astype(np.float32)
                target_phys_out[offset:end] = target_phys.astype(np.float32)

            channel_mse = squared_error_sum / np.maximum(valid_count, 1)
            total_mse = squared_error_sum.sum() / max(valid_count.sum(), 1)
            group.attrs["decoder_scale_channel_mse"] = channel_mse.tolist()
            group.attrs["decoder_scale_masked_mse"] = float(total_mse)
            group.attrs["decoder_scale_masked_rmse"] = float(np.sqrt(total_mse))
            physical_channel_mse = physical_squared_error_sum / np.maximum(
                valid_count, 1
            )
            physical_total_mse = physical_squared_error_sum.sum() / max(
                valid_count.sum(), 1
            )
            group.attrs["physical_channel_mse"] = physical_channel_mse.tolist()
            group.attrs["physical_channel_rmse"] = np.sqrt(
                physical_channel_mse
            ).tolist()
            group.attrs["physical_masked_mse"] = float(physical_total_mse)
            group.attrs["physical_masked_rmse"] = float(
                np.sqrt(physical_total_mse)
            )
            print(
                f"lead={lead} time_ns={future_time} queries={count} "
                f"decoder_scale_rmse={np.sqrt(total_mse):.6f} "
                f"physical_rmse={np.sqrt(physical_total_mse):.6f}"
            )

    output.attrs["complete"] = True
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
