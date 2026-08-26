"""End-to-end BAMUA inference from either 6-hour observations or latent states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.decode_bamua_forecast import (
    _bt_is_standardized,
    _bt_stats,
    _create_array,
    _read,
    _time_int64,
)
from atmosphere.train_fusion import build_model, choose_device, load_checkpoint
from atmosphere.utils import amp_dtype_from_name
from satellite.models import BAMUAAutoEncoder
from satellite.train_bamua import (
    load_model_weights,
    load_yaml_config as load_bamua_yaml,
    make_bamua_config,
)


def _load_queries(args):
    if args.query_npz:
        query = np.load(args.query_npz)
        lon = np.asarray(query["longitude"], dtype=np.float32).reshape(-1)
        lat = np.asarray(query["latitude"], dtype=np.float32).reshape(-1)
        if lon.shape != lat.shape:
            raise ValueError("query longitude and latitude must have the same shape")
        satellite_id = (
            np.asarray(query["satellite_id"], dtype=np.int64).reshape(-1)
            if "satellite_id" in query else
            None
        )
        is_land = (
            np.asarray(query["is_land"], dtype=bool).reshape(-1)
            if "is_land" in query else
            np.full(lon.shape, bool(args.query_is_land), dtype=bool)
        )
        if satellite_id is None:
            if args.query_satellite_id is None:
                raise ValueError(
                    "query NPZ has no satellite_id; provide --query-satellite-id"
                )
            satellite_id = np.full(
                lon.shape, args.query_satellite_id, dtype=np.int64
            )
        if satellite_id.size == 1:
            satellite_id = np.full(lon.shape, satellite_id.item(), dtype=np.int64)
        if is_land.size == 1:
            is_land = np.full(lon.shape, is_land.item(), dtype=bool)
        if satellite_id.shape != lon.shape or is_land.shape != lon.shape:
            raise ValueError("query satellite_id/is_land must be scalar or match lon/lat")
        return lon, lat, satellite_id, is_land

    resolution = float(args.query_resolution_deg)
    if resolution <= 0:
        raise ValueError("query_resolution_deg must be positive")
    latitude = np.linspace(
        -90.0, 90.0, round(180.0 / resolution) + 1, dtype=np.float32
    )
    longitude = np.linspace(
        -180.0, 180.0, round(360.0 / resolution) + 1,
        endpoint=True, dtype=np.float32,
    )[:-1]
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    lon = lon_grid.reshape(-1)
    lat = lat_grid.reshape(-1)
    if args.query_satellite_id is None:
        raise ValueError("--query-satellite-id is required for a global query grid")
    satellite_id = np.full(lon.shape, args.query_satellite_id, dtype=np.int64)
    if args.land_mask_npy:
        land = np.asarray(np.load(args.land_mask_npy), dtype=bool)
        if land.shape != lat_grid.shape:
            raise ValueError(
                f"land mask must have shape {lat_grid.shape}, got {land.shape}"
            )
        is_land = land.reshape(-1)
    else:
        is_land = np.full(lon.shape, bool(args.query_is_land), dtype=bool)
        print(
            "warning: no --land-mask-npy was given; the global query grid uses "
            f"is_land={bool(args.query_is_land)} everywhere"
        )
    return lon, lat, satellite_id, is_land


@torch.no_grad()
def _encode_observation_bin(model, root, sample_index, chunk_size, device,
                            amp_enabled, amp_dtype, max_observations=0, seed=0):
    sample_start = int(root["sample_start"][sample_index])
    sample_count = int(root["sample_count"][sample_index])
    sample_time = int(_time_int64(root["time_series"][sample_index]).item())
    if sample_count < 1:
        raise ValueError(f"BAMUA 6h bin {sample_index} is empty")

    if max_observations > 0 and sample_count > max_observations:
        rng = np.random.default_rng(int(seed) + int(sample_index))
        selected = rng.choice(
            sample_count, size=int(max_observations), replace=False
        )
        selected.sort()
        selected = selected.astype(np.int64) + sample_start
    else:
        selected = None
    selected_count = sample_count if selected is None else len(selected)
    print(
        f"observation_input: sample_index={sample_index} "
        f"sample_time_ns={sample_time} source_count={sample_count} "
        f"selected_count={selected_count}"
    )

    latent_sum = density_sum = None
    for offset in range(0, selected_count, chunk_size):
        end = min(offset + chunk_size, selected_count)
        indices = (
            slice(sample_start + offset, sample_start + end)
            if selected is None else selected[offset:end]
        )
        bt = _read(root["brightness_temperature"], indices).astype(np.float32)
        # The frozen AE always receives the same scale stored in its training Zarr.
        # For the current BAMUA store this is channel-wise z-score BT.
        valid = _read(root["brightness_temperature_valid"], indices).astype(bool)
        satellite_id = _read(root["satellite_id"], indices).astype(np.int64)
        is_land = _read(root["is_land"], indices).astype(bool)
        obs_time = _time_int64(_read(root["time"], indices))
        zenith = _read(root["satellite_zenith_angle"], indices).astype(np.float32)
        azimuth = _read(root["satellite_azimuth"], indices).astype(np.float32)
        lon = _read(root["longitude"], indices).astype(np.float32)
        lat = _read(root["latitude"], indices).astype(np.float32)

        tensors = {
            "bt": torch.from_numpy(bt).to(device).unsqueeze(0),
            "valid": torch.from_numpy(valid).to(device).unsqueeze(0),
            "satellite_id": torch.from_numpy(satellite_id).to(device).unsqueeze(0),
            "is_land": torch.from_numpy(is_land).to(device).unsqueeze(0),
            "obs_time": torch.from_numpy(obs_time).to(device).unsqueeze(0),
            "zenith": torch.from_numpy(zenith).to(device).unsqueeze(0),
            "azimuth": torch.from_numpy(azimuth).to(device).unsqueeze(0),
            "lon": torch.from_numpy(lon).to(device).unsqueeze(0),
            "lat": torch.from_numpy(lat).to(device).unsqueeze(0),
            "sample_time": torch.tensor([sample_time], dtype=torch.long, device=device),
        }
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            feature = model.point_encoder(
                tensors["bt"], tensors["valid"],
                **model._metadata(
                    satellite_id=tensors["satellite_id"],
                    is_land=tensors["is_land"],
                    sample_time=tensors["sample_time"],
                    obs_time=tensors["obs_time"],
                    zenith=tensors["zenith"],
                    azimuth=tensors["azimuth"],
                ),
            )
            chunk_sum, chunk_density = model.point_to_grid.aggregate(
                feature, tensors["lon"], tensors["lat"]
            )
        latent_sum = chunk_sum if latent_sum is None else latent_sum + chunk_sum
        density_sum = (
            chunk_density if density_sum is None else density_sum + chunk_density
        )

    latent = latent_sum / density_sum.clamp_min(model.config.eps)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        latent = model.latent_processor(latent)
    return latent.float(), density_sum.float(), sample_time


def _raw_to_fusion_scale(dataset, name, latent):
    if not dataset.normalize_latents:
        return latent
    spec = dataset.specs[name]
    mean = torch.as_tensor(
        spec.latent_mean, device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1)
    std = torch.as_tensor(
        spec.latent_std, device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1).clamp_min(1.0e-6)
    return (latent - mean) / std


@torch.no_grad()
def _decode_to_output(model, latent_raw, sample_time, queries, output_group,
                      chunk_size, device, amp_enabled, amp_dtype,
                      bt_is_standardized, bt_mean, bt_std):
    lon, lat, satellite_id, is_land = queries
    count = len(lon)
    matrix_chunks = (min(chunk_size, count), model.config.n_channels)
    pred_decoder = _create_array(
        output_group, "pred_bt_decoder_scale",
        (count, model.config.n_channels), matrix_chunks, "f4",
    )
    pred_physical = _create_array(
        output_group, "pred_bt_physical",
        (count, model.config.n_channels), matrix_chunks, "f4",
    )
    for offset in range(0, count, chunk_size):
        end = min(offset + chunk_size, count)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            pred = model.decode(
                latent=latent_raw,
                lon=torch.from_numpy(lon[offset:end]).to(device).unsqueeze(0),
                lat=torch.from_numpy(lat[offset:end]).to(device).unsqueeze(0),
                satellite_id=torch.from_numpy(
                    satellite_id[offset:end]
                ).to(device).unsqueeze(0),
                is_land=torch.from_numpy(is_land[offset:end]).to(device).unsqueeze(0),
                sample_time=torch.tensor(
                    [sample_time], dtype=torch.long, device=device
                ),
            )
        pred = pred[0].float().cpu().numpy()
        pred_decoder[offset:end] = pred
        if bt_is_standardized:
            pred = pred * bt_std[None, :] + bt_mean[None, :]
        pred_physical[offset:end] = pred.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=("observations", "latent"), required=True)
    parser.add_argument("--input-sample-index", type=int, default=0)
    parser.add_argument("--forecast-steps", type=int, default=0)
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--bamua-config", required=True)
    parser.add_argument("--bamua-checkpoint", required=True)
    parser.add_argument("--fusion-config")
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--query-npz")
    parser.add_argument("--query-resolution-deg", type=float, default=2.0)
    parser.add_argument("--query-satellite-id", type=int)
    parser.add_argument("--query-is-land", type=int, choices=(0, 1), default=0)
    parser.add_argument("--land-mask-npy")
    parser.add_argument("--encode-chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--max-input-observations", type=int, default=0,
        help="Maximum observations encoded from the selected 6h bin; 0 uses all.",
    )
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--decode-chunk-size", type=int, default=16_384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.forecast_steps < 0:
        raise ValueError("forecast_steps cannot be negative")
    if args.max_input_observations < 0:
        raise ValueError("max_input_observations cannot be negative")
    needs_fusion_data = args.input_mode == "latent" or args.forecast_steps > 0
    if needs_fusion_data and not args.fusion_config:
        raise ValueError("--fusion-config is required for latent input or forecasting")
    if args.forecast_steps > 0 and not args.fusion_checkpoint:
        raise ValueError("--fusion-checkpoint is required when forecast_steps > 0")
    output_path = Path(args.output_zarr)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Add --overwrite to replace it."
        )

    device = choose_device(args.device)
    amp_enabled = args.mixed_precision and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    bamua_raw = load_bamua_yaml(args.bamua_config)
    bamua_config = make_bamua_config(bamua_raw)
    bamua_model = BAMUAAutoEncoder(bamua_config).to(device)
    bamua_checkpoint = load_checkpoint(args.bamua_checkpoint, "cpu")
    load_model_weights(bamua_model, bamua_checkpoint)
    bamua_model.eval()

    observation_path = bamua_raw.get("data", {}).get("zarr")
    if not observation_path:
        raise ValueError("BAMUA config must contain data.zarr")
    import zarr

    observation_root = zarr.open_group(str(observation_path), mode="r")
    bt_is_standardized = _bt_is_standardized(observation_root)
    if bt_is_standardized is None:
        raise ValueError("Cannot determine brightness_temperature scale from Zarr attrs")
    bt_mean, bt_std = _bt_stats(observation_root, bamua_config.n_channels)
    if bt_is_standardized and bt_mean is None:
        raise ValueError("Standardized BT requires channel_mean/channel_std")

    fusion_cfg = dataset = fusion_model = None
    if needs_fusion_data:
        fusion_cfg = load_yaml(args.fusion_config)
        data_cfg = fusion_cfg["data"]
        dataset = MultiInstrumentLatentSequenceDataset(
            stores=data_cfg["instruments"], rollout_steps=0,
            interval_hours=int(data_cfg.get("interval_hours", 6)),
            normalize_latents=bool(data_cfg.get("normalize_latents", True)),
        )
        if "1bamua" not in dataset.specs:
            raise KeyError("Fusion config must contain data.instruments.1bamua")
    if args.forecast_steps > 0:
        fusion_model = build_model(fusion_cfg, dataset).to(device)
        checkpoint = load_checkpoint(args.fusion_checkpoint, "cpu")
        checkpoint_cfg = checkpoint.get("config", {})
        checkpoint_data = checkpoint_cfg.get("data", {}) if checkpoint_cfg else {}
        if "normalize_latents" in checkpoint_data:
            trained_normalize = bool(checkpoint_data["normalize_latents"])
            if trained_normalize != dataset.normalize_latents:
                raise ValueError(
                    "Fusion normalize_latents differs from the checkpoint: "
                    f"config={dataset.normalize_latents} "
                    f"checkpoint={trained_normalize}"
                )
        fusion_model.load_state_dict(checkpoint["model"])
        fusion_model.eval()

    if args.input_mode == "observations":
        latent_raw, density, sample_time = _encode_observation_bin(
            bamua_model, observation_root, args.input_sample_index,
            args.encode_chunk_size, device, amp_enabled, amp_dtype,
            max_observations=args.max_input_observations,
            seed=args.input_seed,
        )
        if dataset is not None:
            bamua_latent = _raw_to_fusion_scale(dataset, "1bamua", latent_raw)
            latents, densities, available = {}, {}, {}
            for name in dataset.specs:
                if name == "1bamua":
                    latents[name] = bamua_latent
                    densities[name] = density
                    available[name] = torch.ones(1, dtype=torch.bool, device=device)
                else:
                    dim = dataset.specs[name].latent_dim
                    h, w = latent_raw.shape[-2:]
                    latents[name] = torch.zeros(1, dim, h, w, device=device)
                    densities[name] = torch.zeros(1, 1, h, w, device=device)
                    available[name] = torch.zeros(1, dtype=torch.bool, device=device)
        else:
            latents = densities = available = None
    else:
        item = dataset[args.input_sample_index]
        sample_time = int(item["time"][0])
        latents = {
            name: item["latents"][name][:1].to(device)
            for name in dataset.specs
        }
        densities = {
            name: item["densities"][name][:1].to(device)
            for name in dataset.specs
        }
        available = {
            name: item["available"][name][:1].to(device)
            for name in dataset.specs
        }
        if not bool(available["1bamua"][0]):
            raise ValueError("The selected latent sample has no available 1bamua state")
        latent_raw = dataset.denormalize("1bamua", latents["1bamua"]).float()

    queries = _load_queries(args)
    if len(queries[0]) < 1:
        raise ValueError("No query points")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = zarr.open_group(str(output_path), mode="w")
    output.attrs.update({
        "input_mode": args.input_mode,
        "input_sample_index": int(args.input_sample_index),
        "input_sample_time_ns": int(sample_time),
        "forecast_steps": int(args.forecast_steps),
        "decoder_output_standardized": bool(bt_is_standardized),
        "decoder_output_scale": "z-score" if bt_is_standardized else "physical",
        "complete": False,
    })
    lon, lat, satellite_id, is_land = queries
    point_chunks = (min(args.decode_chunk_size, len(lon)),)
    output.create_dataset("longitude", data=lon, chunks=point_chunks, dtype="f4")
    output.create_dataset("latitude", data=lat, chunks=point_chunks, dtype="f4")
    output.create_dataset("satellite_id", data=satellite_id, chunks=point_chunks, dtype="i8")
    output.create_dataset("is_land", data=is_land, chunks=point_chunks, dtype="bool")
    if bt_is_standardized:
        output.create_dataset("channel_mean", data=bt_mean, dtype="f4")
        output.create_dataset("channel_std", data=bt_std, dtype="f4")

    outputs = []
    if args.forecast_steps == 0 or args.include_current:
        outputs.append((0, sample_time, latent_raw))
    if args.forecast_steps > 0:
        output_shapes = fusion_model.spatial_shapes(latents)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            state, _ = fusion_model.fuse(latents, densities, available)
            for lead in range(1, args.forecast_steps + 1):
                state = fusion_model.forecast_state(state)
                predicted = fusion_model.decode_state(state, output_shapes)
                predicted_raw = dataset.denormalize(
                    "1bamua", predicted["1bamua"]["latent"]
                ).float()
                future_time = sample_time + lead * dataset.interval_ns
                outputs.append((lead, future_time, predicted_raw))

    with torch.no_grad():
        for lead, output_time, output_latent in outputs:
            group = output.create_group(f"lead_{lead:03d}")
            group.attrs["lead"] = int(lead)
            group.attrs["time_ns"] = int(output_time)
            _decode_to_output(
                bamua_model, output_latent, output_time, queries, group,
                args.decode_chunk_size, device, amp_enabled, amp_dtype,
                bt_is_standardized, bt_mean, bt_std,
            )
            print(
                f"decoded lead={lead} time_ns={output_time} "
                f"queries={len(lon)}"
            )
    output.attrs["complete"] = True
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()


## 原始观测 → AE Encoder → 当前时刻重构

#  python -m atmosphere.bamua_inference --input-mode observations --input-sample-index 0 --max-input-observations 65536 --input-seed 42 --forecast-steps 0 --bamua-config ..\satellite\configs\bamua_smoke.yaml --bamua-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\runs\bamua_smoke\20260826_085312\best.pth" --query-resolution-deg 2.0 --query-satellite-id 0 --query-is-land 0 --encode-chunk-size 16384 --decode-chunk-size 16384 --output-zarr "F:\lyh_data\data_inference\bamua_obs_current.zarr" --overwrite


## 原始观测 → Fusion/Forecast → 未来亮温

# python -m atmosphere.bamua_inference --input-mode observations --input-sample-index 0 --max-input-observations 65536 --input-seed 42 --forecast-steps 4 --include-current --bamua-config ..\satellite\configs\bamua_smoke.yaml --bamua-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\runs\bamua_smoke\20260826_085312\best.pth" --fusion-config atmosphere\configs\fusion_smoke.yaml --fusion-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\atmosphere_fusion_project\runs\atmosphere_fusion_smoke\20260826_203336\best.pth" --query-resolution-deg 2.0 --query-satellite-id 0 --query-is-land 0 --encode-chunk-size 16384 --decode-chunk-size 16384 --output-zarr "F:\lyh_data\data_inference\bamua_obs_forecast_24h.zarr" --overwrite


## latent → AE Decoder → 当前时刻重构
# python -m atmosphere.bamua_inference --input-mode latent --input-sample-index 0 --forecast-steps 0 --bamua-config ..\satellite\configs\bamua_smoke.yaml --bamua-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\runs\bamua_smoke\20260826_085312\best.pth" --fusion-config atmosphere\configs\fusion_smoke.yaml --query-resolution-deg 2.0 --query-satellite-id 0 --query-is-land 0 --decode-chunk-size 16384 --output-zarr "F:\lyh_data\data_inference\bamua_latent_current.zarr" --overwrite

## latent → Fusion/Forecast → 未来亮温
# python -m atmosphere.bamua_inference --input-mode latent --input-sample-index 0 --forecast-steps 4 --include-current --bamua-config ..\satellite\configs\bamua_smoke.yaml --bamua-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\runs\bamua_smoke\20260826_085312\best.pth" --fusion-config atmosphere\configs\fusion_smoke.yaml --fusion-checkpoint "C:\Users\Lenovo\code\local_code\aardvark-weather-public\atmosphere_fusion_project\runs\atmosphere_fusion_smoke\20260826_203336\best.pth" --query-resolution-deg 2.0 --query-satellite-id 0 --query-is-land 0 --decode-chunk-size 16384 --output-zarr "F:\lyh_data\data_inference\bamua_latent_forecast_24h.zarr" --overwrite