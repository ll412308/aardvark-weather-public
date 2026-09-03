"""End-to-end plots for AE -> fusion -> forecast -> instrument AE decoder.

The current decoder registry contains 1BAMUA. More instruments can be added by
registering their AE config/checkpoint and observation reader in this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atmosphere.bamua_inference import _encode_observation_bin, _raw_to_fusion_scale
from atmosphere.config import load_yaml
from atmosphere.data import MultiInstrumentLatentSequenceDataset
from atmosphere.decode_bamua_forecast import (
    _bt_is_standardized, _bt_stats, _query_indices, _read, _time_int64,
)
from atmosphere.train_fusion import build_model, choose_device, load_checkpoint
from atmosphere.utils import amp_dtype_from_name
from satellite.models import BAMUAAutoEncoder
from satellite.train_bamua import (
    load_model_weights,
    load_yaml_config as load_bamua_yaml,
    make_bamua_config,
)


# BUFR satellite identifier mapping used by the original local GDAS readers.
SATELLITE_NAMES = {
    1: "ERS-1",
    2: "ERS-2",
    3: "METOP-B",
    4: "METOP-A",
    5: "METOP-C",
    206: "NOAA-15",
    207: "NOAA-16",
    208: "NOAA-17",
    209: "NOAA-18",
    223: "NOAA-19",
    224: "NPP",
    225: "NOAA-20",
    226: "NOAA-21",
}


def _csv(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _channels(value, count):
    channels = [int(item) - 1 for item in _csv(value)]
    if not channels or any(channel < 0 or channel >= count for channel in channels):
        raise ValueError(f"channels must be 1-based values in [1, {count}]")
    return channels


def _parse_time(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return int(np.datetime64(value, "ns").astype(np.int64))


def _readable_time(time_ns):
    value = np.datetime64(int(time_ns), "ns")
    return np.datetime_as_string(value, unit="s") + " UTC"


def _filename_time(time_ns):
    value = np.datetime_as_string(np.datetime64(int(time_ns), "ns"), unit="s")
    return value.replace("-", "").replace(":", "") + "Z"


def _physical(values, standardized, mean, std):
    values = np.asarray(values, dtype=np.float32)
    if standardized:
        return values * std.reshape(1, -1) + mean.reshape(1, -1)
    return values


def _decode(model, latent, lon, lat, satellite_id, is_land, sample_time,
            chunk_size, device, amp_enabled, amp_dtype):
    chunks = []
    for start in range(0, len(lon), chunk_size):
        end = min(start + chunk_size, len(lon))
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            prediction = model.decode(
                latent=latent,
                lon=torch.from_numpy(lon[start:end]).to(device).unsqueeze(0),
                lat=torch.from_numpy(lat[start:end]).to(device).unsqueeze(0),
                satellite_id=torch.from_numpy(
                    satellite_id[start:end]
                ).to(device).unsqueeze(0),
                is_land=torch.from_numpy(
                    is_land[start:end]
                ).to(device).unsqueeze(0),
                sample_time=torch.tensor(
                    [sample_time], dtype=torch.long, device=device
                ),
            )
        chunks.append(prediction[0].float().cpu().numpy())
    prediction = np.concatenate(chunks, axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError(
            f"AE decoder produced NaN/Inf at sample_time={sample_time}"
        )
    return prediction


def _plot(worker, kind, data_path, output_dir, instrument, lead,
          interval_hours, channels, point_size, resolution_deg=None,
          color_limits=None, count_tag="", count_text="", output_time=None):
    for channel in channels:
        job = {
            "kind": kind,
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "instrument": instrument,
            "lead": int(lead),
            "lead_hours": int(lead * interval_hours),
            "channel": int(channel),
            "point_size": float(point_size),
            "resolution_deg": resolution_deg,
            "color_min": (
                None if color_limits is None else float(color_limits[0][channel])
            ),
            "color_max": (
                None if color_limits is None else float(color_limits[1][channel])
            ),
            "count_tag": count_tag,
            "count_text": count_text,
            "time_text": _readable_time(output_time),
            "time_tag": _filename_time(output_time),
            "satellite_names": {
                str(key): value for key, value in SATELLITE_NAMES.items()
            },
        }
        job_path = Path(output_dir) / f".{kind}_channel_{channel + 1:02d}.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        try:
            subprocess.run([sys.executable, str(worker), str(job_path)], check=True)
        finally:
            job_path.unlink(missing_ok=True)


def _count_description(context_used, context_total, query_used,
                       query_total=None):
    """Return filename-safe and human-readable observation-count labels."""
    context_fraction = 100.0 * context_used / max(context_total, 1)
    context_tag = (
        f"ctx_{context_used}of{context_total}_"
        f"{context_fraction:.1f}pct"
    ).replace(".", "p")
    context_text = (
        f"context observations: {context_used:,}/{context_total:,} "
        f"({context_fraction:.1f}%)"
    )
    if query_total is None:
        query_tag = f"queries_{query_used}"
        query_text = f"decoder query outputs: {query_used:,}"
    else:
        query_fraction = 100.0 * query_used / max(query_total, 1)
        query_tag = (
            f"target_{query_used}of{query_total}_"
            f"{query_fraction:.1f}pct"
        ).replace(".", "p")
        query_text = (
            f"target outputs: {query_used:,}/{query_total:,} "
            f"({query_fraction:.1f}%)"
        )
    return f"{context_tag}_{query_tag}", f"{context_text}; {query_text}"


def _arbitrary_queries(args, default_satellite_id):
    if args.query_npz:
        source = np.load(args.query_npz)
        lon = np.asarray(source["longitude"], dtype=np.float32).reshape(-1)
        lat = np.asarray(source["latitude"], dtype=np.float32).reshape(-1)
        satellite_id = np.asarray(
            source["satellite_id"] if "satellite_id" in source
            else np.full(lon.shape, default_satellite_id), dtype=np.int64,
        ).reshape(-1)
        is_land = np.asarray(
            source["is_land"] if "is_land" in source
            else np.full(lon.shape, bool(args.query_is_land)), dtype=bool,
        ).reshape(-1)
    else:
        rng = np.random.default_rng(args.seed)
        lon = rng.uniform(-180.0, 180.0, args.arbitrary_point_count).astype(np.float32)
        # Uniform sampling on a sphere rather than concentrating near the poles.
        lat = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, len(lon)))).astype(np.float32)
        satellite_id = np.full(lon.shape, default_satellite_id, dtype=np.int64)
        is_land = np.full(lon.shape, bool(args.query_is_land), dtype=bool)
    if not (lon.shape == lat.shape == satellite_id.shape == is_land.shape):
        raise ValueError("Arbitrary query arrays must have matching one-dimensional shapes")
    return lon, lat, satellite_id, is_land


def _grid_queries(resolution, satellite_id, is_land, land_mask_path=None):
    if resolution <= 0:
        raise ValueError("grid-resolution-deg must be positive")
    lat = np.linspace(-90, 90, round(180 / resolution) + 1, dtype=np.float32)
    lon = np.linspace(-180, 180, round(360 / resolution), endpoint=False,
                      dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    shape = lon_grid.shape
    if land_mask_path:
        land = np.asarray(np.load(land_mask_path), dtype=bool)
        if land.shape != shape:
            raise ValueError(
                f"Grid land mask must have shape {shape}, got {land.shape}"
            )
        land = land.reshape(-1)
    else:
        land = np.full(lon_grid.size, is_land, dtype=bool)
    return (
        lon_grid, lat_grid, lon_grid.reshape(-1), lat_grid.reshape(-1),
        np.full(lon_grid.size, satellite_id, dtype=np.int64),
        land, shape,
    )


def _apply_yaml_arguments(args, config):
    """Copy the readable nested YAML fields into argparse's flat namespace."""
    fields = {
        "fusion_config": ("models", "fusion_config"),
        "fusion_checkpoint": ("models", "fusion_checkpoint"),
        "bamua_config": ("models", "bamua_config"),
        "bamua_checkpoint": ("models", "bamua_checkpoint"),
        "sample_index": ("selection", "sample_index"),
        "start_time": ("selection", "start_time"),
        "input_instruments": ("selection", "input_instruments"),
        "output_instruments": ("selection", "output_instruments"),
        "forecast_steps": ("forecast", "steps"),
        "encode_bamua_observations": ("sampling", "encode_bamua_observations"),
        "max_input_observations": ("sampling", "max_input_observations"),
        "max_target_observations": ("sampling", "max_target_observations"),
        "encode_chunk_size": ("sampling", "encode_chunk_size"),
        "decode_chunk_size": ("sampling", "decode_chunk_size"),
        "query_npz": ("arbitrary_queries", "query_npz"),
        "arbitrary_point_count": ("arbitrary_queries", "point_count"),
        "query_satellite_id": ("query_metadata", "satellite_id"),
        "query_satellite": ("query_metadata", "satellite"),
        "query_is_land": ("query_metadata", "is_land"),
        "grid_resolution_deg": ("grid", "resolution_deg"),
        "grid_land_mask_npy": ("grid", "land_mask_npy"),
        "channels": ("plot", "channels"),
        "point_size": ("plot", "point_size"),
        "color_std_range": ("plot", "color_std_range"),
        "seed": ("runtime", "seed"),
        "device": ("runtime", "device"),
        "mixed_precision": ("runtime", "mixed_precision"),
        "amp_dtype": ("runtime", "amp_dtype"),
        "output_dir": ("output", "directory"),
        "overwrite": ("output", "overwrite"),
    }
    for destination, (section, key) in fields.items():
        section_values = config.get(section, {})
        if key in section_values:
            setattr(args, destination, section_values[key])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="End-to-end inference YAML; its values override other CLI options",
    )
    parser.add_argument("--fusion-config")
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--bamua-config")
    parser.add_argument("--bamua-checkpoint")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--start-time",
        help="Exact start time as Unix nanoseconds or e.g. 2017-01-10T00:00:00",
    )
    parser.add_argument("--input-instruments", default="1bamua")
    parser.add_argument("--output-instruments", default="1bamua")
    parser.add_argument("--forecast-steps", type=int, default=4)
    parser.add_argument(
        "--encode-bamua-observations", action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode the selected raw BAMUA bin instead of reading its saved latent",
    )
    parser.add_argument("--max-input-observations", type=int, default=0)
    parser.add_argument(
        "--max-target-observations", type=int, default=0,
        help=(
            "Maximum target observations decoded at every lead; "
            "0 uses the complete 6-hour target bin"
        ),
    )
    parser.add_argument("--encode-chunk-size", type=int, default=16_384)
    parser.add_argument("--decode-chunk-size", type=int, default=16_384)
    parser.add_argument("--channels", default="1")
    parser.add_argument("--query-npz")
    parser.add_argument("--arbitrary-point-count", type=int, default=20_000)
    parser.add_argument("--query-satellite-id", type=int)
    parser.add_argument(
        "--query-satellite",
        help="Readable satellite name, e.g. METOP-A or NOAA-18",
    )
    parser.add_argument("--query-is-land", type=int, choices=(0, 1), default=0)
    parser.add_argument("--grid-resolution-deg", type=float, default=2.0)
    parser.add_argument("--grid-land-mask-npy")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--color-std-range", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.config:
        args = _apply_yaml_arguments(args, load_yaml(args.config))
    required = (
        "fusion_config", "fusion_checkpoint", "bamua_config",
        "bamua_checkpoint", "output_dir",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(
            "Missing required settings: " + ", ".join(missing)
            + ". Provide them in --config YAML or as command-line options."
        )
    if args.query_satellite:
        reverse_names = {
            name.upper(): satellite_id
            for satellite_id, name in SATELLITE_NAMES.items()
        }
        requested_name = str(args.query_satellite).upper()
        if requested_name not in reverse_names:
            raise ValueError(
                f"Unknown query satellite name: {args.query_satellite}. "
                f"Use one of {sorted(reverse_names)}"
            )
        args.query_satellite_id = reverse_names[requested_name]

    if args.forecast_steps < 0:
        raise ValueError("forecast-steps cannot be negative")
    if args.max_input_observations < 0 or args.max_target_observations < 0:
        raise ValueError("max input/target observations cannot be negative")
    if args.encode_chunk_size < 1 or args.decode_chunk_size < 1:
        raise ValueError("encode/decode chunk sizes must be positive")
    if args.arbitrary_point_count < 1:
        raise ValueError("arbitrary-point-count must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; add --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_cfg = load_yaml(args.fusion_config)
    data_cfg = fusion_cfg["data"]
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_cfg["instruments"], rollout_steps=0,
        interval_hours=int(data_cfg.get("interval_hours", 6)),
        normalize_latents=bool(data_cfg.get("normalize_latents", True)),
    )
    input_instruments = _csv(args.input_instruments)
    output_instruments = _csv(args.output_instruments)
    unknown = set(input_instruments) - set(dataset.names)
    if unknown:
        raise KeyError(f"Input instruments absent from fusion config: {sorted(unknown)}")
    if output_instruments != ["1bamua"]:
        raise NotImplementedError(
            "This first script has an AE decoder registry only for output instrument "
            "1bamua; use --output-instruments 1bamua"
        )

    bamua_raw = load_bamua_yaml(args.bamua_config)
    bamua_config = make_bamua_config(bamua_raw)
    channels = _channels(args.channels, bamua_config.n_channels)
    observation_path = bamua_raw.get("data", {}).get("zarr")
    if not observation_path:
        raise ValueError("BAMUA YAML must contain data.zarr")
    import zarr

    observation_root = zarr.open_group(str(observation_path), mode="r")
    observation_times = _time_int64(observation_root["time_series"][:])
    requested_time = _parse_time(args.start_time)
    if requested_time is None:
        observation_index = int(args.sample_index)
        if not 0 <= observation_index < len(observation_times):
            raise IndexError(f"sample-index outside [0, {len(observation_times) - 1}]")
        start_time = int(observation_times[observation_index])
    else:
        matches = np.flatnonzero(observation_times == requested_time)
        if len(matches) == 0:
            raise KeyError(f"No BAMUA 6-hour bin at time {requested_time}")
        observation_index = int(matches[0])
        start_time = requested_time
    start_matches = np.flatnonzero(dataset.start_times == start_time)
    if len(start_matches) == 0:
        raise KeyError(f"Start time {start_time} is absent from fusion latent stores")
    item = dataset[int(start_matches[0])]

    device = choose_device(args.device)
    amp_enabled = args.mixed_precision and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    fusion_model = build_model(fusion_cfg, dataset).to(device)
    fusion_checkpoint = load_checkpoint(args.fusion_checkpoint, "cpu")
    checkpoint_data = fusion_checkpoint.get("config", {}).get("data", {})
    if "normalize_latents" in checkpoint_data:
        trained_normalize = bool(checkpoint_data["normalize_latents"])
        if trained_normalize != dataset.normalize_latents:
            raise ValueError(
                "Fusion checkpoint and YAML disagree on normalize_latents: "
                f"checkpoint={trained_normalize}, YAML={dataset.normalize_latents}"
            )
    fusion_model.load_state_dict(fusion_checkpoint["model"])
    fusion_model.eval()
    bamua_model = BAMUAAutoEncoder(bamua_config).to(device)
    load_model_weights(
        bamua_model, load_checkpoint(args.bamua_checkpoint, "cpu")
    )
    bamua_model.eval()

    latents = {name: item["latents"][name][:1].to(device) for name in dataset.names}
    densities = {name: item["densities"][name][:1].to(device) for name in dataset.names}
    available = {name: item["available"][name][:1].to(device) for name in dataset.names}
    if args.encode_bamua_observations and "1bamua" in input_instruments:
        latent_raw, density, encoded_time = _encode_observation_bin(
            bamua_model, observation_root, observation_index,
            args.encode_chunk_size, device, amp_enabled, amp_dtype,
            max_observations=args.max_input_observations, seed=args.seed,
        )
        if encoded_time != start_time:
            raise RuntimeError("Encoded BAMUA time differs from selected start time")
        latents["1bamua"] = _raw_to_fusion_scale(
            dataset, "1bamua", latent_raw
        )
        densities["1bamua"] = density
        available["1bamua"] = torch.ones(1, dtype=torch.bool, device=device)
    for name in dataset.names:
        if name not in input_instruments:
            available[name] = torch.zeros(1, dtype=torch.bool, device=device)
    if not any(bool(value[0]) for value in available.values()):
        raise ValueError("None of the selected input instruments is available")

    standardized = _bt_is_standardized(observation_root)
    if standardized is None:
        raise ValueError("Cannot determine BT scale from observation Zarr attributes")
    bt_mean, bt_std = _bt_stats(observation_root, bamua_config.n_channels)
    if standardized and bt_mean is None:
        raise ValueError("Standardized BT data require channel_mean/channel_std")
    color_limits = (
        None if bt_mean is None else (
            bt_mean - args.color_std_range * bt_std,
            bt_mean + args.color_std_range * bt_std,
        )
    )

    # Use the most frequent satellite at t0 unless explicitly overridden.
    sample_start = int(observation_root["sample_start"][observation_index])
    sample_count = int(observation_root["sample_count"][observation_index])
    if sample_count < 1:
        raise ValueError(f"Selected BAMUA bin {observation_index} is empty")
    start_ids = _read(
        observation_root["satellite_id"],
        _query_indices(sample_start, sample_count, min(sample_count, 50_000)),
    ).astype(np.int64)
    unique_ids, id_counts = np.unique(start_ids, return_counts=True)
    unknown_ids = [
        int(value) for value in unique_ids if int(value) not in SATELLITE_NAMES
    ]
    if unknown_ids:
        print(
            "warning: no readable satellite-name mapping for IDs "
            f"{unknown_ids}; their plots will use UNKNOWN-SATELLITE"
        )
    default_satellite_id = (
        int(args.query_satellite_id) if args.query_satellite_id is not None
        else int(unique_ids[id_counts.argmax()])
    )
    # Count the observations represented by every available input latent. For
    # raw BAMUA encoding, use the actual sampled context count instead.
    context_used = 0
    context_total = 0
    latent_roots = dataset._open()
    for name in dataset.names:
        if not bool(available[name][0]):
            continue
        source_index = dataset._time_to_index[name].get(start_time)
        root = latent_roots[name]
        if source_index is None or "sample_count" not in root:
            continue
        instrument_total = int(root["sample_count"][source_index])
        instrument_used = instrument_total
        if name == "1bamua" and args.encode_bamua_observations:
            instrument_total = sample_count
            instrument_used = (
                sample_count if args.max_input_observations == 0
                else min(sample_count, args.max_input_observations)
            )
        context_total += instrument_total
        context_used += instrument_used
    # BAMUA's raw source always has a known count, including older latent stores
    # that may not yet contain their own sample_count array.
    if context_total == 0 and bool(available.get("1bamua", [False])[0]):
        context_total = sample_count
        context_used = (
            sample_count if args.max_input_observations == 0
            else min(sample_count, args.max_input_observations)
        )
    arbitrary_queries = _arbitrary_queries(args, default_satellite_id)
    grid_queries = _grid_queries(
        args.grid_resolution_deg, default_satellite_id,
        bool(args.query_is_land), args.grid_land_mask_npy,
    )
    worker = Path(__file__).with_name("end_to_end_plot_worker.py")
    time_to_observation = {int(value): i for i, value in enumerate(observation_times)}
    output_shapes = fusion_model.spatial_shapes(latents)
    interval_hours = int(data_cfg.get("interval_hours", 6))
    interval_ns = int(dataset.interval_ns)
    metrics = {}

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        state, _ = fusion_model.fuse(latents, densities, available)
        for lead in range(args.forecast_steps + 1):
            if lead > 0:
                state = fusion_model.forecast_state(state)
            decoded = fusion_model.decode_state(state, output_shapes)
            output_time = start_time + lead * interval_ns
            lead_dir = output_dir / f"lead_{lead:03d}_{lead * interval_hours:03d}h"
            lead_dir.mkdir(parents=True, exist_ok=True)

            latent_raw = dataset.denormalize(
                "1bamua", decoded["1bamua"]["latent"]
            ).float()
            source_index = time_to_observation.get(output_time)
            if source_index is not None:
                start = int(observation_root["sample_start"][source_index])
                count = int(observation_root["sample_count"][source_index])
                if count > 0:
                    indices = _query_indices(
                        start, count, args.max_target_observations
                    )
                    lon = _read(observation_root["longitude"], indices).astype(np.float32)
                    lat = _read(observation_root["latitude"], indices).astype(np.float32)
                    sat = _read(observation_root["satellite_id"], indices).astype(np.int64)
                    land = _read(observation_root["is_land"], indices).astype(bool)
                    target = _read(
                        observation_root["brightness_temperature"], indices
                    ).astype(np.float32)
                    valid = _read(
                        observation_root["brightness_temperature_valid"], indices
                    ).astype(bool)
                    prediction = _decode(
                        bamua_model, latent_raw, lon, lat, sat, land,
                        output_time, args.decode_chunk_size, device,
                        amp_enabled, amp_dtype,
                    )
                    target_phys = _physical(target, standardized, bt_mean, bt_std)
                    pred_phys = _physical(prediction, standardized, bt_mean, bt_std)
                    target_phys = np.where(valid, target_phys, np.nan)
                    comparison_path = lead_dir / "observation_positions.npz"
                    np.savez_compressed(
                        comparison_path, lon=lon, lat=lat, satellite_id=sat,
                        valid=valid, target=target_phys, prediction=pred_phys,
                    )
                    count_tag, count_text = _count_description(
                        context_used, context_total, len(indices), count
                    )
                    _plot(
                        worker, "comparison", comparison_path, lead_dir,
                        "1bamua", lead, interval_hours, channels,
                        args.point_size, color_limits=color_limits,
                        count_tag=count_tag, count_text=count_text,
                        output_time=output_time,
                    )
                    error = (pred_phys - target_phys) ** 2
                    metrics[f"lead_{lead:03d}"] = {
                        "lead_hours": lead * interval_hours,
                        "time_ns": output_time,
                        "time_utc": _readable_time(output_time),
                        "query_count": int(len(lon)),
                        "source_target_count": count,
                        "target_fraction": len(lon) / max(count, 1),
                        "physical_masked_rmse": float(
                            np.sqrt(np.nansum(error) / max(valid.sum(), 1))
                        ),
                    }
                else:
                    print(
                        f"warning: lead={lead} has an empty BAMUA target bin; "
                        "comparison plot is skipped"
                    )
            else:
                print(
                    f"warning: lead={lead} time_ns={output_time} has no BAMUA "
                    "target bin; comparison plot is skipped"
                )

            lon, lat, sat, land = arbitrary_queries
            arbitrary_pred = _decode(
                bamua_model, latent_raw, lon, lat, sat, land, output_time,
                args.decode_chunk_size, device, amp_enabled, amp_dtype,
            )
            arbitrary_pred = _physical(
                arbitrary_pred, standardized, bt_mean, bt_std
            )
            arbitrary_path = lead_dir / "arbitrary_global_points.npz"
            np.savez_compressed(
                arbitrary_path, lon=lon, lat=lat, satellite_id=sat,
                is_land=land, prediction=arbitrary_pred,
            )
            count_tag, count_text = _count_description(
                context_used, context_total, len(lon)
            )
            _plot(
                worker, "arbitrary", arbitrary_path, lead_dir, "1bamua",
                lead, interval_hours, channels, args.point_size,
                color_limits=color_limits,
                count_tag=count_tag, count_text=count_text,
                output_time=output_time,
            )

            lon_grid, lat_grid, lon, lat, sat, land, grid_shape = grid_queries
            grid_pred = _decode(
                bamua_model, latent_raw, lon, lat, sat, land, output_time,
                args.decode_chunk_size, device, amp_enabled, amp_dtype,
            )
            grid_pred = _physical(
                grid_pred, standardized, bt_mean, bt_std
            ).reshape(*grid_shape, bamua_config.n_channels)
            grid_path = lead_dir / "regular_global_grid.npz"
            np.savez_compressed(
                grid_path, lon_grid=lon_grid, lat_grid=lat_grid,
                satellite_id=sat.reshape(grid_shape), prediction=grid_pred,
            )
            count_tag, count_text = _count_description(
                context_used, context_total, len(lon)
            )
            _plot(
                worker, "grid", grid_path, lead_dir, "1bamua", lead,
                interval_hours, channels, args.point_size,
                resolution_deg=args.grid_resolution_deg,
                color_limits=color_limits,
                count_tag=count_tag, count_text=count_text,
                output_time=output_time,
            )
            print(
                f"lead={lead} time={np.datetime64(output_time, 'ns')} "
                f"output={lead_dir}"
            )

    summary = {
        "inference_config": (
            str(Path(args.config).resolve()) if args.config else None
        ),
        "fusion_config": str(Path(args.fusion_config).resolve()),
        "fusion_checkpoint": str(Path(args.fusion_checkpoint).resolve()),
        "fusion_checkpoint_epoch": fusion_checkpoint.get("epoch"),
        "bamua_config": str(Path(args.bamua_config).resolve()),
        "bamua_checkpoint": str(Path(args.bamua_checkpoint).resolve()),
        "start_sample_index": observation_index,
        "start_time_ns": start_time,
        "start_time_utc": _readable_time(start_time),
        "input_instruments": input_instruments,
        "input_available": {
            name: bool(available[name][0]) for name in dataset.names
        },
        "context_observations_used": context_used,
        "context_observations_total": context_total,
        "context_observation_fraction": (
            context_used / max(context_total, 1)
        ),
        "output_instruments": output_instruments,
        "forecast_steps": args.forecast_steps,
        "interval_hours": interval_hours,
        "grid_resolution_deg": args.grid_resolution_deg,
        "query_satellite": SATELLITE_NAMES.get(
            default_satellite_id, "UNKNOWN-SATELLITE"
        ),
        "metrics": metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
