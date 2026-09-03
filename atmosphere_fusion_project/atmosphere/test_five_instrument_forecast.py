"""Five-instrument observations -> AE encoders -> fusion/forecast -> observations.

The four radiance instruments share BAMUAAutoEncoder's architecture. GPSRO
uses its own 3-D SetConv autoencoder. Real observations after the initial time
are used only as decoder query locations and verification targets.
"""

from __future__ import annotations

import argparse
import json
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
    _bt_is_standardized,
    _bt_stats,
    _query_indices,
    _read,
    _time_int64,
)
from atmosphere.test_ae_fusion_forecast import (
    _count_description,
    _decode,
    _filename_time,
    _physical,
    _plot,
    _readable_time,
)
from atmosphere.train_fusion import build_model, choose_device, load_checkpoint
from atmosphere.utils import amp_dtype_from_name
from satellite.datasets import GPSROZarrDataset
from satellite.export_gpsro_latents import config_from_checkpoint
from satellite.gpsro_plotting import (
    decode_global_3d,
    load_refractivity_stats,
    save_global_3d,
    save_reconstruction_3d,
    transform_refractivity,
)
from satellite.models import BAMUAAutoEncoder, GPSROAutoEncoder
from satellite.train_bamua import (
    load_model_weights,
    load_yaml_config as load_radiance_yaml,
    make_bamua_config,
)
from satellite.train_gpsro import load_yaml as load_gpsro_yaml


GPSRO_NAME = "gpsro"
# GPSRO figures and saved plotting values are always inverse-standardized to
# log(N). Keep model tensors in standardized space for decoding and metrics.
GPSRO_PLOT_VALUE_SPACE = "log"


def _value_for_instrument(section, key, name, default):
    value = section.get(key, default)
    return value.get(name, default) if isinstance(value, dict) else value


def _time_mapping(root):
    key = "time" if "time" in root and root["time"].ndim == 1 else "time_series"
    if "time_series" in root:
        key = "time_series"
    times = _time_int64(root[key][:])
    return {int(value): index for index, value in enumerate(times.tolist())}


def _selected_indices(start, count, maximum, seed):
    if maximum <= 0 or count <= maximum:
        return slice(start, start + count), count
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(count, size=maximum, replace=False)).astype(np.int64)
    return indices + start, len(indices)


def _load_models(config, device):
    registry = config["models"].get("instrument_autoencoders", {})
    models, raw_configs, checkpoints = {}, {}, {}
    expected = {"1bamua", "atms", "1bhrs4", "1bmhs", GPSRO_NAME}
    missing = expected - set(registry)
    if missing:
        raise KeyError(f"Missing AE registry entries: {sorted(missing)}")

    for name, paths in registry.items():
        checkpoint = load_checkpoint(paths["checkpoint"], "cpu")
        if name == GPSRO_NAME:
            raw = load_gpsro_yaml(paths["config"])
            model_config, raw = config_from_checkpoint(raw, checkpoint)
            model = GPSROAutoEncoder(model_config)
            state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
            model.load_state_dict(state)
        else:
            raw = load_radiance_yaml(paths["config"])
            model = BAMUAAutoEncoder(make_bamua_config(raw))
            load_model_weights(model, checkpoint)
        models[name] = model.to(device).eval()
        raw_configs[name] = raw
        checkpoints[name] = checkpoint
        print(f"loaded_ae={name} checkpoint={paths['checkpoint']}")
    return models, raw_configs, checkpoints


@torch.no_grad()
def _encode_gpsro(model, raw_path, root, sample_index, maximum, chunk_size,
                  seed, device, amp_enabled, amp_dtype):
    start = int(root["sample_start"][sample_index])
    count = int(root["sample_count"][sample_index])
    sample_time = int(_time_int64(root["time_series"][sample_index]))
    if count < 1:
        raise ValueError(f"GPSRO bin {sample_index} is empty")
    indices, selected_count = _selected_indices(
        start, count, int(maximum), int(seed) + sample_index
    )
    dataset = GPSROZarrDataset(raw_path, n_context=1, n_target=1)
    latent_sum = density_sum = None
    for offset in range(0, selected_count, chunk_size):
        end = min(offset + chunk_size, selected_count)
        chunk_indices = (
            slice(start + offset, start + end)
            if isinstance(indices, slice) else indices[offset:end]
        )
        points = dataset._points(root, chunk_indices, sample_time, include_value=True)
        points = {
            key: value.unsqueeze(0).to(device, non_blocking=True)
            for key, value in points.items()
        }
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
        density_sum = chunk_density if density_sum is None else density_sum + chunk_density
    latent_3d = latent_sum / density_sum.clamp_min(model.config.eps)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        latent_2d = model.latent_processor(latent_3d)
    # The fusion adapter expects one horizontal confidence map. This is the
    # vertical sum used by export_gpsro_latents.py as its saved density.
    density_2d = density_sum.sum(dim=2)
    return latent_2d.float(), density_2d.float(), sample_time, count, selected_count


@torch.no_grad()
def _decode_gpsro(model, latent, lon, lat, height, satellite_id, is_land,
                  sample_time, device, amp_enabled, amp_dtype):
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        prediction = model.decode(
            latent=latent,
            lon=torch.from_numpy(lon).to(device).unsqueeze(0),
            lat=torch.from_numpy(lat).to(device).unsqueeze(0),
            height=torch.from_numpy(height).to(device).unsqueeze(0),
            satellite_id=torch.from_numpy(satellite_id).to(device).unsqueeze(0),
            is_land=torch.from_numpy(is_land).to(device).unsqueeze(0),
            sample_time=torch.tensor([sample_time], dtype=torch.long, device=device),
        )
    prediction = prediction[0, :, 0].float().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError("GPSRO decoder produced NaN/Inf")
    return prediction


def _masked_rmse(prediction, target, valid):
    mask = np.asarray(valid, dtype=bool)
    mask &= np.isfinite(prediction) & np.isfinite(target)
    if not mask.any():
        return None
    return float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2)))


def _random_horizontal_points(count, seed):
    """Sample longitude/latitude uniformly over the sphere."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, int(count)).astype(np.float32)
    lat = np.rad2deg(
        np.arcsin(rng.uniform(-1.0, 1.0, int(count)))
    ).astype(np.float32)
    return lon, lat, rng


def _regular_horizontal_grid(resolution):
    resolution = float(resolution)
    if resolution <= 0:
        raise ValueError("generated_queries.grid_resolution_deg must be positive")
    latitude = np.linspace(
        -90.0, 90.0, round(180.0 / resolution) + 1, dtype=np.float32
    )
    longitude = np.linspace(
        -180.0, 180.0, round(360.0 / resolution),
        endpoint=False, dtype=np.float32,
    )
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    return lon_grid, lat_grid, lon_grid.reshape(-1), lat_grid.reshape(-1)


def _generated_satellite_id(name, root, time_map, output_time,
                            generated_config):
    """Choose decoder metadata when no target observations exist.

    A user-supplied ID is preferred. Otherwise use the most common satellite
    ID in the closest non-empty 6-hour bin for this instrument. The ID affects
    decoder metadata only; no observations from that fallback bin enter the
    fusion state or the verification loss.
    """
    configured = generated_config.get("satellite_id")
    if isinstance(configured, dict):
        configured = configured.get(name)
    if configured is not None:
        return int(configured), "configured"

    candidates = sorted(time_map.items(), key=lambda item: abs(item[0] - output_time))
    for _, sample_index in candidates:
        count = int(root["sample_count"][sample_index])
        if count < 1:
            continue
        start = int(root["sample_start"][sample_index])
        satellite_ids = _read(
            root["satellite_id"], slice(start, start + count)
        ).astype(np.int64)
        satellite_ids = satellite_ids[satellite_ids >= 0]
        if satellite_ids.size:
            values, counts = np.unique(satellite_ids, return_counts=True)
            return int(values[np.argmax(counts)]), "nearest_nonempty_bin"
    raise ValueError(
        f"Cannot generate {name}: no satellite_id was configured and its "
        "observation store has no non-empty bin"
    )


def _save_unverified_generated_outputs(
        name, model, latent_raw, root, raw_config, output_time,
        instrument_dir, lead, interval_hours, sampling, plot_config,
        generated_config, input_counts, output_names, seed, device,
        amp_enabled, amp_dtype, radiance_worker, default_id, id_source):
    """Save arbitrary/global predictions when this lead has no target.

    These products contain predictions only. They intentionally do not create
    target/difference plots and do not report an RMSE.
    """
    if not generated_config.get("enabled", True):
        return None

    arbitrary_count = int(generated_config.get("arbitrary_point_count", 20000))

    if name == GPSRO_NAME:
        stats = load_refractivity_stats(raw_config["data"]["zarr"])
        value_space = GPSRO_PLOT_VALUE_SPACE
        random_lon, random_lat, rng = _random_horizontal_points(
            arbitrary_count, seed + 1000 * lead + 97
        )
        random_height = rng.uniform(
            model.config.vertical_min_m,
            model.config.vertical_max_m,
            arbitrary_count,
        ).astype(np.float32)
        random_land = GPSROZarrDataset._is_land(random_lon, random_lat)
        random_satellite = np.full(arbitrary_count, default_id, dtype=np.int64)
        random_prediction = _decode_gpsro(
            model, latent_raw, random_lon, random_lat, random_height,
            random_satellite, random_land, output_time, device,
            amp_enabled, amp_dtype,
        )
        random_values, _ = transform_refractivity(
            random_prediction, random_height, stats, value_space
        )
        np.savez_compressed(
            instrument_dir / "arbitrary_global_points.npz",
            lon=random_lon, lat=random_lat, height=random_height,
            satellite_id=random_satellite, is_land=random_land,
            prediction=random_values,
            prediction_standardized=random_prediction,
        )
        save_global_3d(
            random_lon, random_lat, random_height, random_prediction,
            instrument_dir, "arbitrary_gpsro", stats,
            value_space=value_space,
            max_points=int(plot_config.get("max_points", 50000)),
            point_size=float(plot_config.get("point_size", 3.0)),
            seed=seed + lead,
            title=(
                f"GPSRO arbitrary queries lead={lead} "
                f"({lead * interval_hours} h), no verification target"
            ),
        )

        configured_heights = generated_config.get("gpsro_global_heights_m")
        if configured_heights is None:
            # The upper bound is exclusive in the height-bin statistics.
            # Do not query the unsupported vertical_max_m boundary level.
            configured_heights = np.arange(
                model.config.vertical_min_m,
                model.config.vertical_max_m,
                model.config.vertical_resolution_m,
                dtype=np.float32,
            )
        grid_lon, grid_lat, grid_height, grid_prediction = decode_global_3d(
            model, latent_raw, output_time, default_id,
            float(generated_config.get("grid_resolution_deg", 2.0)),
            configured_heights,
            int(_value_for_instrument(
                sampling, "decode_chunk_size", name, 8192
            )),
            device, amp_enabled, amp_dtype,
        )
        grid_prediction = grid_prediction[:, 0]
        grid_values, _ = transform_refractivity(
            grid_prediction, grid_height, stats, value_space
        )
        np.savez_compressed(
            instrument_dir / "regular_global_grid_3d.npz",
            lon=grid_lon, lat=grid_lat, height=grid_height,
            satellite_id=np.full(len(grid_lon), default_id, dtype=np.int64),
            prediction=grid_values,
            prediction_standardized=grid_prediction,
        )
        save_global_3d(
            grid_lon, grid_lat, grid_height, grid_prediction,
            instrument_dir, "regular_gpsro", stats,
            value_space=value_space,
            max_points=int(plot_config.get("max_points", 50000)),
            point_size=float(plot_config.get("point_size", 3.0)),
            seed=seed + lead,
            title=(
                f"GPSRO regular 3-D grid lead={lead} "
                f"({lead * interval_hours} h), no verification target"
            ),
        )
    else:
        standardized = _bt_is_standardized(root)
        mean, std = _bt_stats(root, model.config.n_channels)
        configured_channels = plot_config.get("channels", {})
        channel_values = (
            configured_channels.get(name, [1])
            if isinstance(configured_channels, dict) else configured_channels
        )
        channels = [int(value) - 1 for value in channel_values]
        if any(channel < 0 or channel >= model.config.n_channels
               for channel in channels):
            raise ValueError(f"Invalid plot channel for {name}")
        limits = None if mean is None else (
            mean - float(plot_config.get("color_std_range", 3.0)) * std,
            mean + float(plot_config.get("color_std_range", 3.0)) * std,
        )
        random_lon, random_lat, _ = _random_horizontal_points(
            arbitrary_count,
            seed + 1000 * lead + 31 * (output_names.index(name) + 1),
        )
        random_land = GPSROZarrDataset._is_land(random_lon, random_lat)
        random_satellite = np.full(arbitrary_count, default_id, dtype=np.int64)
        random_prediction = _decode(
            model, latent_raw, random_lon, random_lat, random_satellite,
            random_land, output_time,
            int(_value_for_instrument(
                sampling, "decode_chunk_size", name, 8192
            )),
            device, amp_enabled, amp_dtype,
        )
        random_prediction = _physical(
            random_prediction, standardized, mean, std
        )
        random_path = instrument_dir / "arbitrary_global_points.npz"
        np.savez_compressed(
            random_path, lon=random_lon, lat=random_lat,
            satellite_id=random_satellite, is_land=random_land,
            prediction=random_prediction,
        )
        used = input_counts.get(name, {}).get("used", 0)
        total = input_counts.get(name, {}).get("total", 0)
        arbitrary_tag, arbitrary_text = _count_description(
            used, total, arbitrary_count
        )
        _plot(
            radiance_worker, "arbitrary", random_path, instrument_dir,
            name, lead, interval_hours, channels,
            float(plot_config.get("point_size", 3.0)),
            color_limits=limits, count_tag=arbitrary_tag,
            count_text=arbitrary_text, output_time=output_time,
        )

        lon_grid, lat_grid, flat_lon, flat_lat = _regular_horizontal_grid(
            generated_config.get("grid_resolution_deg", 2.0)
        )
        grid_land = GPSROZarrDataset._is_land(flat_lon, flat_lat)
        grid_satellite = np.full(len(flat_lon), default_id, dtype=np.int64)
        grid_prediction = _decode(
            model, latent_raw, flat_lon, flat_lat, grid_satellite, grid_land,
            output_time,
            int(_value_for_instrument(
                sampling, "decode_chunk_size", name, 8192
            )),
            device, amp_enabled, amp_dtype,
        )
        grid_prediction = _physical(
            grid_prediction, standardized, mean, std
        ).reshape(*lon_grid.shape, model.config.n_channels)
        grid_path = instrument_dir / "regular_global_grid.npz"
        np.savez_compressed(
            grid_path, lon_grid=lon_grid, lat_grid=lat_grid,
            satellite_id=grid_satellite.reshape(lon_grid.shape),
            is_land=grid_land.reshape(lon_grid.shape),
            prediction=grid_prediction,
        )
        grid_tag, grid_text = _count_description(
            used, total, len(flat_lon)
        )
        _plot(
            radiance_worker, "grid", grid_path, instrument_dir,
            name, lead, interval_hours, channels,
            float(plot_config.get("point_size", 3.0)),
            resolution_deg=float(generated_config.get(
                "grid_resolution_deg", 2.0
            )),
            color_limits=limits, count_tag=grid_tag,
            count_text=grid_text, output_time=output_time,
        )

    return {
        "satellite_id": int(default_id),
        "satellite_id_source": id_source,
        "arbitrary_query_count": arbitrary_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_section = config["models"]
    selection = config.get("selection", {})
    sampling = config.get("sampling", {})
    plot_config = config.get("plot", {})
    generated_config = config.get("generated_queries", {})
    runtime = config.get("runtime", {})

    output_dir = Path(config["output"]["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not config["output"].get("overwrite", False):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(runtime.get("device", "auto"))
    amp_enabled = bool(runtime.get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_name(runtime.get("amp_dtype", "bfloat16"))
    seed = int(runtime.get("seed", 42))

    fusion_config = load_yaml(model_section["fusion_config"])
    data_config = fusion_config["data"]
    dataset = MultiInstrumentLatentSequenceDataset(
        stores=data_config["instruments"], rollout_steps=0,
        interval_hours=int(data_config.get("interval_hours", 6)),
        normalize_latents=bool(data_config.get("normalize_latents", True)),
    )
    fusion_model = build_model(fusion_config, dataset).to(device)
    fusion_checkpoint = load_checkpoint(model_section["fusion_checkpoint"], "cpu")
    fusion_model.load_state_dict(fusion_checkpoint["model"])
    fusion_model.eval()

    ae_models, raw_configs, _ = _load_models(config, device)
    import zarr
    roots, time_maps = {}, {}
    for name, raw in raw_configs.items():
        raw_path = raw.get("data", {}).get("zarr")
        if not raw_path:
            raise ValueError(f"{name} AE config has no data.zarr")
        roots[name] = zarr.open_group(str(raw_path), mode="r")
        time_maps[name] = _time_mapping(roots[name])

    reference_name = str(selection.get("reference_instrument", "1bamua"))
    reference_root = roots[reference_name]
    requested_time = selection.get("start_time")
    if requested_time is None:
        reference_index = int(selection.get("sample_index", 0))
        start_time = int(_time_int64(reference_root["time_series"][reference_index]))
    else:
        try:
            start_time = int(requested_time)
        except (TypeError, ValueError):
            start_time = int(np.datetime64(requested_time, "ns").astype(np.int64))

    input_names = [str(name) for name in selection.get("input_instruments", dataset.names)]
    output_names = [str(name) for name in selection.get("output_instruments", dataset.names)]
    unknown = (set(input_names) | set(output_names)) - set(dataset.names)
    if unknown:
        raise KeyError(f"Instruments absent from fusion config: {sorted(unknown)}")

    latents, densities, available = {}, {}, {}
    input_counts = {}
    for name, spec in dataset.specs.items():
        source_index = time_maps[name].get(start_time)
        source_count = (
            int(roots[name]["sample_count"][source_index])
            if source_index is not None else 0
        )
        # A regular 6-hour time entry can legitimately exist with zero
        # observations. Treat both a missing timestamp and an empty bin as an
        # unavailable instrument, leaving a correctly shaped zero placeholder.
        if name not in input_names or source_index is None or source_count < 1:
            latents[name] = torch.zeros(1, spec.latent_dim, spec.height, spec.width, device=device)
            densities[name] = torch.zeros(1, 1, spec.height, spec.width, device=device)
            available[name] = torch.zeros(1, dtype=torch.bool, device=device)
            input_counts[name] = {"used": 0, "total": source_count}
            reason = "not selected" if name not in input_names else (
                "missing time" if source_index is None else "empty 6-hour bin"
            )
            print(f"unavailable_input={name} reason={reason}")
            continue
        sample_index = source_index
        maximum = int(_value_for_instrument(
            sampling, "max_input_observations", name, 0
        ))
        chunk_size = int(_value_for_instrument(
            sampling, "encode_chunk_size", name, 16384
        ))
        if name == GPSRO_NAME:
            raw_path = raw_configs[name]["data"]["zarr"]
            latent_raw, density, encoded_time, total, used = _encode_gpsro(
                ae_models[name], raw_path, roots[name], sample_index,
                maximum, chunk_size, seed, device, amp_enabled, amp_dtype,
            )
        else:
            latent_raw, density, encoded_time = _encode_observation_bin(
                ae_models[name], roots[name], sample_index, chunk_size,
                device, amp_enabled, amp_dtype,
                max_observations=maximum, seed=seed,
            )
            total = int(roots[name]["sample_count"][sample_index])
            used = total if maximum <= 0 else min(total, maximum)
        if encoded_time != start_time:
            raise RuntimeError(f"{name} encoded time differs from start time")
        if tuple(latent_raw.shape[-2:]) != (spec.height, spec.width):
            raise ValueError(
                f"{name} AE latent grid {tuple(latent_raw.shape[-2:])} does not "
                f"match fusion grid {(spec.height, spec.width)}"
            )
        latents[name] = _raw_to_fusion_scale(dataset, name, latent_raw)
        densities[name] = density
        available[name] = torch.ones(1, dtype=torch.bool, device=device)
        input_counts[name] = {"used": used, "total": total}
        print(f"encoded_input={name} time={_readable_time(start_time)} used={used}/{total}")

    missing_inputs = [name for name in input_names if not bool(available[name][0])]
    if selection.get("require_all_inputs", True) and missing_inputs:
        raise ValueError(
            f"Selected start time lacks required input observations: {missing_inputs}"
        )
    if not any(bool(value[0]) for value in available.values()):
        raise ValueError("No selected input instrument is available")

    output_shapes = fusion_model.spatial_shapes(latents)
    interval_hours = int(data_config.get("interval_hours", 6))
    steps = int(config.get("forecast", {}).get("steps", 0))
    maximum_targets = sampling.get("max_target_observations", 0)
    metrics = {}
    radiance_worker = Path(__file__).with_name("end_to_end_plot_worker.py")

    # Hold generated-query decoder metadata fixed through the rollout. This
    # ensures lead-to-lead changes describe the forecast state rather than a
    # different satellite embedding. Verification at real observation points
    # still uses each observation's own satellite_id below.
    generated_satellites = {}
    if generated_config.get("enabled", True):
        for name in output_names:
            satellite_id, id_source = _generated_satellite_id(
                name, roots[name], time_maps[name], start_time,
                generated_config,
            )
            generated_satellites[name] = {
                "satellite_id": int(satellite_id),
                "source": id_source,
            }
            print(
                f"fixed_generated_satellite={name} "
                f"satellite_id={satellite_id} source={id_source}"
            )

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        state, fusion_weights = fusion_model.fuse(latents, densities, available)
        for lead in range(steps + 1):
            if lead > 0:
                state = fusion_model.forecast_state(state)
            instrument_latents = fusion_model.decode_state(state, output_shapes)
            output_time = start_time + lead * dataset.interval_ns
            lead_dir = output_dir / f"lead_{lead:03d}_{lead * interval_hours:03d}h"
            lead_dir.mkdir(parents=True, exist_ok=True)
            metrics[f"lead_{lead:03d}"] = {}

            for name in output_names:
                root = roots[name]
                source_index = time_maps[name].get(output_time)
                count = (
                    int(root["sample_count"][source_index])
                    if source_index is not None else 0
                )
                target_available = source_index is not None and count > 0

                # Decoding a generated field does not require a verification
                # target. Always prepare the predicted instrument latent first.
                latent_raw = dataset.denormalize(
                    name, instrument_latents[name]["latent"]
                ).float()
                instrument_dir = lead_dir / name
                instrument_dir.mkdir(parents=True, exist_ok=True)

                if not target_available:
                    reason = (
                        "no timestamp" if source_index is None
                        else "empty 6-hour bin"
                    )
                    print(
                        f"no_target={name} lead={lead} reason={reason} "
                        f"time={_readable_time(output_time)}"
                    )
                    generated = None
                    if generated_config.get("enabled", True):
                        generated = _save_unverified_generated_outputs(
                            name=name,
                            model=ae_models[name],
                            latent_raw=latent_raw,
                            root=root,
                            raw_config=raw_configs[name],
                            output_time=output_time,
                            instrument_dir=instrument_dir,
                            lead=lead,
                            interval_hours=interval_hours,
                            sampling=sampling,
                            plot_config=plot_config,
                            generated_config=generated_config,
                            input_counts=input_counts,
                            output_names=output_names,
                            seed=seed,
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            radiance_worker=radiance_worker,
                            default_id=(
                                generated_satellites[name]["satellite_id"]
                            ),
                            id_source=generated_satellites[name]["source"],
                        )
                    metrics[f"lead_{lead:03d}"][name] = {
                        "time_ns": int(output_time),
                        "time_utc": _readable_time(output_time),
                        "target_available": False,
                        "query_count": 0,
                        "source_count": int(count),
                        "standardized_masked_rmse": None,
                        "physical_masked_rmse": None,
                        "generated": generated,
                    }
                    print(
                        f"generated_without_target={name} lead={lead} "
                        f"enabled={generated is not None}"
                    )
                    continue

                start = int(root["sample_start"][source_index])
                maximum = int(_value_for_instrument(
                    sampling, "max_target_observations", name, 0
                ))
                indices = _query_indices(start, count, maximum)
                lon = _read(root["longitude"], indices).astype(np.float32)
                lat = _read(root["latitude"], indices).astype(np.float32)
                satellite_id = _read(root["satellite_id"], indices).astype(np.int64)

                if name == GPSRO_NAME:
                    height = _read(root["height_m"], indices).astype(np.float32)
                    is_land = GPSROZarrDataset._is_land(lon, lat)
                    target = _read(root["refractivity"], indices).astype(np.float32)
                    valid = np.isfinite(target)
                    prediction = _decode_gpsro(
                        ae_models[name], latent_raw, lon, lat, height,
                        satellite_id, is_land, output_time, device,
                        amp_enabled, amp_dtype,
                    )
                    stats = load_refractivity_stats(raw_configs[name]["data"]["zarr"])
                    value_space = GPSRO_PLOT_VALUE_SPACE
                    target_plot, value_label = transform_refractivity(
                        target, height, stats, value_space
                    )
                    prediction_plot, _ = transform_refractivity(
                        prediction, height, stats, value_space
                    )
                    data_path = instrument_dir / "observation_positions.npz"
                    np.savez_compressed(
                        data_path, lon=lon, lat=lat, height=height,
                        satellite_id=satellite_id, valid=valid,
                        truth=target_plot, prediction=prediction_plot,
                        target_standardized=target,
                        prediction_standardized=prediction,
                    )
                    plot_path = instrument_dir / "comparison_gpsro_3d.png"
                    save_reconstruction_3d(
                        lon, lat, height, target, prediction, valid,
                        instrument_dir, "comparison_gpsro_3d", stats,
                        value_space=value_space,
                        max_points=int(plot_config.get("max_points", 50000)),
                        point_size=float(plot_config.get("point_size", 3.0)),
                        seed=seed + lead,
                        title_prefix=(
                            f"GPSRO lead={lead} ({lead * interval_hours} h) "
                            f"time={_readable_time(output_time)}"
                        ),
                    )
                    if generated_config.get("enabled", True):
                        # Use one fixed satellite embedding for every lead.
                        default_id = generated_satellites[name]["satellite_id"]
                        arbitrary_count = int(
                            generated_config.get("arbitrary_point_count", 20000)
                        )
                        random_lon, random_lat, rng = _random_horizontal_points(
                            arbitrary_count, seed + 1000 * lead + 97
                        )
                        random_height = rng.uniform(
                            ae_models[name].config.vertical_min_m,
                            ae_models[name].config.vertical_max_m,
                            arbitrary_count,
                        ).astype(np.float32)
                        random_land = GPSROZarrDataset._is_land(
                            random_lon, random_lat
                        )
                        random_satellite = np.full(
                            arbitrary_count, default_id, dtype=np.int64
                        )
                        random_prediction = _decode_gpsro(
                            ae_models[name], latent_raw, random_lon, random_lat,
                            random_height, random_satellite, random_land,
                            output_time, device, amp_enabled, amp_dtype,
                        )
                        random_values, _ = transform_refractivity(
                            random_prediction, random_height, stats, value_space
                        )
                        np.savez_compressed(
                            instrument_dir / "arbitrary_global_points.npz",
                            lon=random_lon, lat=random_lat,
                            height=random_height,
                            satellite_id=random_satellite,
                            is_land=random_land,
                            prediction=random_values,
                            prediction_standardized=random_prediction,
                        )
                        save_global_3d(
                            random_lon, random_lat, random_height,
                            random_prediction, instrument_dir,
                            "arbitrary_gpsro", stats,
                            value_space=value_space,
                            max_points=int(plot_config.get("max_points", 50000)),
                            point_size=float(plot_config.get("point_size", 3.0)),
                            seed=seed + lead,
                            title=(
                                f"GPSRO arbitrary queries lead={lead} "
                                f"({lead * interval_hours} h)"
                            ),
                        )

                        configured_heights = generated_config.get(
                            "gpsro_global_heights_m"
                        )
                        if configured_heights is None:
                            # Match the half-open training height range
                            # [vertical_min_m, vertical_max_m).
                            configured_heights = np.arange(
                                ae_models[name].config.vertical_min_m,
                                ae_models[name].config.vertical_max_m,
                                ae_models[name].config.vertical_resolution_m,
                                dtype=np.float32,
                            )
                        grid_lon, grid_lat, grid_height, grid_prediction = (
                            decode_global_3d(
                                ae_models[name], latent_raw, output_time,
                                default_id,
                                float(generated_config.get(
                                    "grid_resolution_deg", 2.0
                                )),
                                configured_heights,
                                int(_value_for_instrument(
                                    sampling, "decode_chunk_size", name, 8192
                                )),
                                device, amp_enabled, amp_dtype,
                            )
                        )
                        grid_prediction = grid_prediction[:, 0]
                        grid_values, _ = transform_refractivity(
                            grid_prediction, grid_height, stats, value_space
                        )
                        np.savez_compressed(
                            instrument_dir / "regular_global_grid_3d.npz",
                            lon=grid_lon, lat=grid_lat, height=grid_height,
                            satellite_id=np.full(
                                len(grid_lon), default_id, dtype=np.int64
                            ),
                            prediction=grid_values,
                            prediction_standardized=grid_prediction,
                        )
                        save_global_3d(
                            grid_lon, grid_lat, grid_height, grid_prediction,
                            instrument_dir, "regular_gpsro", stats,
                            value_space=value_space,
                            max_points=int(plot_config.get("max_points", 50000)),
                            point_size=float(plot_config.get("point_size", 3.0)),
                            seed=seed + lead,
                            title=(
                                f"GPSRO regular 3-D grid lead={lead} "
                                f"({lead * interval_hours} h)"
                            ),
                        )
                    metric_prediction, metric_target, metric_valid = prediction, target, valid
                    extra_metric = {
                        "output_space": value_space,
                        "output_space_rmse": _masked_rmse(
                            prediction_plot, target_plot, valid
                        ),
                    }
                else:
                    is_land = _read(root["is_land"], indices).astype(bool)
                    target = _read(root["brightness_temperature"], indices).astype(np.float32)
                    valid = _read(root["brightness_temperature_valid"], indices).astype(bool)
                    prediction = _decode(
                        ae_models[name], latent_raw, lon, lat, satellite_id,
                        is_land, output_time,
                        int(_value_for_instrument(
                            sampling, "decode_chunk_size", name, 16384
                        )), device, amp_enabled, amp_dtype,
                    )
                    standardized = _bt_is_standardized(root)
                    mean, std = _bt_stats(root, ae_models[name].config.n_channels)
                    target_plot = _physical(target, standardized, mean, std)
                    prediction_plot = _physical(prediction, standardized, mean, std)
                    target_plot = np.where(valid, target_plot, np.nan)
                    data_path = instrument_dir / "observation_positions.npz"
                    np.savez_compressed(
                        data_path, lon=lon, lat=lat,
                        satellite_id=satellite_id, valid=valid,
                        target=target_plot, prediction=prediction_plot,
                    )
                    configured_channels = plot_config.get("channels", {})
                    channel_values = (
                        configured_channels.get(name, [1])
                        if isinstance(configured_channels, dict)
                        else configured_channels
                    )
                    channels = [int(value) - 1 for value in channel_values]
                    if any(c < 0 or c >= ae_models[name].config.n_channels for c in channels):
                        raise ValueError(f"Invalid plot channel for {name}")
                    limits = None if mean is None else (
                        mean - float(plot_config.get("color_std_range", 3.0)) * std,
                        mean + float(plot_config.get("color_std_range", 3.0)) * std,
                    )
                    used = input_counts.get(name, {}).get("used", 0)
                    total = input_counts.get(name, {}).get("total", 0)
                    count_tag, count_text = _count_description(
                        used, total, len(lon), count
                    )
                    _plot(
                        radiance_worker, "comparison", data_path,
                        instrument_dir, name, lead, interval_hours,
                        channels, float(plot_config.get("point_size", 3.0)),
                        color_limits=limits, count_tag=count_tag,
                        count_text=count_text, output_time=output_time,
                    )
                    if generated_config.get("enabled", True):
                        # Use one fixed satellite embedding for every lead.
                        default_id = generated_satellites[name]["satellite_id"]
                        arbitrary_count = int(
                            generated_config.get("arbitrary_point_count", 20000)
                        )
                        random_lon, random_lat, _ = _random_horizontal_points(
                            arbitrary_count,
                            seed + 1000 * lead + 31 * (output_names.index(name) + 1),
                        )
                        random_land = GPSROZarrDataset._is_land(
                            random_lon, random_lat
                        )
                        random_satellite = np.full(
                            arbitrary_count, default_id, dtype=np.int64
                        )
                        random_prediction = _decode(
                            ae_models[name], latent_raw, random_lon, random_lat,
                            random_satellite, random_land, output_time,
                            int(_value_for_instrument(
                                sampling, "decode_chunk_size", name, 8192
                            )), device, amp_enabled, amp_dtype,
                        )
                        random_prediction = _physical(
                            random_prediction, standardized, mean, std
                        )
                        random_path = instrument_dir / "arbitrary_global_points.npz"
                        np.savez_compressed(
                            random_path, lon=random_lon, lat=random_lat,
                            satellite_id=random_satellite,
                            is_land=random_land,
                            prediction=random_prediction,
                        )
                        arbitrary_tag, arbitrary_text = _count_description(
                            used, total, arbitrary_count
                        )
                        _plot(
                            radiance_worker, "arbitrary", random_path,
                            instrument_dir, name, lead, interval_hours,
                            channels, float(plot_config.get("point_size", 3.0)),
                            color_limits=limits, count_tag=arbitrary_tag,
                            count_text=arbitrary_text, output_time=output_time,
                        )

                        lon_grid, lat_grid, flat_lon, flat_lat = (
                            _regular_horizontal_grid(
                                generated_config.get("grid_resolution_deg", 2.0)
                            )
                        )
                        grid_land = GPSROZarrDataset._is_land(
                            flat_lon, flat_lat
                        )
                        grid_satellite = np.full(
                            len(flat_lon), default_id, dtype=np.int64
                        )
                        grid_prediction = _decode(
                            ae_models[name], latent_raw, flat_lon, flat_lat,
                            grid_satellite, grid_land, output_time,
                            int(_value_for_instrument(
                                sampling, "decode_chunk_size", name, 8192
                            )), device, amp_enabled, amp_dtype,
                        )
                        grid_prediction = _physical(
                            grid_prediction, standardized, mean, std
                        ).reshape(*lon_grid.shape, ae_models[name].config.n_channels)
                        grid_path = instrument_dir / "regular_global_grid.npz"
                        np.savez_compressed(
                            grid_path, lon_grid=lon_grid, lat_grid=lat_grid,
                            satellite_id=grid_satellite.reshape(lon_grid.shape),
                            is_land=grid_land.reshape(lon_grid.shape),
                            prediction=grid_prediction,
                        )
                        grid_tag, grid_text = _count_description(
                            used, total, len(flat_lon)
                        )
                        _plot(
                            radiance_worker, "grid", grid_path,
                            instrument_dir, name, lead, interval_hours,
                            channels, float(plot_config.get("point_size", 3.0)),
                            resolution_deg=float(generated_config.get(
                                "grid_resolution_deg", 2.0
                            )),
                            color_limits=limits, count_tag=grid_tag,
                            count_text=grid_text, output_time=output_time,
                        )
                    metric_prediction, metric_target, metric_valid = prediction, target, valid
                    extra_metric = {
                        "physical_masked_rmse": _masked_rmse(
                            prediction_plot, target_plot, valid
                        )
                    }

                metrics[f"lead_{lead:03d}"][name] = {
                    "time_ns": int(output_time),
                    "time_utc": _readable_time(output_time),
                    "target_available": True,
                    "query_count": int(len(lon)),
                    "source_count": int(count),
                    "standardized_masked_rmse": _masked_rmse(
                        metric_prediction, metric_target, metric_valid
                    ),
                    **extra_metric,
                }
                print(f"decoded={name} lead={lead} queries={len(lon)}/{count}")

    summary = {
        "config": str(Path(args.config).resolve()),
        "fusion_config": model_section["fusion_config"],
        "fusion_checkpoint": model_section["fusion_checkpoint"],
        "fusion_checkpoint_epoch": fusion_checkpoint.get("epoch"),
        "start_time_ns": int(start_time),
        "start_time_utc": _readable_time(start_time),
        "input_instruments": input_names,
        "input_available": {name: bool(available[name][0]) for name in dataset.names},
        "input_observation_counts": input_counts,
        "output_instruments": output_names,
        "generated_query_satellites": generated_satellites,
        "forecast_steps": steps,
        "interval_hours": interval_hours,
        "metrics": metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()


#  python -m atmosphere.test_five_instrument_forecast --config atmosphere/configs/test_ae_fusion_forecast_five_instruments.yaml
