"""Plot-data preparation and global 3-D decoding for GPSRO training."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from satellite.datasets import GPSROZarrDataset


def _as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _run_plot_worker(job):
    """Render in a process that never imports PyTorch or the land-mask array."""
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=output_dir,
        encoding="utf-8", delete=False,
    ) as handle:
        job_path = Path(handle.name)
        json.dump(job, handle)
    try:
        subprocess.run(
            [sys.executable,
             str(Path(__file__).with_name("gpsro_plot_worker.py")),
             str(job_path)],
            check=True,
        )
    finally:
        job_path.unlink(missing_ok=True)


def load_refractivity_stats(zarr_path):
    """Load the height-dependent inverse transform for GPSRO refractivity."""
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    required = (
        "refractivity_log_mean", "refractivity_log_std",
        "height_bin_lower_m", "height_bin_upper_m",
    )
    missing = [name for name in required if name not in root]
    if missing:
        raise ValueError(f"GPSRO plotting statistics are missing: {missing}")
    return {
        "mean": np.asarray(root["refractivity_log_mean"][:], dtype=np.float64),
        "std": np.maximum(
            np.asarray(root["refractivity_log_std"][:], dtype=np.float64),
            1.0e-8,
        ),
        "lower": np.asarray(root["height_bin_lower_m"][:], dtype=np.float64),
        "upper": np.asarray(root["height_bin_upper_m"][:], dtype=np.float64),
    }


def transform_refractivity(values, height_m, stats, value_space="log"):
    """Convert standardized values to standardized, log(N), or physical N."""
    values = _as_numpy(values).reshape(-1).astype(np.float64)
    height_m = _as_numpy(height_m).reshape(-1).astype(np.float64)
    value_space = str(value_space).lower()
    if value_space == "standardized":
        return values.astype(np.float32), "Standardized log refractivity"
    indices = np.searchsorted(stats["lower"], height_m, side="right") - 1
    indices = np.clip(indices, 0, len(stats["mean"]) - 1)
    log_refractivity = values * stats["std"][indices] + stats["mean"][indices]
    if value_space == "log":
        return log_refractivity.astype(np.float32), "Log refractivity (N-units)"
    if value_space == "physical":
        return np.exp(log_refractivity).astype(np.float32), "Refractivity (N-units)"
    raise ValueError("plot.value_space must be 'standardized', 'log', or 'physical'")


def save_loss_plot(history, output_dir, current_epoch, log_scale=True):
    if not history.get("train_loss"):
        return None
    output_dir = Path(output_dir)
    path = output_dir / f"loss_epoch_{int(current_epoch):04d}.png"
    _run_plot_worker({
        "kind": "loss",
        "output_dir": str(output_dir),
        "output_path": str(path),
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "log_scale": bool(log_scale),
    })
    return path


def save_reconstruction_3d(lon, lat, height, truth, prediction, valid,
                           output_dir, prefix, stats, value_space="log",
                           max_points=60_000, point_size=4.0, seed=0,
                           title_prefix="GPSRO reconstruction"):
    """Save side-by-side target/reconstruction/difference 3-D scatter panels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height_np = _as_numpy(height).reshape(-1)
    truth_plot, label = transform_refractivity(
        truth, height_np, stats, value_space
    )
    prediction_plot, _ = transform_refractivity(
        prediction, height_np, stats, value_space
    )
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output_dir, delete=False
    ) as handle:
        data_path = Path(handle.name)
    np.savez(
        data_path,
        lon=_as_numpy(lon).reshape(-1),
        lat=_as_numpy(lat).reshape(-1),
        height=height_np,
        truth=truth_plot,
        prediction=prediction_plot,
        valid=_as_numpy(valid).astype(bool).reshape(-1),
    )
    try:
        _run_plot_worker({
            "kind": "reconstruction_3d",
            "output_dir": str(output_dir),
            "output_path": str(output_dir / f"{prefix}_3d_points.png"),
            "data_path": str(data_path),
            "max_points": int(max_points),
            "point_size": float(point_size),
            "seed": int(seed),
            "value_label": label,
            "title_prefix": str(title_prefix),
        })
    finally:
        data_path.unlink(missing_ok=True)


@torch.no_grad()
def decode_global_3d(model, latent, sample_time, satellite_id,
                     horizontal_resolution_deg, heights_m, chunk_size,
                     device, amp_enabled=False, amp_dtype=torch.float16):
    """Decode a common GPSRO latent on a regular lon/lat/height query grid."""
    resolution = float(horizontal_resolution_deg)
    longitude = np.arange(-180.0, 180.0, resolution, dtype=np.float32)
    latitude = np.arange(-90.0, 90.0, resolution, dtype=np.float32)
    # Always include the North Pole without letting awkward resolutions (for
    # example 7 degrees) create an invalid latitude greater than 90 degrees.
    if not len(latitude) or latitude[-1] < 90.0:
        latitude = np.concatenate([latitude, np.asarray([90.0], np.float32)])
    heights = np.asarray(heights_m, dtype=np.float32).reshape(-1)
    # GPSRO standardisation statistics use half-open height bins. In
    # particular, vertical_max_m is an upper boundary rather than a trained
    # query level. Drop out-of-range configured levels before decoding so an
    # unsupported top layer cannot dominate the log(N) colour scale.
    valid_heights = np.isfinite(heights)
    valid_heights &= heights >= float(model.config.vertical_min_m)
    valid_heights &= heights < float(model.config.vertical_max_m)
    heights = heights[valid_heights]
    if not len(heights):
        raise ValueError(
            "GPSRO global heights must contain at least one finite value in "
            "[vertical_min_m, vertical_max_m)"
        )
    height_grid, lat_grid, lon_grid = np.meshgrid(
        heights, latitude, longitude, indexing="ij"
    )
    flat_lon = lon_grid.reshape(-1)
    flat_lat = lat_grid.reshape(-1)
    flat_height = height_grid.reshape(-1)
    is_land = GPSROZarrDataset._is_land(flat_lon, flat_lat)
    n_query = len(flat_lon)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        prediction = model.decode(
            latent=latent,
            lon=torch.from_numpy(flat_lon).to(device).unsqueeze(0),
            lat=torch.from_numpy(flat_lat).to(device).unsqueeze(0),
            height=torch.from_numpy(flat_height).to(device).unsqueeze(0),
            satellite_id=torch.full(
                (1, n_query), int(satellite_id), dtype=torch.long, device=device
            ),
            is_land=torch.from_numpy(is_land).to(device).unsqueeze(0),
            sample_time=torch.as_tensor(
                sample_time, dtype=torch.long, device=device
            ).reshape(1),
            chunk_size=int(chunk_size),
        )
    return flat_lon, flat_lat, flat_height, prediction[0].float().cpu().numpy()


def save_global_3d(lon, lat, height, prediction, output_dir, prefix, stats,
                   value_space="log", max_points=100_000, point_size=3.0,
                   seed=0, title="GPSRO global 3-D reconstruction"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values, label = transform_refractivity(
        prediction, height, stats, value_space
    )
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output_dir, delete=False
    ) as handle:
        data_path = Path(handle.name)
    np.savez(
        data_path, lon=_as_numpy(lon).reshape(-1),
        lat=_as_numpy(lat).reshape(-1),
        height=_as_numpy(height).reshape(-1), prediction=values,
    )
    try:
        _run_plot_worker({
            "kind": "global_3d",
            "output_dir": str(output_dir),
            "output_path": str(output_dir / f"{prefix}_global_3d.png"),
            "data_path": str(data_path),
            "max_points": int(max_points),
            "point_size": float(point_size),
            "seed": int(seed),
            "value_label": label,
            "title": str(title),
        })
    finally:
        data_path.unlink(missing_ok=True)
