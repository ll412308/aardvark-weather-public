"""Plot-data preparation and global decoding for BAMUA training."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _run_plot_worker(job):
    """Render with Matplotlib in a process that never imports PyTorch."""
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
            [sys.executable, str(Path(__file__).with_name("plot_worker.py")),
             str(job_path)],
            check=True,
        )
    finally:
        job_path.unlink(missing_ok=True)


def save_loss_plot(history, output_dir, current_epoch, log_scale=True):
    """Save the current loss plot outside the PyTorch training process."""
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


def save_point_reconstruction_plots(lon, lat, truth, prediction, valid,
                                    satellite_id, channels, output_dir, prefix,
                                    max_points=100_000, point_size=3.0,
                                    bt_standardized=False, bt_mean=None,
                                    bt_std=None, color_std_range=3.0):
    """Render one three-row point figure per satellite and BT channel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output_dir, delete=False
    ) as handle:
        data_path = Path(handle.name)
    np.savez(
        data_path,
        lon=_as_numpy(lon).reshape(-1),
        lat=_as_numpy(lat).reshape(-1),
        truth=_as_numpy(truth),
        prediction=_as_numpy(prediction),
        valid=_as_numpy(valid).astype(bool),
        satellite_id=_as_numpy(satellite_id).astype(np.int64).reshape(-1),
    )
    try:
        _run_plot_worker({
            "kind": "points",
            "output_dir": str(output_dir),
            "data_path": str(data_path),
            "prefix": prefix,
            "channels": [int(value) for value in channels],
            "max_points": int(max_points),
            "point_size": float(point_size),
            "bt_standardized": bool(bt_standardized),
            "bt_mean": bt_mean,
            "bt_std": bt_std,
            "color_std_range": float(color_std_range),
        })
    finally:
        data_path.unlink(missing_ok=True)


@torch.no_grad()
def decode_global_grid(model, latent, sample_time, satellite_id, is_land,
                       resolution_deg, chunk_size, device, amp_enabled=False,
                       amp_dtype=torch.float16):
    """Decode one latent state on a regular global longitude/latitude grid."""
    resolution = float(resolution_deg)
    if resolution <= 0:
        raise ValueError("global_resolution_deg must be positive")
    latitude = np.linspace(
        -90.0, 90.0, round(180.0 / resolution) + 1, dtype=np.float32
    )
    longitude = np.linspace(
        -180.0, 180.0, round(360.0 / resolution) + 1,
        dtype=np.float32,
    )[:-1]
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    flat_lon = lon_grid.reshape(-1)
    flat_lat = lat_grid.reshape(-1)
    predictions = []
    model.eval()
    for start in range(0, len(flat_lon), chunk_size):
        end = min(start + chunk_size, len(flat_lon))
        n_query = end - start
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
        ):
            pred = model.decode(
                latent=latent,
                lon=torch.from_numpy(flat_lon[start:end]).to(device).unsqueeze(0),
                lat=torch.from_numpy(flat_lat[start:end]).to(device).unsqueeze(0),
                satellite_id=torch.full(
                    (1, n_query), int(satellite_id), dtype=torch.long,
                    device=device,
                ),
                is_land=torch.full(
                    (1, n_query), bool(is_land), dtype=torch.bool,
                    device=device,
                ),
                sample_time=torch.as_tensor(
                    sample_time, dtype=torch.long, device=device
                ).reshape(1),
            )
        predictions.append(pred[0].float().cpu())
    prediction = torch.cat(predictions).numpy().reshape(
        len(latitude), len(longitude), model.config.n_channels
    )
    return lon_grid, lat_grid, prediction


def save_global_reconstruction_plots(lon_grid, lat_grid, prediction, channels,
                                     output_dir, prefix, satellite_id,
                                     bt_standardized=False, bt_mean=None,
                                     bt_std=None, color_std_range=3.0):
    """Render global grids outside the PyTorch training process."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output_dir, delete=False
    ) as handle:
        data_path = Path(handle.name)
    np.savez(
        data_path,
        lon_grid=_as_numpy(lon_grid),
        lat_grid=_as_numpy(lat_grid),
        prediction=_as_numpy(prediction),
    )
    try:
        _run_plot_worker({
            "kind": "global",
            "output_dir": str(output_dir),
            "data_path": str(data_path),
            "prefix": prefix,
            "satellite_id": int(satellite_id),
            "channels": [int(value) for value in channels],
            "bt_standardized": bool(bt_standardized),
            "bt_mean": bt_mean,
            "bt_std": bt_std,
            "color_std_range": float(color_std_range),
        })
    finally:
        data_path.unlink(missing_ok=True)
