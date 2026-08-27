"""Matplotlib-only worker; intentionally does not import PyTorch."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "axes.linewidth": 1.8,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "lines.linewidth": 2.5,
})


def _finish_axes(axis):
    axis.grid(True, color="0.85", linewidth=0.8, alpha=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.5, length=6)


def _bt_stats(job):
    mean = job.get("bt_mean")
    std = job.get("bt_std")
    if mean is None or std is None:
        return None, None
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _physical_bt(values, channel, job):
    mean, std = _bt_stats(job)
    if job.get("bt_standardized", False):
        if mean is None:
            raise ValueError("Physical BT plotting requires channel_mean/channel_std")
        return values * std[channel] + mean[channel]
    return values


def _shared_color_limits(channel, fallback_values, job):
    mean, std = _bt_stats(job)
    if mean is not None:
        width = float(job.get("color_std_range", 3.0)) * std[channel]
        return float(mean[channel] - width), float(mean[channel] + width)
    return tuple(float(value) for value in np.nanpercentile(
        fallback_values, [1.0, 99.0]
    ))


def _loss(job):
    train_items = job["train_loss"]
    epochs = np.asarray([item["epoch"] for item in train_items], dtype=int)
    train = np.asarray([item["loss"] for item in train_items], dtype=float)
    val_by_epoch = {
        int(item["epoch"]): float(item["loss"])
        for item in job["val_loss"]
    }
    val = np.asarray([val_by_epoch.get(int(epoch), np.nan) for epoch in epochs])
    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.plot(epochs, train, color="#1769aa", marker="o", markersize=6,
              label="Training MSE")
    axis.plot(epochs, val, color="#c62828", marker="s", markersize=7,
              label="Validation MSE")
    axis.set_title("1BAMUA AutoEncoder Loss")
    axis.set_xlabel("Epoch")
    if job.get("log_scale", True):
        axis.set_yscale("log")
        axis.set_ylabel("Masked MSE (log scale)")
    else:
        axis.set_ylabel("Masked MSE")
    axis.set_xticks(epochs)
    axis.legend(frameon=True, framealpha=1.0, edgecolor="0.25")
    _finish_axes(axis)
    path = Path(job["output_path"])
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    for old_path in Path(job["output_dir"]).glob("loss_epoch_*.png"):
        if old_path != path:
            old_path.unlink()


def _points(job):
    data = np.load(job["data_path"])
    lon, lat = data["lon"], data["lat"]
    truth, prediction = data["truth"], data["prediction"]
    valid, satellite_id = data["valid"], data["satellite_id"]
    output_dir = Path(job["output_dir"])
    for sat_id in np.unique(satellite_id):
        satellite_mask = satellite_id == sat_id
        for channel_number in job["channels"]:
            channel = int(channel_number) - 1
            indices = np.flatnonzero(satellite_mask & valid[:, channel])
            max_points = int(job["max_points"])
            if max_points and len(indices) > max_points:
                positions = np.linspace(0, len(indices) - 1, max_points).astype(int)
                indices = indices[positions]
            if len(indices) == 0:
                continue
            x, y = lon[indices], lat[indices]
            observed = _physical_bt(truth[indices, channel], channel, job)
            reconstructed = _physical_bt(
                prediction[indices, channel], channel, job
            )
            difference = reconstructed - observed
            combined = np.concatenate([observed, reconstructed])
            vmin, vmax = _shared_color_limits(channel, combined, job)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(combined))
                vmax = float(np.nanmax(combined) + 1.0e-6)
            error_limit = max(
                float(np.nanpercentile(np.abs(difference), 99.0)), 1.0e-6
            )
            fig, axes = plt.subplots(
                3, 1, figsize=(15, 17), sharex=True, sharey=True,
                constrained_layout=True,
            )
            panels = (
                (observed, "Observed", "viridis", vmin, vmax),
                (reconstructed, "Reconstructed", "viridis", vmin, vmax),
                (difference, "Difference: reconstructed - observed", "coolwarm",
                 -error_limit, error_limit),
            )
            for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
                artist = axis.scatter(
                    x, y, c=values, s=float(job["point_size"]), cmap=cmap,
                    vmin=lower, vmax=upper, linewidths=0, rasterized=True,
                )
                axis.set_title(
                    f"Satellite {int(sat_id)} - Channel {channel_number:02d} - {title}"
                )
                axis.set_ylabel("Latitude (degree)")
                axis.set_xlim(-180.0, 180.0)
                axis.set_ylim(-90.0, 90.0)
                _finish_axes(axis)
                colorbar = fig.colorbar(artist, ax=axis, pad=0.015)
                colorbar.set_label(
                    "BT (physical units)" if title != panels[-1][1]
                    else "BT difference (physical units)"
                )
            axes[-1].set_xlabel("Longitude (degree)")
            path = output_dir / (
                f"{job['prefix']}_satellite_{int(sat_id)}_"
                f"channel_{channel_number:02d}_points.png"
            )
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)


def _global(job):
    data = np.load(job["data_path"])
    lon_grid, lat_grid = data["lon_grid"], data["lat_grid"]
    prediction = data["prediction"]
    sat_id = int(job["satellite_id"])
    output_dir = Path(job["output_dir"])
    for channel_number in job["channels"]:
        channel = int(channel_number) - 1
        values = _physical_bt(prediction[..., channel], channel, job)
        vmin, vmax = _shared_color_limits(channel, values, job)
        fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
        artist = axis.pcolormesh(
            lon_grid, lat_grid, values, shading="auto", cmap="viridis",
            vmin=vmin, vmax=vmax, rasterized=True,
        )
        axis.set_title(
            f"Global Reconstruction - Satellite {sat_id} - "
            f"Channel {channel_number:02d}"
        )
        axis.set_xlabel("Longitude (degree)")
        axis.set_ylabel("Latitude (degree)")
        axis.set_xlim(-180.0, 180.0)
        axis.set_ylim(-90.0, 90.0)
        _finish_axes(axis)
        colorbar = fig.colorbar(artist, ax=axis, pad=0.02)
        colorbar.set_label("BT (physical units)")
        path = output_dir / (
            f"{job['prefix']}_satellite_{sat_id}_"
            f"channel_{channel_number:02d}_global.png"
        )
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)
    {"loss": _loss, "points": _points, "global": _global}[job["kind"]](job)


if __name__ == "__main__":
    main()
