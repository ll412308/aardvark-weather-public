"""Matplotlib-only GPSRO plot worker; intentionally never imports PyTorch."""

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
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "axes.linewidth": 1.6,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
    "lines.linewidth": 2.4,
})


def _sample(indices, maximum, seed):
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def _limits(values, percentiles=(1.0, 99.0)):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    lower, upper = np.percentile(values, percentiles)
    if lower == upper:
        upper = lower + 1.0e-6
    return float(lower), float(upper)


def _style_3d(axis):
    axis.set_xlabel("Longitude (degree)", labelpad=10)
    axis.set_ylabel("Latitude (degree)", labelpad=10)
    axis.set_zlabel("Height (km)", labelpad=8)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.view_init(elev=24, azim=-58)
    axis.set_proj_type("ortho")
    axis.set_box_aspect((2.0, 1.0, 0.9))
    axis.grid(True, color="0.85", linewidth=0.6)


def _loss(job):
    items = job["train_loss"]
    epochs = np.asarray([item["epoch"] for item in items], dtype=int)
    train = np.asarray([item["loss"] for item in items], dtype=float)
    val_map = {int(item["epoch"]): float(item["loss"])
               for item in job["val_loss"]}
    validation = np.asarray(
        [val_map.get(int(epoch), np.nan) for epoch in epochs], dtype=float
    )
    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.plot(epochs, train, color="#1769aa", marker="o", markersize=5,
              label="Training MSE")
    axis.plot(epochs, validation, color="#c4512d", marker="s", markersize=6,
              label="Validation MSE")
    axis.set_title("GPSRO AutoEncoder Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked MSE" + (" (log scale)" if job["log_scale"] else ""))
    if job["log_scale"]:
        axis.set_yscale("log")
    axis.grid(True, color="0.85", linewidth=0.8)
    axis.legend(frameon=True, edgecolor="0.25")
    for spine in axis.spines.values():
        spine.set_linewidth(1.6)
    path = Path(job["output_path"])
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    for old in Path(job["output_dir"]).glob("loss_epoch_*.png"):
        if old != path:
            old.unlink()


def _reconstruction_3d(job):
    data = np.load(job["data_path"])
    valid = data["valid"] & np.isfinite(data["truth"]) & np.isfinite(data["prediction"])
    indices = _sample(
        np.flatnonzero(valid), int(job["max_points"]), int(job["seed"])
    )
    if not len(indices):
        return
    lon, lat = data["lon"][indices], data["lat"][indices]
    height = data["height"][indices] / 1000.0
    truth = data["truth"][indices]
    prediction = data["prediction"][indices]
    difference = prediction - truth
    value_min, value_max = _limits(np.concatenate([truth, prediction]))
    error_limit = max(_limits(np.abs(difference), (0.0, 99.0))[1], 1.0e-6)
    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    panels = (
        (truth, "Target", "viridis", value_min, value_max),
        (prediction, "Reconstruction", "viridis", value_min, value_max),
        (difference, "Difference: reconstruction - target", "coolwarm",
         -error_limit, error_limit),
    )
    for position, (values, title, cmap, vmin, vmax) in enumerate(panels, 1):
        axis = fig.add_subplot(1, 3, position, projection="3d")
        artist = axis.scatter(
            lon, lat, height, c=values, s=float(job["point_size"]),
            cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
            alpha=0.78, rasterized=True,
        )
        axis.set_title(title)
        _style_3d(axis)
        colorbar = fig.colorbar(artist, ax=axis, shrink=0.68, pad=0.04)
        colorbar.set_label(job["value_label"] + (" difference" if position == 3 else ""))
    fig.suptitle(
        f"{job['title_prefix']} | plotted points={len(indices):,}", fontsize=20
    )
    fig.savefig(job["output_path"], dpi=180, facecolor="white")
    plt.close(fig)


def _global_3d(job):
    data = np.load(job["data_path"])
    valid = np.isfinite(data["prediction"])
    indices = _sample(
        np.flatnonzero(valid), int(job["max_points"]), int(job["seed"])
    )
    if not len(indices):
        return
    values = data["prediction"][indices]
    vmin, vmax = _limits(values)
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    artist = axis.scatter(
        data["lon"][indices], data["lat"][indices],
        data["height"][indices] / 1000.0,
        c=values, s=float(job["point_size"]), cmap="viridis",
        vmin=vmin, vmax=vmax, linewidths=0, alpha=0.72, rasterized=True,
    )
    _style_3d(axis)
    axis.set_title(f"{job['title']} | plotted grid points={len(indices):,}")
    colorbar = fig.colorbar(artist, ax=axis, shrink=0.72, pad=0.04)
    colorbar.set_label(job["value_label"])
    fig.savefig(job["output_path"], dpi=180, facecolor="white")
    plt.close(fig)


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)
    functions = {
        "loss": _loss,
        "reconstruction_3d": _reconstruction_3d,
        "global_3d": _global_3d,
    }
    functions[job["kind"]](job)


if __name__ == "__main__":
    main()
