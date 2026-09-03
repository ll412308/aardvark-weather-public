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
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "axes.linewidth": 1.8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,
})


def _finish(axis):
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.5, length=5)


def _loss(job):
    history = job["history"]
    metrics = (
        ("total", "Total loss"),
        ("current_latent", "Current reconstruction MSE"),
        ("future_latent", "Future forecast MSE"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    for axis, (key, title) in zip(axes, metrics):
        train_items = history.get("train", [])
        epochs = np.asarray([int(item["epoch"]) for item in train_items])
        train_values = np.asarray([float(item[key]) for item in train_items])
        val_by_epoch = {
            int(item["epoch"]): float(item[key])
            for item in history.get("val", [])
        }
        val_values = np.asarray([
            val_by_epoch.get(int(epoch), np.nan) for epoch in epochs
        ])
        axis.plot(epochs, train_values, color="#1769aa", marker="o",
                  markersize=5, label="Train")
        axis.plot(epochs, val_values, color="#c62828", marker="s",
                  markersize=5, label="Validation")
        if job.get("log_scale", True):
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss" + (" (log scale)" if job.get("log_scale", True) else ""))
        axis.grid(True, color="0.85", linewidth=0.8)
        axis.legend(frameon=True, edgecolor="0.25")
        _finish(axis)
    fig.suptitle("Atmosphere Fusion and Forecast Training", fontsize=20)
    path = Path(job["output_path"])
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    for old_path in Path(job["output_dir"]).glob("loss_epoch_*.png"):
        if old_path != path:
            old_path.unlink()


def _latent(job):
    data = np.load(job["data_path"])
    target = np.asarray(data["target"], dtype=np.float32)
    prediction = np.asarray(data["prediction"], dtype=np.float32)
    difference = prediction - target
    combined = np.concatenate([target.reshape(-1), prediction.reshape(-1)])
    vmin, vmax = np.nanpercentile(combined, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined) + 1e-6)
    error_limit = max(float(np.nanpercentile(np.abs(difference), 99.0)), 1e-6)
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 15), sharex=True, sharey=True,
        constrained_layout=True,
    )
    panels = (
        (target, "Target latent", "viridis", vmin, vmax),
        (prediction, "Predicted latent", "viridis", vmin, vmax),
        (difference, "Difference: prediction - target", "coolwarm",
         -error_limit, error_limit),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
        artist = axis.imshow(
            values, origin="lower", aspect="auto", cmap=cmap,
            vmin=lower, vmax=upper, interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_ylabel("Latitude grid index")
        _finish(axis)
        colorbar = fig.colorbar(artist, ax=axis, pad=0.015)
        colorbar.set_label("Latent value")
    axes[-1].set_xlabel("Longitude grid index")
    fig.suptitle(
        f"Epoch {job['epoch']} | {job['instrument']} | {job['phase']} | "
        f"latent channel {job['channel']}",
        fontsize=19,
    )
    fig.savefig(Path(job["output_path"]), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)
    {"loss": _loss, "latent": _latent}[job["kind"]](job)


if __name__ == "__main__":
    main()
