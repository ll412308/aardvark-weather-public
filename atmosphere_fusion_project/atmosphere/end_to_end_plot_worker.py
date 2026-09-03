"""Matplotlib-only worker for end-to-end AE/fusion/forecast evaluation."""

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
})


def _finish(axis):
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Longitude (degree)")
    axis.set_ylabel("Latitude (degree)")
    axis.grid(True, color="0.82", linewidth=0.7, alpha=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)


def _limits(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        hi = lo + 1.0e-6
    return float(lo), float(hi)


def _plot_limits(job, values):
    if job.get("color_min") is not None and job.get("color_max") is not None:
        return float(job["color_min"]), float(job["color_max"])
    return _limits(values)


def _satellite_name(job, satellite_id):
    return job["satellite_names"].get(str(int(satellite_id)), "UNKNOWN-SATELLITE")


def _safe_name(value):
    return str(value).replace(" ", "-").replace("/", "-")


def comparison(job, data):
    lon, lat = data["lon"], data["lat"]
    target, prediction, valid = data["target"], data["prediction"], data["valid"]
    satellite_id = data["satellite_id"]
    channel = int(job["channel"])
    mask = valid[:, channel] & np.isfinite(target[:, channel])
    for sat_id in np.unique(satellite_id[mask]):
        use = mask & (satellite_id == sat_id)
        if not np.any(use):
            continue
        truth = target[use, channel]
        pred = prediction[use, channel]
        diff = pred - truth
        satellite_name = _satellite_name(job, sat_id)
        vmin, vmax = _plot_limits(job, np.concatenate([truth, pred]))
        error_limit = max(float(np.nanpercentile(np.abs(diff), 99)), 1.0e-6)
        fig, axes = plt.subplots(
            3, 1, figsize=(15, 17), sharex=True, sharey=True,
            constrained_layout=True,
        )
        panels = [
            (truth, "Target", "viridis", vmin, vmax),
            (pred, "Prediction", "viridis", vmin, vmax),
            (diff, "Prediction - target", "coolwarm", -error_limit, error_limit),
        ]
        for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
            artist = axis.scatter(
                lon[use], lat[use], c=values, s=job["point_size"],
                cmap=cmap, vmin=lower, vmax=upper, linewidths=0,
                rasterized=True,
            )
            axis.set_title(
                f"{job['instrument']} {satellite_name} | {job['time_text']} | "
                f"lead={job['lead']} ({job['lead_hours']} h) | "
                f"channel={channel + 1:02d} | {title}\n{job['count_text']}"
            )
            _finish(axis)
            fig.colorbar(artist, ax=axis, pad=0.015).set_label("Physical BT")
        path = Path(job["output_dir"]) / (
            f"comparison_{_safe_name(satellite_name)}_{job['time_tag']}_"
            f"channel_{channel + 1:02d}_"
            f"{job['count_tag']}.png"
        )
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)


def arbitrary(job, data):
    channel = int(job["channel"])
    values = data["prediction"][:, channel]
    satellite_names = sorted({
        _satellite_name(job, value) for value in np.unique(data["satellite_id"])
    })
    satellite_text = ", ".join(satellite_names)
    satellite_tag = _safe_name("multiple" if len(satellite_names) > 1
                               else satellite_names[0])
    vmin, vmax = _plot_limits(job, values)
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    artist = axis.scatter(
        data["lon"], data["lat"], c=values, s=job["point_size"],
        cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0,
        rasterized=True,
    )
    axis.set_title(
        f"Arbitrary global queries | {job['instrument']} {satellite_text} | "
        f"{job['time_text']} | "
        f"lead={job['lead']} ({job['lead_hours']} h) | "
        f"channel={channel + 1:02d}\n{job['count_text']}"
    )
    _finish(axis)
    fig.colorbar(artist, ax=axis, pad=0.015).set_label("Predicted physical BT")
    path = Path(job["output_dir"]) / (
        f"arbitrary_points_{satellite_tag}_{job['time_tag']}_"
        f"channel_{channel + 1:02d}_{job['count_tag']}.png"
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def grid(job, data):
    channel = int(job["channel"])
    values = data["prediction"][..., channel]
    satellite_name = _satellite_name(job, data["satellite_id"].reshape(-1)[0])
    vmin, vmax = _plot_limits(job, values)
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    artist = axis.pcolormesh(
        data["lon_grid"], data["lat_grid"], values,
        shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        rasterized=True,
    )
    axis.set_title(
        f"Regular global grid ({job['resolution_deg']:g} degree) | "
        f"{job['instrument']} {satellite_name} | {job['time_text']} | "
        f"lead={job['lead']} "
        f"({job['lead_hours']} h) | channel={channel + 1:02d}\n"
        f"{job['count_text']}"
    )
    _finish(axis)
    fig.colorbar(artist, ax=axis, pad=0.015).set_label("Predicted physical BT")
    path = Path(job["output_dir"]) / (
        f"grid_{_safe_name(satellite_name)}_{job['time_tag']}_"
        f"channel_{channel + 1:02d}_{job['count_tag']}.png"
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)
    data = np.load(job["data_path"])
    Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    {"comparison": comparison, "arbitrary": arbitrary, "grid": grid}[
        job["kind"]
    ](job, data)


if __name__ == "__main__":
    main()
