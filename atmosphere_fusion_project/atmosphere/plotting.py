"""Prepare fusion-training plots and render them outside the PyTorch process."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _run_worker(job):
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


def save_loss_plot(history, output_dir, epoch, log_scale=True):
    """Plot total/current/future train and validation losses."""
    if not history.get("train"):
        return None
    output_dir = Path(output_dir)
    path = output_dir / f"loss_epoch_{int(epoch):04d}.png"
    _run_worker({
        "kind": "loss",
        "output_dir": str(output_dir),
        "output_path": str(path),
        "history": history,
        "log_scale": bool(log_scale),
    })
    return path


def save_latent_comparison(target, prediction, output_dir, epoch,
                           instrument, phase, channel):
    """Plot one latent channel as target, prediction and prediction-target."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"epoch_{int(epoch):04d}_{instrument}_{phase}_"
        f"channel_{int(channel) + 1:03d}.png"
    )
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=output_dir, delete=False
    ) as handle:
        data_path = Path(handle.name)
    np.savez(
        data_path,
        target=torch.as_tensor(target).detach().float().cpu().numpy(),
        prediction=torch.as_tensor(prediction).detach().float().cpu().numpy(),
    )
    try:
        _run_worker({
            "kind": "latent",
            "output_dir": str(output_dir),
            "output_path": str(path),
            "data_path": str(data_path),
            "epoch": int(epoch),
            "instrument": str(instrument),
            "phase": str(phase),
            "channel": int(channel) + 1,
        })
    finally:
        data_path.unlink(missing_ok=True)
    return path

