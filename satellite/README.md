# 1BAMUA local SetConv autoencoder

This package is deliberately independent of Aardvark's ViT, forecast processor,
U-Net, and multi-instrument paths.

Aardvark's `convDeepSet` constructs separate longitude and latitude kernel
matrices and combines them with dense `einsum` operations. Its `OffToOn` line is
`...cw,...wx,...wy->...cxy`. That representation is useful for the layouts used
by Aardvark, but is not a scalable general operator for hundreds of thousands of
paired `(lon_i, lat_i)` observations.

Here each observation is kept paired. It contributes to a configurable local
neighbourhood (`3 x 3` by default) using a learnable Gaussian, followed by
`scatter_add`. Longitude indices wrap periodically and latitude indices are
clipped at the poles. Both directions therefore use `O(B N K)` work and memory,
where `K` is the small fixed number of neighbours.

Run from the repository root:

```powershell
python -m satellite.test_bamua
python -m satellite.test_bamua --zarr "F:\lyh_data\data_zarr_no_provider_filter\1bamua.zarr" --context 100 --target 20 --target-overlap 0.5
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml
```

`--target-overlap` controls how many target points are also present in the
context set. `0.0` gives fully held-out targets, `1.0` gives ordinary
autoencoder-style targets sampled from context, and `0.5` mixes both signals.

Resume the newest saved checkpoint for the same `run_name`:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --resume
```

Or resume a specific checkpoint:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --resume "runs/bamua_smoke/20260825_120000/latest.pth"
```

Each run directory contains `loss_log.jsonl`, `best.pth`, periodic
`epoch_XXXX.pth`/`latest.pth`, and `reconstructions/epoch_XXXX.pth`. A
reconstruction file contains the first validation batch's prediction, target,
validity mask, longitude, latitude, and sample time.

The YAML also controls reproducibility, automatic mixed precision, and the
learning-rate schedule:

```yaml
train:
  seed: 42
  deterministic: false
  mixed_precision: true
  amp_dtype: "float16"

scheduler:
  name: "warmup_cosine"
  warmup_epochs: 2
  min_lr: 1.0e-6
```

`warmup_cosine` is updated after every optimizer step. It linearly raises the
learning rate during warmup and then follows a cosine curve down to `min_lr`.
Mixed precision is enabled only when CUDA is available; CPU training falls back
to float32 automatically. Model, optimizer, scheduler, AMP scaler, loss history,
early-stopping state, and random-number-generator states are all checkpointed.

## Three-dimensional SetConv

`satellite.models` also exports `SetConv3DOffToOn` and `SetConv3DOnToOff`.
They map paired `(longitude, latitude, vertical)` points to/from a local
`[B, D, Z, H, W]` grid. The vertical input can be geometric altitude/elevation,
geopotential, geopotential height, or pressure:

```python
from satellite.models import SetConv3DOffToOn

setconv = SetConv3DOffToOn(
    grid_resolution_deg=5.0,
    vertical_min_m=0.0,
    vertical_max_m=20_000.0,
    vertical_resolution_m=1_000.0,
)
grid, density = setconv(
    features, longitude, latitude, pressure,
    vertical_type="pressure", vertical_unit="hPa",
)
```

All vertical metadata are converted to geopotential height in metres. Pressure
uses a configurable log-pressure scale-height approximation, because pressure
cannot be converted uniquely to geometric height without an atmospheric
temperature/humidity profile. The existing `BAMUAAutoEncoder` remains 2-D;
using the 3-D grid end-to-end also requires a 3-D latent processor.

Complete 6-hour bins are sorted by Zarr `time_series`: early bins are used for
training, the following bins for validation, and the latest bins for testing.
The final test reconstructs every observation in a selected bin three times,
using 100%, 90%, and 80% of its observations as encoder context. Context
encoding and all-point decoding are chunked to control GPU memory. Run only the
full reconstruction test from the newest checkpoint with:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --test-only
```

Results are written under `test_reconstructions/sample_XXXX/`. `metrics.json`
reports RMSE over every query, context points, held-out points, and each BT
channel. `target_all.pth` is stored once, while each context percentage has its
own prediction file.

## GPSRO three-dimensional autoencoder

The GPSRO autoencoder reconstructs the standardized log-refractivity already
stored in `gpsro.zarr`. Because that Zarr does not store `is_land`, the Dataset
generates it from longitude/latitude with `global_land_mask` for every sampled
context and target point. Paired `(longitude, latitude, height_m)` points are
encoded to a local 3-D latent grid and decoded at arbitrary 3-D query points.
The library expands a high-resolution global mask in memory, so the supplied
Windows training YAML deliberately keeps `num_workers: 0`.
The default 5-degree/5-km configuration changes shapes as follows:

```text
refractivity/context coordinates [B,Nc,1] / [B,Nc]
  -> point features [B,Nc,64]
  -> SetConv3D OffToOn [B,64,14,37,72]
  -> fold feature/height [B,896,37,72]
  -> shared 2-D latent processor [B,64,37,72]
  -> decoder restore [B,64,14,37,72]
  -> SetConv3D OnToOff [B,Nt,64]
  -> predicted refractivity [B,Nt,1]
```

The `[B,64,37,72]` processor output is the GPSRO representation intended for
later fusion/forecasting. Re-expansion to a 3-D grid belongs to the GPSRO
decoder only.

Run the lightweight synthetic model test, the real-Zarr 100/20-point test,
and training from the repository root:

```powershell
python -m satellite.models.gpsro_autoencoder
python -m satellite.test_gpsro --zarr "F:/lyh_data/gps_zarr_no_provider/gpsro.zarr"
python -m satellite.train_gpsro --config satellite/configs/gpsro_train.yaml
```

Resume from the most recent saved state with:

```powershell
python -m satellite.train_gpsro --config satellite/configs/gpsro_train.yaml --resume "runs/gpsro_autoencoder/<run-time>/latest.pth"
```

GPSRO runs also save a log-scale loss curve, one three-panel 3-D validation
scatter (`target`, `reconstruction`, and `difference`), and two test figures:
a random real-observation three-panel scatter and a decoded regular global
longitude/latitude/height grid. Plots are rendered in a separate process so
Matplotlib and PyTorch do not load conflicting OpenMP runtimes. To draw only
the test figures from an existing checkpoint:

```powershell
python -m satellite.train_gpsro --config satellite/configs/gpsro_train.yaml --resume "runs/gpsro_autoencoder/<run-time>/best.pth" --test-only
```

Export every complete GPSRO six-hour bin to a fusion-ready latent Zarr with:

```powershell
python -m satellite.export_gpsro_latents --config satellite/configs/gpsro_train.yaml --checkpoint "runs/gpsro_autoencoder/<run-time>/best.pth" --output-zarr "F:/lyh_data/data_latent/gpsro_latents.zarr" --calculate-stats --standardize-latents --overwrite
```

The primary arrays are `latent [T,D,H,W]` and vertically summed
`density [T,1,H,W]`. Optional `density_3d [T,1,Z,H,W]` preserves diagnostic
vertical coverage. If another instrument uses a different H/W, pass for example
`--output-resolution-deg 2.0`; resampling is always explicit and recorded in
the Zarr attributes.
