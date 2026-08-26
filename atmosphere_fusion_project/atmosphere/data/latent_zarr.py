from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class InstrumentLatentSpec:
    name: str
    path: str
    latent_dim: int
    height: int
    width: int
    latent_mean: np.ndarray
    latent_std: np.ndarray
    stored_standardized: bool


class MultiInstrumentLatentSequenceDataset(Dataset):
    """Time-aligned lazy reader for per-instrument latent Zarr stores.

    Each store is expected to contain:
      time        [T] int64 Unix nanoseconds
      latent      [T,D,H,W] float32
      density     [T,1,H,W] float32
      available   [T] bool (optional; defaults to True)
      latent_mean [D] float32/float64 (optional)
      latent_std  [D] float32/float64 (optional)
      attrs['latents_standardized'] bool (optional; defaults to False)

    Different instruments may have different D, but H/W must match.
    Missing instruments at a particular time are returned as zero tensors with
    available=False.  Sequences are built on a regular interval, e.g. 6 hours.
    """

    def __init__(
        self,
        stores: Mapping[str, str],
        rollout_steps: int = 1,
        interval_hours: int = 6,
        normalize_latents: bool = True,
    ):
        if not stores:
            raise ValueError("At least one latent store is required")
        self.paths = {str(k): str(v) for k, v in stores.items()}
        self.names = tuple(self.paths.keys())
        self.rollout_steps = int(rollout_steps)
        self.sequence_length = self.rollout_steps + 1
        self.interval_ns = int(interval_hours * 3600 * 1_000_000_000)
        self.normalize_latents = bool(normalize_latents)
        self._roots = None

        import zarr

        self.specs: dict[str, InstrumentLatentSpec] = {}
        self._time_to_index: dict[str, dict[int, int]] = {}
        global_times: set[int] = set()
        hw = None
        for name, path in self.paths.items():
            root = zarr.open_group(path, mode="r")
            if "export_complete" in root.attrs and not bool(
                root.attrs["export_complete"]
            ):
                raise ValueError(
                    f"{name}: latent export is incomplete: {path}"
                )
            time_key = "time" if "time" in root else "time_series"
            times = self._to_int64_time(root[time_key][:])
            latent_shape = tuple(root["latent"].shape)
            if len(latent_shape) != 4:
                raise ValueError(f"{name}: latent must be [T,D,H,W], got {latent_shape}")
            _, dim, height, width = latent_shape
            if hw is None:
                hw = (height, width)
            elif hw != (height, width):
                raise ValueError(
                    f"All instrument latents must share H/W. Expected {hw}, "
                    f"but {name} has {(height, width)}"
                )
            if "latent_mean" in root:
                mean = np.asarray(root["latent_mean"][:], dtype=np.float32)
                std = np.asarray(root["latent_std"][:], dtype=np.float32)
            else:
                mean = np.zeros(dim, dtype=np.float32)
                std = np.ones(dim, dtype=np.float32)
            std = np.maximum(std, 1.0e-6)
            self.specs[name] = InstrumentLatentSpec(
                name=name,
                path=path,
                latent_dim=dim,
                height=height,
                width=width,
                latent_mean=mean,
                latent_std=std,
                stored_standardized=bool(
                    root.attrs.get("latents_standardized", False)
                ),
            )
            mapping = {int(t): i for i, t in enumerate(times.tolist())}
            self._time_to_index[name] = mapping
            global_times.update(mapping.keys())

        self.global_times = np.asarray(sorted(global_times), dtype=np.int64)
        global_set = set(int(t) for t in self.global_times.tolist())
        starts = []
        for t0 in self.global_times.tolist():
            seq = [int(t0) + s * self.interval_ns for s in range(self.sequence_length)]
            if not all(t in global_set for t in seq):
                continue
            if all(not any(t in self._time_to_index[n] for n in self.names) for t in seq):
                continue
            starts.append(int(t0))
        if not starts:
            raise ValueError("No complete regular-time sequences found across latent stores")
        self.start_times = np.asarray(starts, dtype=np.int64)

    @staticmethod
    def _to_int64_time(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[ns]").astype(np.int64)
        return value.astype(np.int64)

    def _open(self):
        if self._roots is None:
            import zarr
            self._roots = {
                name: zarr.open_group(path, mode="r")
                for name, path in self.paths.items()
            }
        return self._roots

    def __len__(self):
        return len(self.start_times)

    def split_chronological(self, val_fraction=0.1, test_fraction=0.1):
        """Return non-overlapping train/val/test sequence indices.

        We split the underlying time axis first, then keep only sequences fully
        contained in one region. This avoids a training rollout target leaking
        across a validation boundary.
        """
        if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1):
            raise ValueError("val_fraction/test_fraction must be in [0,1)")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")
        times = self.global_times
        n = len(times)
        i_train_end = max(1, int(round(n * (1 - val_fraction - test_fraction))))
        i_val_end = max(i_train_end + 1, int(round(n * (1 - test_fraction))))
        i_val_end = min(i_val_end, n - 1) if n > 2 else n
        train_end = int(times[i_train_end - 1])
        val_end = int(times[i_val_end - 1]) if i_val_end > i_train_end else train_end

        train, val, test = [], [], []
        seq_span = self.rollout_steps * self.interval_ns
        for i, t0 in enumerate(self.start_times.tolist()):
            tend = int(t0) + seq_span
            if tend <= train_end:
                train.append(i)
            elif int(t0) > train_end and tend <= val_end:
                val.append(i)
            elif int(t0) > val_end:
                test.append(i)
        return train, val, test

    def __getitem__(self, index):
        roots = self._open()
        t0 = int(self.start_times[index])
        times = [t0 + s * self.interval_ns for s in range(self.sequence_length)]

        latents, densities, available = {}, {}, {}
        for name in self.names:
            spec = self.specs[name]
            root = roots[name]
            latent_steps, density_steps, available_steps = [], [], []
            mean = torch.from_numpy(spec.latent_mean).view(-1, 1, 1)
            std = torch.from_numpy(spec.latent_std).view(-1, 1, 1)
            for t in times:
                source_index = self._time_to_index[name].get(int(t))
                is_available = source_index is not None
                if is_available and "available" in root:
                    is_available = bool(root["available"][source_index])
                if is_available:
                    latent = torch.from_numpy(
                        np.asarray(root["latent"][source_index], dtype=np.float32)
                    )
                    if "density" in root:
                        density = torch.from_numpy(
                            np.asarray(root["density"][source_index], dtype=np.float32)
                        )
                    else:
                        density = torch.ones((1, spec.height, spec.width), dtype=torch.float32)
                    if spec.stored_standardized and not self.normalize_latents:
                        # Caller requested raw scale, but the store contains
                        # standardized values, so invert the stored transform.
                        latent = latent * std + mean
                        print(f"Inverting standardized latent for {name} at time {t}")
                    elif not spec.stored_standardized and self.normalize_latents:
                        latent = (latent - mean) / std
                        print(f"Standardizing latent for {name} at time {t}")
                else:
                    latent = torch.zeros(
                        (spec.latent_dim, spec.height, spec.width), dtype=torch.float32
                    )
                    density = torch.zeros((1, spec.height, spec.width), dtype=torch.float32)
                latent_steps.append(latent)
                density_steps.append(density)
                available_steps.append(is_available)
            latents[name] = torch.stack(latent_steps, dim=0)
            densities[name] = torch.stack(density_steps, dim=0)
            available[name] = torch.tensor(available_steps, dtype=torch.bool)

        return {
            "time": torch.tensor(times, dtype=torch.long),
            "latents": latents,
            "densities": densities,
            "available": available,
        }

    def denormalize(self, name: str, latent: torch.Tensor):
        if not self.normalize_latents:
            return latent
        spec = self.specs[name]
        mean = torch.as_tensor(spec.latent_mean, device=latent.device, dtype=latent.dtype)
        std = torch.as_tensor(spec.latent_std, device=latent.device, dtype=latent.dtype)
        view = [1] * latent.ndim
        channel_axis = latent.ndim - 3
        view[channel_axis] = spec.latent_dim
        return latent * std.view(*view) + mean.view(*view)


def _readable_time(time_ns):
    return np.datetime_as_string(np.datetime64(int(time_ns), "ns"), unit="s")


def _print_test_summary(path):
    """Small one-sequence test for a latent Zarr store."""
    import zarr

    print(f"zarr_path={path}")
    root = zarr.open_group(path, mode="r")
    print("\n[zarr arrays]")
    for name in sorted(root.keys()):
        value = root[name]
        if hasattr(value, "shape"):
            print(f"{name}: shape={value.shape} dtype={value.dtype}")
    print("\n[zarr attributes]")
    for name, value in sorted(dict(root.attrs).items()):
        print(f"{name}={value}")

    dataset = MultiInstrumentLatentSequenceDataset(
        stores={"1bamua": path},
        rollout_steps=1,
        interval_hours=6,
        normalize_latents=True,
    )
    train_idx, val_idx, test_idx = dataset.split_chronological(
        val_fraction=0.1, test_fraction=0.1
    )
    spec = dataset.specs["1bamua"]
    print("\n[dataset]")
    print(f"sequence_count={len(dataset)}")
    print(
        f"split_count train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)}"
    )
    print(
        f"instrument=1bamua latent_dim={spec.latent_dim} "
        f"height={spec.height} width={spec.width}"
    )
    print(f"stored_standardized={spec.stored_standardized}")
    print(
        f"saved_mean_shape={spec.latent_mean.shape} "
        f"saved_std_shape={spec.latent_std.shape}"
    )

    item = dataset[0]
    print("\n[first sequence]")
    print(f"time_shape={tuple(item['time'].shape)} dtype={item['time'].dtype}")
    for step, time_ns in enumerate(item["time"].tolist()):
        print(f"step={step} time_ns={time_ns} time={_readable_time(time_ns)}")
    for name in dataset.names:
        latent = item["latents"][name]
        density = item["densities"][name]
        available = item["available"][name]
        print(f"\n[{name}]")
        print(f"latent: shape={tuple(latent.shape)} dtype={latent.dtype}")
        print(f"density: shape={tuple(density.shape)} dtype={density.dtype}")
        print(
            f"available: shape={tuple(available.shape)} "
            f"values={available.tolist()}"
        )
        print(
            f"latent_first_step mean={latent[0].mean().item():.6f} "
            f"std={latent[0].std(unbiased=False).item():.6f}"
        )
        print(
            f"density_first_step min={density[0].min().item():.6f} "
            f"max={density[0].max().item():.6f} "
            f"mean={density[0].mean().item():.6f}"
        )
        raw_latent = dataset.denormalize(name, latent[0])
        print(
            f"denormalized_first_step: shape={tuple(raw_latent.shape)} "
            f"mean={raw_latent.mean().item():.6f}"
        )


if __name__ == "__main__":
    _print_test_summary(
        r"F:\lyh_data\data_latent\1bamua_latents.zarr"
    )
