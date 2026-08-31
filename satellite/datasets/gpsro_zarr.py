"""Context/target sampling from six-hour GPSRO Zarr bins."""

import numpy as np
import torch
from torch.utils.data import Dataset


class GPSROZarrDataset(Dataset):
    def __init__(self, path, n_context=65_536, n_target=16_384,
                 target_overlap=0.5, seed=0, return_indices=False):
        self.path = str(path)
        self.n_context = int(n_context)
        self.n_target = int(n_target)
        self.target_overlap = float(target_overlap)
        if not 0.0 <= self.target_overlap <= 1.0:
            raise ValueError("target_overlap must be in [0, 1]")
        self.seed = int(seed)
        self.return_indices = bool(return_indices)
        self.epoch = 0
        self._root = None
        root = self._open()
        counts = np.asarray(root["sample_count"][:], dtype=np.int64)
        # Empty bins cannot form a reconstruction sample. Keeping this mapping
        # also prevents DataLoader workers from failing halfway through an epoch.
        self.sample_indices = np.flatnonzero(counts >= 2).astype(np.int64)
        if len(self.sample_indices) == 0:
            raise ValueError("GPSRO Zarr contains no 6-hour bin with >=2 observations")

    def _open(self):
        if self._root is None:
            import zarr
            self._root = zarr.open_group(self.path, mode="r")
        return self._root

    def __len__(self):
        return len(self.sample_indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _time_int64(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[ns]").astype(np.int64)
        return value.astype(np.int64)

    @staticmethod
    def _read(root, name, indices):
        array = root[name]
        if isinstance(indices, slice):
            return np.asarray(array[indices])
        try:
            return np.asarray(array.oindex[indices])
        except AttributeError:
            return np.asarray(array.get_orthogonal_selection((indices,)))

    @staticmethod
    def _is_land(longitude, latitude):
        """Return a vectorized land mask for sampled GPSRO coordinates."""
        try:
            from global_land_mask import globe
        except ImportError as exc:
            raise ImportError(
                "GPSRO is_land generation requires 'global-land-mask'. "
                "Install it with: pip install global-land-mask"
            ) from exc

        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)
        result = np.zeros(longitude.shape, dtype=bool)
        valid = np.isfinite(longitude) & np.isfinite(latitude)
        valid &= (latitude >= -90.0) & (latitude <= 90.0)
        if np.any(valid):
            # The library expects longitude in [-180, 180]. GPSRO files may
            # use either that convention or [0, 360), so normalize it here.
            wrapped_lon = (
                np.remainder(longitude[valid] + 180.0, 360.0) - 180.0
            )
            result[valid] = globe.is_land(latitude[valid], wrapped_lon)
        return result

    def sample_time(self, dataset_index):
        root = self._open()
        source_index = int(self.sample_indices[dataset_index])
        return int(self._time_int64(root["time_series"][source_index]))

    def _points(self, root, indices, sample_time, include_value):
        longitude = self._read(root, "longitude", indices)
        latitude = self._read(root, "latitude", indices)
        points = {
            "lon": torch.from_numpy(longitude).float(),
            "lat": torch.from_numpy(latitude).float(),
            "height": torch.from_numpy(self._read(root, "height_m", indices)).float(),
            "satellite_id": torch.from_numpy(
                self._read(root, "satellite_id", indices)
            ).long(),
            "is_land": torch.from_numpy(
                self._is_land(longitude, latitude)
            ).bool(),
            "obs_time": torch.from_numpy(
                self._time_int64(self._read(root, "time", indices))
            ).long(),
            "sample_time": torch.tensor(sample_time, dtype=torch.long),
        }
        if include_value:
            value = torch.from_numpy(
                self._read(root, "refractivity", indices)
            ).float().unsqueeze(-1)
            points["refractivity"] = value
            points["valid"] = torch.isfinite(value)
        return points

    def get_full_sample(self, dataset_index):
        root = self._open()
        source_index = int(self.sample_indices[dataset_index])
        start = int(root["sample_start"][source_index])
        count = int(root["sample_count"][source_index])
        sample_time = int(self._time_int64(root["time_series"][source_index]))
        points = self._points(
            root, slice(start, start + count), sample_time, include_value=True
        )
        return {
            "observations": points,
            "sample_time": torch.tensor(sample_time, dtype=torch.long),
            "count": count,
            "sample_index": source_index,
        }

    def __getitem__(self, dataset_index):
        root = self._open()
        source_index = int(self.sample_indices[dataset_index])
        start = int(root["sample_start"][source_index])
        count = int(root["sample_count"][source_index])
        rng = np.random.default_rng(
            self.seed + int(dataset_index) + self.epoch * len(self)
        )
        max_context = count if self.target_overlap == 1.0 else count - 1
        n_context = min(self.n_context, max_context)
        n_target = min(self.n_target, count)
        permutation = rng.permutation(count).astype(np.int64) + start
        context_indices = permutation[:n_context]
        heldout_indices = permutation[n_context:]
        overlap_count = min(round(n_target * self.target_overlap), n_context)
        heldout_count = min(n_target - overlap_count, len(heldout_indices))
        overlap_count = min(
            n_context,
            overlap_count + (n_target - overlap_count - heldout_count),
        )
        parts = []
        if overlap_count:
            parts.append(rng.choice(
                context_indices, size=overlap_count, replace=False
            ))
        if heldout_count:
            parts.append(heldout_indices[:heldout_count])
        if not parts:
            raise ValueError(f"GPSRO bin {source_index} produced no target points")
        target_indices = np.concatenate(parts)
        rng.shuffle(target_indices)
        sample_time = int(self._time_int64(root["time_series"][source_index]))

        context = self._points(
            root, context_indices, sample_time, include_value=True
        )
        target_all = self._points(
            root, target_indices, sample_time, include_value=True
        )
        target_value = target_all.pop("refractivity")
        target_valid = target_all.pop("valid")
        item = {
            "context": context,
            "target": target_all,
            "target_refractivity": target_value,
            "target_valid": target_valid,
            "sample_time": torch.tensor(sample_time, dtype=torch.long),
            "source_sample_index": torch.tensor(source_index, dtype=torch.long),
        }
        if self.return_indices:
            item["context_idx"] = torch.from_numpy(context_indices).long()
            item["target_idx"] = torch.from_numpy(target_indices).long()
        return item


if __name__ == "__main__":
    path = r"F:\lyh_data\gps_zarr_no_provider\gpsro.zarr"
    dataset = GPSROZarrDataset(
        path, n_context=4096, n_target=1024,
        target_overlap=0.5, return_indices=True,
    )
    item = dataset[0]
    overlap = len(
        set(item["context_idx"].tolist()) & set(item["target_idx"].tolist())
    )
    print(f"usable_bins={len(dataset)}")
    print(f"source_sample_index={item['source_sample_index'].item()}")
    print(f"sample_time={np.datetime64(item['sample_time'].item(), 'ns')}")
    print(f"context={len(item['context_idx'])} target={len(item['target_idx'])}")
    print(f"target_overlap={overlap / max(len(item['target_idx']), 1):.4f}")
    for group_name in ("context", "target"):
        print(f"[{group_name}]")
        for name, value in item[group_name].items():
            print(f"  {name}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"target_refractivity={tuple(item['target_refractivity'].shape)}")
    print(f"target_valid_fraction={item['target_valid'].float().mean().item():.6f}")
    print(f"context_land_fraction={item['context']['is_land'].float().mean().item():.6f}")
    print(f"target_land_fraction={item['target']['is_land'].float().mean().item():.6f}")
