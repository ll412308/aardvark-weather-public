"""Random context/target sampling from one 1BAMUA six-hour Zarr bin."""

import numpy as np
import torch
from torch.utils.data import Dataset


class BAMUAZarrDataset(Dataset):
    def __init__(self, path, n_context=65_536, n_target=16_384,
                 target_overlap=0.5, seed=0, return_indices=False):
        self.path = str(path)
        self.n_context = int(n_context)
        self.n_target = int(n_target)
        self.target_overlap = float(target_overlap)  # 如果为0，target与context(输入)完全不重叠，为1，target与context完全重叠
        if not 0.0 <= self.target_overlap <= 1.0:
            raise ValueError("target_overlap must be in [0, 1]")
        self.seed = int(seed)
        self.return_indices = bool(return_indices)
        self._root = None  # Open separately in each DataLoader worker.
        root = self._open()
        self._length = len(root["time_series"])
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _open(self):
        if self._root is None:
            import zarr
            self._root = zarr.open_group(self.path, mode="r")
        return self._root

    def __len__(self):
        return self._length

    @staticmethod
    def _int64_time(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.datetime64):
            return value.astype("datetime64[ns]").astype(np.int64)
        return value.astype(np.int64)

    @staticmethod
    def _read(root, name, indices):
        """Read only the requested points; indices may be an array or a slice."""
        array = root[name]
        if isinstance(indices, slice):
            return np.asarray(array[indices])
        try:
            return np.asarray(array.oindex[indices])
        except AttributeError:
            return np.asarray(array.get_orthogonal_selection((indices,)))

    def _common(self, root, indices, sample_time):
        read = lambda name: self._read(root, name, indices)
        return {
            "lon": torch.from_numpy(read("longitude")).float(),
            "lat": torch.from_numpy(read("latitude")).float(),
            "satellite_id": torch.from_numpy(read("satellite_id")).long(),
            "is_land": torch.from_numpy(read("is_land")).bool(),
            "obs_time": torch.from_numpy(self._int64_time(read("time"))).long(),
            "sample_time": torch.tensor(sample_time, dtype=torch.long),
            "zenith": torch.from_numpy(read("satellite_zenith_angle")).float(),
            "azimuth": torch.from_numpy(read("satellite_azimuth")).float(),
        }

    def get_full_sample(self, idx):
        """Read every observation from one complete 6h bin for final testing."""
        root = self._open()
        start = int(root["sample_start"][idx])
        count = int(root["sample_count"][idx])
        sample_time = self._int64_time(root["time_series"][idx]).item()
        indices = slice(start, start + count)
        observations = self._common(root, indices, sample_time)
        observations["bt"] = torch.from_numpy(
            self._read(root, "brightness_temperature", indices)
        ).float()
        observations["valid"] = torch.from_numpy(
            self._read(root, "brightness_temperature_valid", indices)
        ).bool()
        return {
            "observations": observations,
            "sample_time": torch.tensor(sample_time, dtype=torch.long),
            "count": count,
            "sample_index": int(idx),
        }

    def __getitem__(self, idx):
        root = self._open()
        start = int(root["sample_start"][idx])
        count = int(root["sample_count"][idx])
        if count < 2:
            raise ValueError(f"6h bin {idx} has only {count} observation(s)")
        rng = np.random.default_rng(self.seed + idx + self.epoch * self._length)
        # Keep one held-out point available unless the user asks for fully overlapping targets.
        max_context = count if self.target_overlap == 1.0 else count - 1
        n_context = min(self.n_context, max_context)
        n_target = min(self.n_target, count)
        permutation = rng.permutation(count) + start
        context_idx = permutation[:n_context]
        heldout_idx = permutation[n_context:]
        n_overlap = min(round(n_target * self.target_overlap), n_context)
        n_heldout = min(n_target - n_overlap, len(heldout_idx))
        # If the bin is too small for the requested held-out fraction, keep target size up
        # by drawing more targets from context.
        n_overlap = min(n_context, n_overlap + (n_target - n_overlap - n_heldout))
        target_parts = []
        if n_overlap:
            target_parts.append(rng.choice(context_idx, size=n_overlap, replace=False))
        if n_heldout:
            target_parts.append(heldout_idx[:n_heldout])
        if not target_parts:
            raise ValueError("No target observations were sampled; check n_target and n_context")
        target_idx = np.concatenate(target_parts)
        rng.shuffle(target_idx)
        sample_time = self._int64_time(root["time_series"][idx]).item()

        context = self._common(root, context_idx, sample_time)
        context["bt"] = torch.from_numpy(
            self._read(root, "brightness_temperature", context_idx)
        ).float()
        context["valid"] = torch.from_numpy(
            self._read(root, "brightness_temperature_valid", context_idx)
        ).bool()
        target = self._common(root, target_idx, sample_time)
        target_bt = torch.from_numpy(
            self._read(root, "brightness_temperature", target_idx)
        ).float()
        target_valid = torch.from_numpy(
            self._read(root, "brightness_temperature_valid", target_idx)
        ).bool()
        item = {"context": context, "target": target,
                "target_bt": target_bt, "target_valid": target_valid,
                "sample_time": torch.tensor(sample_time, dtype=torch.long)}
        if self.return_indices:
            item["context_idx"] = torch.from_numpy(context_idx).long()
            item["target_idx"] = torch.from_numpy(target_idx).long()
        return item


def _shape_summary(name, value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            print(f"{name}: shape=() dtype={value.dtype} value={value.item()}")
        else:
            print(f"{name}: shape={tuple(value.shape)} dtype={value.dtype}")
    else:
        print(f"{name}: {value}")


def _format_ns_time(ns):
    return np.datetime_as_string(np.datetime64(int(ns), "ns"), unit="s")


def _print_sample_counts(dataset, max_rows=20):
    root = dataset._open()
    counts = np.asarray(root["sample_count"][:], dtype=np.int64)
    starts = np.asarray(root["sample_start"][:], dtype=np.int64)
    times = dataset._int64_time(root["time_series"][:])
    print("[sample counts]")
    print(f"num_samples={len(counts)}")
    print(f"min_count={counts.min()} max_count={counts.max()} mean_count={counts.mean():.1f}")
    print(f"showing_first={min(max_rows, len(counts))}")
    for i in range(min(max_rows, len(counts))):
        print(
            f"sample={i:04d} start={starts[i]} count={counts[i]} "
            f"time_ns={int(times[i])} time={_format_ns_time(times[i])}"
        )
    print("")





    

if __name__ == "__main__":
    zarr_path = r"F:\lyh_data\data_zarr_no_provider_filter\1bamua.zarr"
    n_context = 65_536
    n_target = 16_384
    target_overlap = 0.5
    seed = 0
    idx = 0
    dataset = BAMUAZarrDataset(
        zarr_path,
        n_context=n_context,
        n_target=n_target,
        target_overlap=target_overlap,
        seed=seed,
        return_indices=True,
    )
    _print_sample_counts(dataset)
    item = dataset[idx]
    context = item["context"]
    target = item["target"]
    context_idx = item["context_idx"]
    target_idx = item["target_idx"]
    overlap = len(set(context_idx.tolist()) & set(target_idx.tolist()))

    print(f"dataset_length={len(dataset)}")
    print(f"sample_idx={idx}")
    _shape_summary("sample_time", item["sample_time"])
    print(f"sample_time_readable={_format_ns_time(item['sample_time'].item())}")
    print(f"context_count={len(context_idx)}")
    print(f"target_count={len(target_idx)}")
    print(f"target_overlap_count={overlap}")
    print(f"target_overlap_fraction={overlap / max(len(target_idx), 1):.4f}")
    print("")
    print("[context]")
    for key, value in context.items():
        _shape_summary(key, value)
    print("")
    print("[target]")
    for key, value in target.items():
        _shape_summary(key, value)
    print("")
    _shape_summary("target_bt", item["target_bt"])
    _shape_summary("target_valid", item["target_valid"])
    print(f"target_valid_true_fraction={item['target_valid'].float().mean().item():.4f}")
