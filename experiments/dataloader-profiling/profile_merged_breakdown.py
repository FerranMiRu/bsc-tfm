"""Per-method timing for MergedZarrDataset.__getitem__.

Wraps every helper with time.perf_counter and splits `_read_dynamic`
into `isel` (lazy graph build) and `.to_numpy()` (compute + materialize)
so we can see whether the merged Phase-1 ~200 ms is dominated by the
xarray graph build, the compute, or the post-processing extracts.

CSV stdout: sample_idx,phase,wall_ms.
Phases: __getitem__ (whole), _parse_index, _read_dynamic_isel,
_read_dynamic_to_numpy, _build_continuous_sequence, _extract_kg_sequence,
_process_static, _extract_hilda_prior, _extract_hilda_target.
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, SpatialDimsNames


CONFIG = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")) / "inputs/debug.yaml"
MERGED_STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"

NUM_SAMPLES = 100
SEED = 42

HELPERS = (
    "_parse_index",
    "_build_continuous_sequence",
    "_extract_kg_sequence",
    "_process_static",
    "_extract_hilda_prior",
    "_extract_hilda_target",
)


def build_dataset(raw_data):
    lock_dir = Path(tempfile.mkdtemp(prefix="ai4land_breakdown_"))
    cfg_data = dict(raw_data)
    cfg_data["lock_dir"] = lock_dir
    cfg_data["stores"] = MERGED_STORE
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return MergedZarrDataset(cfg=dp, land_index_path=dp.train_land_index_path)


def install_timers(dataset, timings):
    for name in HELPERS:
        original = getattr(dataset, name)

        def make_wrapper(orig, key):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = orig(*args, **kwargs)
                timings[key].append((time.perf_counter() - start) * 1000.0)
                return result

            return wrapper

        setattr(dataset, name, make_wrapper(original, name))

    def timed_read_dynamic(t_idx, lat_slice, lon_slice):
        needed = dataset._compute_dynamic_timesteps(t_idx)

        if not needed:
            timings["_read_dynamic_isel"].append(0.0)
            timings["_read_dynamic_to_numpy"].append(0.0)
            return np.empty((0, 0, 0, 0), dtype=np.float32), {}

        ts_sorted = sorted(needed)

        t0 = time.perf_counter()
        arr = dataset.dataset["dynamic"].isel(
            time=ts_sorted,
            **{
                SpatialDimsNames.LATITUDE.value: lat_slice,
                SpatialDimsNames.LONGITUDE.value: lon_slice,
            },
        )
        t1 = time.perf_counter()
        dyn = arr.to_numpy()
        t2 = time.perf_counter()

        timings["_read_dynamic_isel"].append((t1 - t0) * 1000.0)
        timings["_read_dynamic_to_numpy"].append((t2 - t1) * 1000.0)

        ts_to_idx = {ts: i for i, ts in enumerate(ts_sorted)}
        return dyn, ts_to_idx

    dataset._read_dynamic = timed_read_dynamic


def main():
    raw = yaml.safe_load(CONFIG.open())
    raw_data = raw["data"]

    dataset = build_dataset(raw_data)

    phases = (*HELPERS, "_read_dynamic_isel", "_read_dynamic_to_numpy", "__getitem__")
    timings = {p: [] for p in phases}

    install_timers(dataset, timings)

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(dataset), size=NUM_SAMPLES).tolist()

    for idx in indices:
        start = time.perf_counter()
        dataset[int(idx)]
        timings["__getitem__"].append((time.perf_counter() - start) * 1000.0)

    print("sample_idx,phase,wall_ms")
    for phase in phases:
        for sample_i, ms in enumerate(timings[phase]):
            print(f"{sample_i},{phase},{ms:.3f}")


if __name__ == "__main__":
    main()
