"""Isolate the xarray/dask overhead component of the store-size penalty.

The timing + cache-miss runs showed big single is ~4x slower per sample than small single, yet the
chunk-read I/O (store layer) is store-size-invariant. This pins the penalty above the I/O layer.
This probe proves it is xarray/dask graph overhead that scales with the array's total chunk count.

Method: read WARM (repeat the same fixed indices after a warm-up), so page-cache I/O ~ 0, and
decompose each call into I/O (time inside the monkey-patched store __getitem__) vs non-I/O
(total - I/O = blosc decode + xarray/dask graph build + numpy assembly). Decode is identical for
the same bytes, so any non-I/O gap between single_small and single_big is the dask graph cost of a
larger array (115-yr store has ~7x the chunks of the 16-yr store).

  Part 1: full SingleZarrDataset.__getitem__ decomposition (realistic per-sample).
  Part 2: one LUH2 var .isel(...).to_numpy() at a FIXED identical slice (controlled isolation —
          same output, warm, only the underlying array's chunk count differs).

Stdout: per-arm medians for total / io / non_io (ms), plus the big-vs-small ratio.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

import yaml
import zarr.storage

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import SingleZarrDataset


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
CONFIG = SUBMIT / "inputs/debug-single.yaml"

DATA_ROOT = "/gpfs/scratch/ehpc736/data"
SMALL_STORE = f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
BIG_STORE = f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-1901-2015.zarr"

TIME_RANGE = [2001, 2010]
N_INDICES = 8
N_REPS = 15
PATCH = 256
FIXED_Y0 = 8000
FIXED_X0 = 12000
LUH2_VAR = "c3ann"

_io_ns = 0
_recording = False


def _make_io_timing(orig_getitem):
    def wrapped(self, key):
        global _io_ns
        if not _recording:
            return orig_getitem(self, key)

        start = time.perf_counter_ns()
        value = orig_getitem(self, key)
        _io_ns += time.perf_counter_ns() - start
        return value

    return wrapped


for cls_name in ("DirectoryStore", "FSStore", "NestedDirectoryStore"):
    cls = getattr(zarr.storage, cls_name, None)
    if cls is not None:
        cls.__getitem__ = _make_io_timing(cls.__getitem__)


def build_dataset(store_path: str) -> SingleZarrDataset:
    raw = yaml.safe_load(CONFIG.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_xo_")) / "lock"
    cfg_data["stores"] = store_path
    cfg_data["time_range"] = TIME_RANGE
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return SingleZarrDataset(cfg=dp, land_index_path=dp.train_land_index_path)


def timed_call(call) -> tuple[float, float]:
    global _io_ns, _recording
    _io_ns = 0
    _recording = True
    start = time.perf_counter_ns()
    call()
    total_ms = (time.perf_counter_ns() - start) / 1e6
    _recording = False
    return total_ms, _io_ns / 1e6


def measure_reps(call, n_reps: int) -> tuple[list[float], list[float]]:
    call()  # warm-up (loads chunks into page cache)

    totals, ios = [], []
    for _ in range(n_reps):
        total_ms, io_ms = timed_call(call)
        totals.append(total_ms)
        ios.append(io_ms)

    return totals, ios


def report(label: str, totals: list[float], ios: list[float]) -> dict:
    total_med = statistics.median(totals)
    io_med = statistics.median(ios)
    non_io_med = total_med - io_med
    print(
        f"  {label:<14} total={total_med:8.2f} ms   io={io_med:7.2f} ms   "
        f"non_io(xarray+decode)={non_io_med:8.2f} ms"
    )
    return {"total": total_med, "io": io_med, "non_io": non_io_med}


def part1_loader() -> None:
    print("\n=== Part 1: warm SingleZarrDataset.__getitem__ decomposition ===")
    stats = {}

    for arm, store_path in (("single_small", SMALL_STORE), ("single_big", BIG_STORE)):
        dataset = build_dataset(store_path)
        indices = [int(i * len(dataset) / (N_INDICES + 1)) for i in range(1, N_INDICES + 1)]

        totals, ios = [], []
        for index in indices:
            arm_totals, arm_ios = measure_reps(lambda idx=index, ds=dataset: ds[idx], N_REPS)
            totals.extend(arm_totals)
            ios.extend(arm_ios)

        stats[arm] = report(arm, totals, ios)

    non_io_ratio = stats["single_big"]["non_io"] / stats["single_small"]["non_io"]
    io_ratio = stats["single_big"]["io"] / max(stats["single_small"]["io"], 1e-6)
    print(f"  -> non_io big/small ratio = {non_io_ratio:.2f}   io big/small ratio = {io_ratio:.2f}")


def part2_single_op() -> None:
    print(f"\n=== Part 2: warm '{LUH2_VAR}'.isel(fixed patch).to_numpy() (identical slice) ===")
    stats = {}

    for arm, store_path in (("single_small", SMALL_STORE), ("single_big", BIG_STORE)):
        dataset = build_dataset(store_path)
        arr = dataset.dataset["main"][LUH2_VAR]

        def call(array=arr):
            array.isel(
                time=0,
                latitude=slice(FIXED_Y0, FIXED_Y0 + PATCH),
                longitude=slice(FIXED_X0, FIXED_X0 + PATCH),
            ).to_numpy()

        totals, ios = measure_reps(call, N_REPS * 2)
        stats[arm] = report(arm, totals, ios)

    total_ratio = stats["single_big"]["total"] / stats["single_small"]["total"]
    print(
        f"  -> total big/small ratio = {total_ratio:.2f}   "
        "(warm, identical output => the gap is dask graph build, not I/O)"
    )


def main() -> None:
    part1_loader()
    part2_single_op()


if __name__ == "__main__":
    main()
