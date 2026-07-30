"""Cache-miss proof: small single zarr vs big single zarr.

`single_small` and `single_big` are read over the SAME 14-year window, so per-sample chunk-access
pattern and file-open count are identical -- the only difference is the store's total chunk-file
(inode) count on disk. `single_big_full` reads the big store over its full 113-year span as a
working-set control. Every cold/warm chunk open is timed via a monkey-patched store __getitem__.

If at the matched window `single_big` shows a higher miss rate / cold-open cost than `single_small`,
the store-size effect is the inode/dentry metadata cache-miss rate and nothing else. The big_full
arm connects that miss rate to the touched working set within one fixed layout.

CSV stdout: arm,store,sample_idx,chunk_key,wall_us  (read by tests/analyze_cache_miss.py).
"""

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml
import zarr.storage

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import SingleZarrDataset


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
CONFIG = SUBMIT / "inputs/debug-single.yaml"

DATA_ROOT = "/gpfs/scratch/ehpc736/data"
SMALL_STORE = f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
BIG_STORE = f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-1901-2015.zarr"

NUM_SAMPLES = 150
SEED = 42

ARMS = [
    ("single_small", SMALL_STORE, [2001, 2014]),
    ("single_big", BIG_STORE, [2001, 2014]),
    ("single_big_full", BIG_STORE, [1902, 2014]),
]

_chunk_re = re.compile(r"^\d+(\.\d+)*$")
_log = []
_recording = False


def _store_label(store) -> str:
    path = getattr(store, "path", "") or getattr(store, "dir_path", "") or ""
    return Path(path).name if path else ""


def _make_timing(orig_getitem):
    def _wrapped(self, key):
        parts = key.rsplit("/", 1)
        is_chunk = len(parts) == 2 and _chunk_re.fullmatch(parts[1])

        if not (_recording and is_chunk):
            return orig_getitem(self, key)

        start = time.perf_counter_ns()
        value = orig_getitem(self, key)
        wall_us = (time.perf_counter_ns() - start) / 1000.0
        _log.append((_store_label(self), key, wall_us))
        return value

    return _wrapped


for cls_name in ("DirectoryStore", "FSStore", "NestedDirectoryStore"):
    cls = getattr(zarr.storage, cls_name, None)
    if cls is not None:
        cls.__getitem__ = _make_timing(cls.__getitem__)


def build_dataset(store_path, time_range):
    raw = yaml.safe_load(CONFIG.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_cm_")) / "lock"
    cfg_data["stores"] = store_path
    cfg_data["time_range"] = time_range
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return SingleZarrDataset(cfg=dp, land_index_path=dp.train_land_index_path)


def measure(arm, store_path, time_range) -> None:
    global _recording
    dataset = build_dataset(store_path, time_range)

    torch.manual_seed(SEED)
    indices = torch.randint(0, len(dataset), (NUM_SAMPLES,)).tolist()

    _recording = True
    for sample_idx, idx in enumerate(indices):
        _log.clear()
        dataset[idx]

        for store_label, key, wall_us in _log:
            print(f"{arm},{store_label},{sample_idx},{key},{wall_us:.2f}")
        sys.stdout.flush()
    _recording = False


def main() -> None:
    print("arm,store,sample_idx,chunk_key,wall_us")

    for arm, store_path, time_range in ARMS:
        measure(arm, store_path, time_range)


if __name__ == "__main__":
    main()
