"""Replication of Run 38's cold-cost-vs-working-set finding, as a span sweep.

Same probe mechanism as profile_cold_warm.py (time every cold chunk open), but
the arms sweep ONLY the time-window width on the SAME multizarr stores, so the
distinct-inode working set grows monotonically while layout is held fixed. If the
per-cold-open cost rises monotonically with span, the Run 38 finding replicates.

Arms (narrow -> wide; later arms run on a warmer node -> conservative):
  single_14  : bundled single store, 14-yr span (matched-span control / anchor)
  multi_14   : multizarr, 14-yr span   (~36 K LUH2 inodes)
  multi_40   : multizarr, 40-yr span
  multi_66   : multizarr, 66-yr span
  multi_113  : multizarr, 113-yr span  (~283 K LUH2 inodes)

CSV stdout: arm,store,sample_idx,chunk_key,wall_us  (same schema as Run 38, so
scripts/analyze_cold_warm.py reads it directly).
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
from ai4land.utils.datasets import MultiZarrDataset, SingleZarrDataset


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
SINGLE_CONFIG = SUBMIT / "inputs/debug-single.yaml"
MULTI_CONFIG = SUBMIT / "inputs/profiling-multizarr.yaml"

ARMS = [
    ("single_14", SingleZarrDataset, SINGLE_CONFIG, [2001, 2014]),
    ("multi_14", MultiZarrDataset, MULTI_CONFIG, [2001, 2014]),
    ("multi_40", MultiZarrDataset, MULTI_CONFIG, [1975, 2014]),
    ("multi_66", MultiZarrDataset, MULTI_CONFIG, [1948, 2014]),
    ("multi_113", MultiZarrDataset, MULTI_CONFIG, [1902, 2014]),
]

NUM_SAMPLES = 80
SEED = 42

_chunk_re = re.compile(r"^\d+(\.\d+)*$")
_log = []
_recording = False


def _store_label(store):
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


def build_dataset(dataset_class, config_path, time_range):
    raw = yaml.safe_load(config_path.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_cs_")) / "cs"
    cfg_data["time_range"] = time_range
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return dataset_class(cfg=dp, land_index_path=dp.train_land_index_path)


def measure(arm, dataset_class, config_path, time_range):
    global _recording
    dataset = build_dataset(dataset_class, config_path, time_range)

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


def main():
    print("arm,store,sample_idx,chunk_key,wall_us")

    for arm, dataset_class, config_path, time_range in ARMS:
        measure(arm, dataset_class, config_path, time_range)


if __name__ == "__main__":
    main()
