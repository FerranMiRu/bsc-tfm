"""Per-chunk-read latency trace to separate cold vs warm zarr opens.

Monkey-patches the zarr store __getitem__ to time every chunk-data read, then
replays NUM_SAMPLES random land-pixel samples through the dataset for each arm.
The per-read latency histogram is bimodal (warm ~0.1 ms / cold ~20 ms); the cold
fraction is the miss rate m. Arms vary the working set so we can test whether a
bigger working set -> more cold opens -> longer per-sample time:

  single_match : bundled single store,  14-yr span  (small working set, reference)
  multi_match  : 5-store multizarr,     14-yr span  (same span, different layout)
  multi_full   : 5-store multizarr,     113-yr span (same layout, huge working set)

multi_full vs multi_match isolates working set with zero layout confound.

CSV stdout: arm,store,sample_idx,chunk_key,wall_us
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
    ("single_match", SingleZarrDataset, SINGLE_CONFIG, [2001, 2014]),
    ("multi_match", MultiZarrDataset, MULTI_CONFIG, [2001, 2014]),
    ("multi_full", MultiZarrDataset, MULTI_CONFIG, [1902, 2014]),
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
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_cw_")) / "cw"
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
