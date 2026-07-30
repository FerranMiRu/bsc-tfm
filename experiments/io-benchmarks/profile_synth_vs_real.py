"""Synthetic 1M pool (sizes from real chunk distribution) vs the real zarr store.

1000 random reads from each (2000 total, shuffled interleaved).
"""

import csv
import random
import shutil
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/tmp/synth_vs_real")
REAL_STORE = Path(
    "/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
)
CHUNK_DIST_CSV = Path(
    "/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/jobs-map/41548701/chunk_sizes.csv"
)
EXCLUDE = {"time", "latitude", "longitude"}

SYNTH_COUNT = 1_000_000
N_READS = 1000
SEED = 42

chunk_sizes = []
with open(CHUNK_DIST_CSV) as f:
    for row in csv.DictReader(f):
        if row["var"] not in EXCLUDE:
            chunk_sizes.append(int(row["size_bytes"]))

shutil.rmtree(BASE_DIR, ignore_errors=True)
synth_dir = BASE_DIR / "synth"
synth_dir.mkdir(parents=True)

size_rng = random.Random(SEED)
for i in range(SYNTH_COUNT):
    size = size_rng.choice(chunk_sizes)
    (synth_dir / f"f{i}").write_bytes(b"\0" * size)

rng = random.Random(SEED)

real_chunks = []
for var_dir in sorted(REAL_STORE.iterdir()):
    if not var_dir.is_dir():
        continue

    for entry in var_dir.iterdir():
        parts = entry.name.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            real_chunks.append(entry)

synth_paths = [
    (synth_dir / f"f{i}", "synth")
    for i in rng.sample(range(SYNTH_COUNT), N_READS)
]
real_paths = [(p, "real") for p in rng.sample(real_chunks, N_READS)]

reads = synth_paths + real_paths
rng.shuffle(reads)

print("source,path,wall_us,file_bytes")
for path, source in reads:
    start = time.perf_counter_ns()
    with open(path, "rb") as fh:
        data = fh.read()
    wall_us = (time.perf_counter_ns() - start) / 1000.0
    print(f"{source},{path},{wall_us:.2f},{len(data)}")

shutil.rmtree(BASE_DIR, ignore_errors=True)
