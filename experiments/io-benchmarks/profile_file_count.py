"""Sweep file_count x file_size on GPFS; time per-file random open+read.

Each cell: pool of N files; read up to N_SAMPLES random files, individually
timed. Tests whether per-open cost grows with pool size (directory-size /
metadata-cache hypothesis).

Caveat: page cache is warm during reads since Phase 1 just wrote the files.
"""

import random
import shutil
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/tmp/file_count_test")
SIZES = [10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]
COUNTS = [1, 10, 100, 1000, 10000]
MAX_TOTAL = 10 * 1024**3
N_SAMPLES = 10000
SEED = 42

shutil.rmtree(BASE_DIR, ignore_errors=True)
BASE_DIR.mkdir(parents=True)

for size in SIZES:
    chunk = b"\0" * size
    for count in COUNTS:
        if size * count > MAX_TOTAL:
            continue
        cell_dir = BASE_DIR / f"s{size}_n{count}"
        cell_dir.mkdir()
        for i in range(count):
            (cell_dir / f"f{i}").write_bytes(chunk)

rng = random.Random(SEED)

reads = []
for size in SIZES:
    for count in COUNTS:
        if size * count > MAX_TOTAL:
            continue
        cell_dir = BASE_DIR / f"s{size}_n{count}"
        for i in rng.sample(range(count), min(count, N_SAMPLES)):
            reads.append((cell_dir, size, count, i))

rng.shuffle(reads)

print("pool_size,file_size,file_index,wall_us")
for cell_dir, size, count, i in reads:
    start = time.perf_counter_ns()
    with open(cell_dir / f"f{i}", "rb") as fh:
        fh.read()
    wall_us = (time.perf_counter_ns() - start) / 1000.0
    print(f"{count},{size},{i},{wall_us:.2f}")

shutil.rmtree(BASE_DIR, ignore_errors=True)
