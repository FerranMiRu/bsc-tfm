"""ABAB rerun of Run 16 data: A1 -> B1 -> A2 -> B2.

Same dataloader_sim stores as Run 19. The extra B2 pass tells us whether B's
measurement also warms within a job (small pool → fast cache fill) or whether
B is stable across passes.
"""

import multiprocessing
import random
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/dataloader_sim")
STORE_A = BASE_DIR / "store_A_23dirs"
STORE_B = BASE_DIR / "store_B_1dir"

N_SUBDIRS_A = 23
FILES_PER_SUBDIR_A = 13000
FILES_PER_SAMPLE_A = 40

N_FILES_B = 13000
FILES_PER_SAMPLE_B = 2

N_WORKERS = 12
SAMPLES_PER_WORKER = 100
SEED = 42


def worker_a(args):
    worker_id, seed = args
    rng = random.Random(seed)
    results = []

    for sample in range(SAMPLES_PER_WORKER):
        start = time.perf_counter_ns()

        for _ in range(FILES_PER_SAMPLE_A):
            d = rng.randrange(N_SUBDIRS_A)
            i = rng.randrange(FILES_PER_SUBDIR_A)

            with open(STORE_A / f"sub{d}" / f"f{i}", "rb") as fh:
                fh.read()

        ms = (time.perf_counter_ns() - start) / 1e6
        results.append((worker_id, sample, ms))

    return results


def worker_b(args):
    worker_id, seed = args
    rng = random.Random(seed)
    results = []

    for sample in range(SAMPLES_PER_WORKER):
        start = time.perf_counter_ns()

        for _ in range(FILES_PER_SAMPLE_B):
            i = rng.randrange(N_FILES_B)
            with open(STORE_B / "sub0" / f"f{i}", "rb") as fh:
                fh.read()

        ms = (time.perf_counter_ns() - start) / 1e6
        results.append((worker_id, sample, ms))

    return results


def main():
    with multiprocessing.Pool(N_WORKERS) as pool:
        a1 = pool.map(worker_a, [(i, SEED + 1000 + i) for i in range(N_WORKERS)])
        b1 = pool.map(worker_b, [(i, SEED + 2000 + i) for i in range(N_WORKERS)])
        a2 = pool.map(worker_a, [(i, SEED + 3000 + i) for i in range(N_WORKERS)])
        b2 = pool.map(worker_b, [(i, SEED + 4000 + i) for i in range(N_WORKERS)])

    print("store,pass,worker,sample,wall_ms")

    for batch in a1:
        for w, s, ms in batch:
            print(f"A,1,{w},{s},{ms:.2f}")

    for batch in b1:
        for w, s, ms in batch:
            print(f"B,1,{w},{s},{ms:.2f}")

    for batch in a2:
        for w, s, ms in batch:
            print(f"A,2,{w},{s},{ms:.2f}")

    for batch in b2:
        for w, s, ms in batch:
            print(f"B,2,{w},{s},{ms:.2f}")


if __name__ == "__main__":
    main()
