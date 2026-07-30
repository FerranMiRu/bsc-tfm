"""Generate sbatch commands for a multi-rep training-loop benchmark chain.

Each rep is 18 jobs alternating merged ↔ single. Within each side, the 9 (ranks,
num_workers) combinations from {1, 4, 8} × {8, 12, 20} are independently shuffled
under the given seed. All jobs run sequentially via `--dependency=afterany` so no
two zarr-store readers overlap. Multiple reps are chained back-to-back.

Each submission also appends one row to a tag TSV
(`jobs-map/chain_<unix_ts>_tags.tsv`) so downstream analysis can recover
job_id → (dataset, ranks, num_workers, seed, step).

Usage:
    uv run python scripts/build_run_chain.py 42 43 44 > chain_cmds.sh
    ssh mn5 'cd /gpfs/scratch/bsc32/bsc096444/ai4land-tfm && bash -s' < chain_cmds.sh
"""

import random
import sys

CONFIGS = [(rank, workers) for rank in (1, 4, 8) for workers in (8, 12, 20)]


def build_rep(seed: int, base_step: int) -> list[tuple[str, int, int, int]]:
    """One rep: 18 jobs (m, s, m, s, …) with independently shuffled (rank, worker)."""
    rng = random.Random(seed)
    merged = list(CONFIGS)
    single = list(CONFIGS)
    rng.shuffle(merged)
    rng.shuffle(single)

    jobs = []

    for slot in range(9):
        jobs.append(("merged", *merged[slot], base_step + 2 * slot + 1))
        jobs.append(("single", *single[slot], base_step + 2 * slot + 2))

    return jobs


def emit(seeds: list[int]) -> None:
    all_jobs = []
    for seed in seeds:
        all_jobs.extend([(seed, *job) for job in build_rep(seed, len(all_jobs))])

    print("#!/usr/bin/env bash")
    print(f"# {len(all_jobs)} jobs across {len(seeds)} rep(s); seeds={seeds}")
    print("set -e")
    print('cd "${HOME%/}/../bsc096444/ai4land-tfm" 2>/dev/null || cd /gpfs/scratch/bsc32/bsc096444/ai4land-tfm')
    print('TAG_FILE="jobs-map/chain_$(date +%s)_tags.tsv"')
    print('mkdir -p jobs-map')
    print('printf "job_id\\tdataset\\tranks\\tnum_workers\\tseed\\tstep\\n" > "$TAG_FILE"')
    print('PREV=""')

    for seed, dataset, ranks, workers, step in all_jobs:
        script = f"scripts/launch_bench_{ranks}rank.sh"
        config = f"debug-{dataset}"
        overrides = f"training.num_workers={workers}"

        print(f"# step {step}: seed={seed} {dataset} {ranks}r×{workers}")
        print(f'PREV=$(sbatch --parsable ${{PREV:+--dependency=afterany:$PREV}} {script} {config} {overrides})')
        print(f'printf "%s\\t%s\\t%d\\t%d\\t%d\\t%d\\n" "$PREV" "{dataset}" {ranks} {workers} {seed} {step} >> "$TAG_FILE"')
        print(f'echo "step {step:>3}: $PREV  {dataset} {ranks}r×{workers} seed={seed}"')

    print('echo "Tag file: $TAG_FILE"')


def main() -> None:
    seeds = [int(arg) for arg in sys.argv[1:]] or [42]
    emit(seeds)


if __name__ == "__main__":
    main()
