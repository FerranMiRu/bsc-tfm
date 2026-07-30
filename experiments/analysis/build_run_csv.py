"""Read chain tag file + per-job logs, emit one CSV row per job.

CSV columns: job_id, dataset, ranks, num_workers, seed, step, wall_s, startup_s,
samples_total, sps_wall, b_med, b_mean, b_p95.

`wall_s` and `startup_s` use SLURM `Start`/`End` from `sacct` (queried over ssh
to MN5) so they include Python startup, accelerate launch, and model init. If
sacct is unreachable, falls back to first-log-timestamp → `Training complete`.

`samples_total = EPOCHS × (TRAIN_STEPS + VAL_STEPS) × BATCH_SIZE × ranks`.

Usage:
    uv run python scripts/build_run_csv.py jobs-map/chain_<ts>_tags.tsv > run.csv
"""

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SSH_HOST = "mn5"

EPOCHS = 5
TRAIN_STEPS = 300
VAL_STEPS = 10
BATCH_SIZE = 8

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")
BATCH_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\].*Batch \[")
DONE_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\].*Epoch (\d+) done")
START_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\].*Starting training from epoch")
COMPLETE_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\].*Training complete")


def parse_ts(date_part: str, ms: str) -> datetime:
    return datetime.strptime(f"{date_part}.{ms}", "%Y-%m-%d %H:%M:%S.%f")


def parse_log(job_id: int) -> dict | None:
    path = Path(f"jobs-map/{job_id}/{job_id}.out")
    if not path.exists():
        return None

    lines = path.read_text().splitlines()

    job_start = train_start = train_end = None
    batches: list[datetime] = []
    dones: list[datetime] = []

    for line in lines:
        first = TS_RE.match(line)
        if first and job_start is None:
            job_start = parse_ts(first.group(1), first.group(2))

        start_match = START_RE.search(line)
        if start_match and train_start is None:
            train_start = parse_ts(start_match.group(1), start_match.group(2))

        complete_match = COMPLETE_RE.search(line)
        if complete_match:
            train_end = parse_ts(complete_match.group(1), complete_match.group(2))

        done_match = DONE_RE.search(line)
        if done_match:
            dones.append(parse_ts(done_match.group(1), done_match.group(2)))

        batch_match = BATCH_RE.search(line)
        if batch_match:
            batches.append(parse_ts(batch_match.group(1), batch_match.group(2)))

    if not (job_start and train_start and dones and batches):
        return None

    bounds = [train_start] + dones
    deltas: list[float] = []
    train_times: list[float] = []

    for epoch_idx in range(1, min(5, len(bounds) - 1)):
        epoch_batches = [ts for ts in batches if bounds[epoch_idx] < ts <= bounds[epoch_idx + 1]]

        for k in range(1, len(epoch_batches)):
            deltas.append((epoch_batches[k] - epoch_batches[k - 1]).total_seconds())

        if len(epoch_batches) >= 2:
            train_times.append((epoch_batches[-1] - epoch_batches[0]).total_seconds())

    if not deltas:
        return None

    deltas.sort()

    return {
        "job_start": job_start,
        "train_start": train_start,
        "train_end": train_end or batches[-1],
        "b_med": deltas[len(deltas) // 2],
        "b_mean": sum(deltas) / len(deltas),
        "b_p95": deltas[int(len(deltas) * 0.95)],
        "train_t": sum(train_times) / len(train_times) if train_times else 0.0,
    }


def fetch_sacct(job_ids: list[int]) -> dict[int, tuple[datetime | None, datetime | None]]:
    if not job_ids:
        return {}

    id_arg = ",".join(str(jid) for jid in job_ids)
    cmd = ["ssh", SSH_HOST, f"sacct -X -j {id_arg} --format=JobID,Start,End -P -n"]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=60).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    result: dict[int, tuple[datetime | None, datetime | None]] = {}

    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            jid = int(parts[0])
        except ValueError:
            continue

        start = datetime.strptime(parts[1], "%Y-%m-%dT%H:%M:%S") if parts[1] not in ("Unknown", "") else None
        end = datetime.strptime(parts[2], "%Y-%m-%dT%H:%M:%S") if parts[2] not in ("Unknown", "") else None
        result[jid] = (start, end)

    return result


def main() -> None:
    tag_file = sys.argv[1]
    rows: list[dict[str, str]] = []

    with open(tag_file) as handle:
        rows.extend(csv.DictReader(handle, delimiter="\t"))

    sacct = fetch_sacct([int(row["job_id"]) for row in rows])
    if sacct:
        print(f"# sacct: {len(sacct)} jobs fetched", file=sys.stderr)
    else:
        print("# sacct unavailable, falling back to log timestamps", file=sys.stderr)

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "job_id", "dataset", "ranks", "num_workers", "seed", "step",
        "wall_s", "startup_s", "samples_total", "sps_wall",
        "b_med", "b_mean", "b_p95",
    ])

    for row in rows:
        job_id = int(row["job_id"])
        dataset = row["dataset"]
        ranks = int(row["ranks"])
        workers = int(row["num_workers"])
        seed = int(row["seed"])
        step = int(row["step"])

        metrics = parse_log(job_id)
        if not metrics:
            continue

        slurm_start, slurm_end = sacct.get(job_id, (None, None))
        wall_start = slurm_start or metrics["job_start"]
        wall_end = slurm_end or metrics["train_end"]

        wall = (wall_end - wall_start).total_seconds()
        startup = (metrics["train_start"] - wall_start).total_seconds()
        samples = EPOCHS * (TRAIN_STEPS + VAL_STEPS) * BATCH_SIZE * ranks

        denom = metrics["train_t"] + metrics["b_mean"]
        sps_wall = (TRAIN_STEPS * BATCH_SIZE * ranks) / denom if denom else 0.0

        writer.writerow([
            job_id, dataset, ranks, workers, seed, step,
            f"{wall:.1f}", f"{startup:.1f}", samples, f"{sps_wall:.2f}",
            f"{metrics['b_med']:.3f}", f"{metrics['b_mean']:.3f}", f"{metrics['b_p95']:.3f}",
        ])


if __name__ == "__main__":
    main()
