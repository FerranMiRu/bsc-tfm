"""Write per-batch wall-clock deltas (CSV) and report total wall for ai4land training jobs."""

import csv
import re
import statistics
from datetime import datetime
from pathlib import Path


BATCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*Epoch \[(\d+)/\d+\], Batch \[(\d+)/"
)
WALL_RE = re.compile(r"(START|END) TIME: (.+)")

TRAINING_LOGS = Path("/Users/fmr/development/bsc/ai4land-tfm/logs/training")


def parse_wall(raw: str) -> datetime:
    parts = raw.strip().split()
    parts = parts[:4] + parts[5:]  # drop the timezone token (e.g. CEST)
    return datetime.strptime(" ".join(parts), "%a %b %d %H:%M:%S %Y")


def parse_log(out_path: Path) -> tuple[list[tuple[int, int, datetime]], datetime, datetime]:
    batches = []
    start = end = None

    for line in out_path.read_text().splitlines():
        batch_match = BATCH_RE.search(line)

        if batch_match:
            ts = datetime.strptime(batch_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            batches.append((int(batch_match.group(2)), int(batch_match.group(3)), ts))

        wall_match = WALL_RE.search(line)
        if wall_match:
            when = parse_wall(wall_match.group(2))
            if wall_match.group(1) == "START":
                start = when
            else:
                end = when

    return batches, start, end


def write_csv(csv_path: Path, batches: list[tuple[int, int, datetime]]) -> list[float]:
    deltas = []

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "batch", "delta_s"])

        for i in range(1, len(batches)):
            epoch, batch_num, ts = batches[i]
            prev_epoch, _, prev_ts = batches[i - 1]
            delta = (ts - prev_ts).total_seconds()

            if epoch == prev_epoch:
                writer.writerow([epoch, batch_num, f"{delta:.3f}"])
                deltas.append(delta)

    return deltas


def report(job_dir: Path) -> None:
    out_path = job_dir / f"{job_dir.name}.out"

    if not out_path.exists():
        return

    batches, start, end = parse_log(out_path)

    if not batches:
        print(f"{job_dir.name}: no batch logs")
        return

    csv_path = job_dir / "batch_times.csv"
    deltas = write_csv(csv_path, batches)

    print(f"\n=== {job_dir.name} ===")
    print(f"  wrote {csv_path}  ({len(deltas)} intra-epoch deltas)")

    if deltas:
        ordered = sorted(deltas)
        p50 = statistics.median(ordered)
        mean = statistics.mean(ordered)
        p95 = ordered[int(len(ordered) * 0.95)]
        stalls = sum(1 for d in ordered if d > 1.0)
        print(
            f"  per-batch s: p50={p50:.3f} mean={mean:.3f} p95={p95:.3f} "
            f"max={ordered[-1]:.3f} stalls>1s={stalls}"
        )
        print(f"  samples/s/gpu (batch=8): p50={8 / p50:.1f} mean={8 / mean:.1f}")

    first_ts, last_ts = batches[0][2], batches[-1][2]
    print(f"  training span (first->last batch): {(last_ts - first_ts).total_seconds():.0f}s")

    if start and end:
        total = (end - start).total_seconds()
        print(f"  TOTAL WALL (START->END): {total:.0f}s = {total / 3600:.2f}h")


def main() -> None:
    for job_dir in sorted(TRAINING_LOGS.iterdir()):
        if job_dir.is_dir():
            report(job_dir)


if __name__ == "__main__":
    main()
