"""Shared parsing for the dataloader-experiment result logs.

Logs live under ``../runs/final-training/`` (relative to this package). Each training job writes two
files under ``runs/final-training/training/<jid>/``:

- ``<jid>.out``              — SLURM stdout: the accelerate launch command (carries
                              ``--num_machines`` and ``--config-name``) plus START/END TIME markers.
- ``unet_training_<jid>.log`` — per-batch and per-epoch INFO lines.

Each evaluation writes ``runs/final-training/evaluation/<jid>.out``; each probe writes
``runs/final-training/probes/<jid>/<jid>.out``. The functions here turn those files into typed
records the analyzer modules group and compare.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "final-training"
TRAINING_DIR = RUNS / "training"
EVAL_DIR = RUNS / "evaluation"
PROBE_DIR = RUNS / "probes"

BATCH_SIZE = 8
STALL_MS = 1000.0
WARMUP_BATCHES = 50
MIN_VALID_BATCHES = 60
EXPECTED_EPOCHS = 10
DEGRADED_MEAN_MS = 1000.0
DEAD_MEAN_MS = 5000.0

_BATCH_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Epoch \[(\d+)/\d+\], Batch \[(\d+)/"
)
_VAL_RE = re.compile(r"Epoch \[(\d+)/\d+\] \| Train: ([\d.]+) \| Val: ([\d.]+)")
_MACHINES_RE = re.compile(r"--num_machines (\d+)")
_CONFIG_RE = re.compile(r"--config-name (\S+)")
_CHECKPOINT_RE = re.compile(r"unet_impl/(\d+)/best_model")
_WALL_RE = re.compile(r"(START|END) TIME: (.+)")


def parse_loader_workers(config_name: str) -> tuple[str, int]:
    """Map an accelerate ``--config-name`` to (loader, workers_per_rank).

    ``final-training`` -> (merged, 12); ``final-training-w20`` -> (merged, 20);
    ``final-training-raw-w12`` -> (raw, 12).
    """
    loader = "raw" if "raw" in config_name else "merged"
    suffix = re.search(r"w(\d+)$", config_name)
    workers = int(suffix.group(1)) if suffix else 12
    return loader, workers


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def coefficient_of_variation(values: list[float]) -> float:
    usable = [v for v in values if v is not None]

    if len(usable) < 2 or statistics.mean(usable) == 0:
        return 0.0

    return statistics.pstdev(usable) / statistics.mean(usable)


def _parse_wall_token(raw: str) -> datetime:
    parts = raw.strip().split()
    parts = parts[:4] + parts[5:]
    return datetime.strptime(" ".join(parts), "%a %b %d %H:%M:%S %Y")


def _wall_hours(out_text: str) -> float | None:
    start = end = None

    for marker in _WALL_RE.finditer(out_text):
        when = _parse_wall_token(marker.group(2))

        if marker.group(1) == "START":
            start = when
        else:
            end = when

    if start is None or end is None:
        return None

    return (end - start).total_seconds() / 3600.0


def _intra_epoch_deltas(log_path: Path) -> list[float]:
    timestamps = []

    for line in log_path.read_text().splitlines():
        match = _BATCH_RE.search(line)

        if match:
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            timestamps.append((int(match.group(2)), stamp))

    deltas = []

    for previous, current in pairwise(timestamps):
        if previous[0] == current[0]:
            deltas.append((current[1] - previous[1]).total_seconds() * 1000.0)

    return deltas


def _val_curve(log_path: Path) -> list[tuple[int, float, float]]:
    curve = []

    for line in log_path.read_text().splitlines():
        match = _VAL_RE.search(line)

        if match:
            curve.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))

    return curve


@dataclass
class TrainingRun:
    jid: str
    loader: str
    nodes: int
    workers: int
    n_batches: int
    p50_ms: float
    mean_ms: float
    p95_ms: float
    stall_pct: float
    sps_per_gpu: float
    wall_hours: float | None
    best_val: float | None
    epochs: int

    @property
    def category(self) -> str:
        return f"{self.nodes}r{self.workers}w"

    @property
    def group(self) -> str:
        return f"{self.loader}/{self.category}"

    @property
    def complete(self) -> bool:
        return self.epochs >= EXPECTED_EPOCHS

    @property
    def degraded(self) -> bool:
        return self.mean_ms > DEGRADED_MEAN_MS

    @property
    def flag(self) -> str:
        if self.degraded:
            return "DEGRADED"

        if not self.complete:
            return f"INCOMPLETE/ep{self.epochs}"

        return "ok"


def _parse_training(job_dir: Path) -> TrainingRun | None:
    jid = job_dir.name
    out_path = job_dir / f"{jid}.out"
    log_path = job_dir / f"unet_training_{jid}.log"

    if not out_path.exists() or not log_path.exists():
        return None

    out_text = out_path.read_text()
    machines = _MACHINES_RE.search(out_text)
    config = _CONFIG_RE.search(out_text)

    if not machines or not config:
        return None

    deltas = _intra_epoch_deltas(log_path)

    if len(deltas) < MIN_VALID_BATCHES:
        return None

    steady = deltas[WARMUP_BATCHES:] if len(deltas) > 2 * WARMUP_BATCHES else deltas
    mean_ms = statistics.mean(steady)
    curve = _val_curve(log_path)
    loader, workers = parse_loader_workers(config.group(1))

    return TrainingRun(
        jid=jid,
        loader=loader,
        nodes=int(machines.group(1)),
        workers=workers,
        n_batches=len(deltas),
        p50_ms=statistics.median(steady),
        mean_ms=mean_ms,
        p95_ms=percentile(steady, 0.95),
        stall_pct=100.0 * sum(1 for d in steady if d > STALL_MS) / len(steady),
        sps_per_gpu=1000.0 * BATCH_SIZE / mean_ms,
        wall_hours=_wall_hours(out_text),
        best_val=min((v for _, _, v in curve), default=None),
        epochs=len(curve),
    )


def load_training_runs() -> list[TrainingRun]:
    runs = [_parse_training(job_dir) for job_dir in sorted(TRAINING_DIR.glob("4*"))]
    return [run for run in runs if run is not None and run.mean_ms <= DEAD_MEAN_MS]


@dataclass
class EvalRun:
    jid: str
    train_jid: str
    nodes: int
    year_accuracy: dict[int, float] = field(default_factory=dict)
    year_macro_f1: dict[int, float] = field(default_factory=dict)
    class_accuracy: dict[str, float] = field(default_factory=dict)


_YEAR_RE = re.compile(r"Year (\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)")


def _parse_eval(out_path: Path) -> EvalRun | None:
    text = out_path.read_text()
    checkpoint = _CHECKPOINT_RE.search(text)
    machines = _MACHINES_RE.search(text)

    if not checkpoint:
        return None

    run = EvalRun(
        jid=out_path.stem,
        train_jid=checkpoint.group(1),
        nodes=int(machines.group(1)) if machines else 0,
    )

    for match in _YEAR_RE.finditer(text):
        year = int(match.group(1))
        run.year_accuracy[year] = float(match.group(2))
        run.year_macro_f1[year] = float(match.group(3))

    class_header = re.search(r"Metric\s+\|\s+(Ocean.+)", text)
    class_values = re.search(r"Accuracy\s+\|\s+([\d.\s|]+)", text)

    if class_header and class_values:
        names = [name.strip() for name in class_header.group(1).split("|") if name.strip()]
        scores = [float(value) for value in re.findall(r"[\d.]+", class_values.group(1))]
        run.class_accuracy = dict(zip(names, scores, strict=False))

    return run


def load_eval_runs() -> list[EvalRun]:
    runs = [_parse_eval(path) for path in sorted(EVAL_DIR.glob("*.out"))]
    return [run for run in runs if run is not None]


@dataclass
class SampleTimingArm:
    median_ms: float
    mean_ms: float
    cold_ms: float
    warm_ms: float


_SAMPLE_RE = re.compile(
    r"(\w+)\s+median=\s*([\d.]+) ms\s+mean=\s*([\d.]+) ms\s+min=\s*[\d.]+\s+max=\s*[\d.]+"
    r"\s+cold\(first10\)=\s*([\d.]+)\s+warm\(last10\)=\s*([\d.]+)"
)


def load_sample_timing(jid: str) -> dict[str, SampleTimingArm]:
    out_path = PROBE_DIR / jid / f"{jid}.out"
    arms = {}

    for match in _SAMPLE_RE.finditer(out_path.read_text()):
        arms[match.group(1)] = SampleTimingArm(
            median_ms=float(match.group(2)),
            mean_ms=float(match.group(3)),
            cold_ms=float(match.group(4)),
            warm_ms=float(match.group(5)),
        )

    return arms


@dataclass
class ThroughputArm:
    workers: int
    sps: float
    p50_ms: float
    p95_ms: float
    stalls: int


_TP_HEAD_RE = re.compile(r"(\w+)\s+workers=(\d+)\s+sps=\s*([\d.]+)")
_TP_BODY_RE = re.compile(
    r"per-batch ms: p50=\s*([\d.]+) p95=\s*([\d.]+) max=\s*[\d.]+\s+stalls\(>1s\)=(\d+)"
)


def load_throughput(jid: str) -> dict[str, ThroughputArm]:
    out_path = PROBE_DIR / jid / f"{jid}.out"
    lines = out_path.read_text().splitlines()
    arms = {}

    for index, line in enumerate(lines):
        head = _TP_HEAD_RE.search(line)

        if not head:
            continue

        body = _TP_BODY_RE.search(lines[index + 1]) if index + 1 < len(lines) else None
        arms[head.group(1)] = ThroughputArm(
            workers=int(head.group(2)),
            sps=float(head.group(3)),
            p50_ms=float(body.group(1)) if body else 0.0,
            p95_ms=float(body.group(2)) if body else 0.0,
            stalls=int(body.group(3)) if body else 0,
        )

    return arms
