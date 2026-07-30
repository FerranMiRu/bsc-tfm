"""Per-batch wall-clock figures for the full-window merged trainings (xarray vs raw), plus the
xarray store-size finding. All runs use the full merged store (1901-2015), 1 node / 4 ranks /
12 workers, train window 1960-2000, batch 8, accumulation 4, 10 epochs.

    xarray loader   = job 42401736 (dataset_type: merged)      2.34 h, 7.7% stalls
    raw-zarr loader = job 42441491 (dataset_type: merged_raw)   1.32 h, ~0% stalls

Per-batch deltas are read from ``unet_training_<jid>.log`` (the same source as ``common.py``): the
time between consecutive intra-epoch ``Batch [i/2500]`` INFO lines. The store-size numbers are the
single-thread per-sample medians from the store-size probe (single/merged small vs big) and the
clean raw probe (``merged_big_raw``).

Run from the ``experiments`` directory (so the ``analysis`` package is importable):

    uv run --with matplotlib --with numpy python -m analysis.plot_fulltrain

Figures are written to ``<repo>/presentation-figures/``.
"""

from __future__ import annotations

import re
from datetime import datetime
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import BATCH_SIZE, REPO, TRAINING_DIR

OUT_DIR = REPO.parent / "presentation-figures"

XARR_JID = "42401736"  # merged + xarray, 4r12w
RAW_JID = "42441491"  # merged_raw,     4r12w

C_XARR = "#ff7f0e"  # orange = xarray path
C_RAW = "#2ca02c"  # green  = raw path
C_SINGLE = "#1f77b4"  # blue   = single-zarr (store-size chart)
FLOOR_S = 0.153  # GPU compute floor, s/batch (batch=8)

EPOCH, B0, B1 = 5, 100, 340  # 240-batch mid-epoch window (20x the 12-worker refill cadence)

_BATCH_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Epoch \[(\d+)/\d+\], Batch \[(\d+)/"
)


def batch_records(jid: str) -> list[tuple[int, int, float]]:
    """Return (epoch, batch, delta_s) for each intra-epoch batch of a training job."""
    log_path = TRAINING_DIR / jid / f"unet_training_{jid}.log"
    stamps = []

    for line in log_path.read_text().splitlines():
        match = _BATCH_RE.search(line)

        if match:
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            stamps.append((int(match.group(2)), int(match.group(3)), stamp))

    records = []

    for previous, current in pairwise(stamps):
        if previous[0] == current[0]:  # same epoch: skip the epoch-boundary gap
            records.append((current[0], current[1], (current[2] - previous[2]).total_seconds()))

    return records


def window(records: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    chosen = sorted((b, d) for (e, b, d) in records if e == EPOCH and B0 <= b < B1)
    return np.array([b for b, _ in chosen]), np.array([d for _, d in chosen])


def describe(name: str, records: list[tuple[int, int, float]]) -> None:
    v = np.array([d for _, _, d in records])
    print(
        f"{name:8s} n={len(v)} p50={np.median(v) * 1000:.0f}ms mean={v.mean() * 1000:.0f}ms "
        f"p95={np.percentile(v, 95) * 1000:.0f}ms max={v.max() * 1000:.0f}ms "
        f"stalls>1s={100 * np.mean(v > 1):.1f}% sps/gpu={BATCH_SIZE / np.median(v):.1f}"
    )


def perbatch_panel(ax, batches, deltas, color, title, ylim) -> None:
    ax.bar(batches, deltas, width=1.0, color=color)
    ax.axhline(FLOOR_S, ls="--", lw=0.8, color="0.4")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(batches.min() - 1, batches.max() + 1)
    ax.set_ylabel("wall time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, loc="left", fontsize=11)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    xarr = batch_records(XARR_JID)
    raw = batch_records(RAW_JID)
    describe("xarray", xarr)
    describe("raw", raw)

    bx, vx = window(xarr)
    br, vr = window(raw)
    ylim = (0.08, max(vx.max(), vr.max()) * 1.5)

    # single-panel: xarray
    fig, ax = plt.subplots(figsize=(10, 3.0))
    perbatch_panel(
        ax,
        bx,
        vx,
        C_XARR,
        "Merged store, full time range (xarray loader): 2.34 h, 7.7% stalls",
        ylim,
    )
    ax.set_xlabel("training batch index")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fulltrain_xarray_perbatch.png", dpi=200)
    plt.close(fig)

    # single-panel: raw
    fig, ax = plt.subplots(figsize=(10, 3.0))
    perbatch_panel(
        ax,
        br,
        vr,
        C_RAW,
        "Merged store, full time range (raw-zarr loader): 1.32 h, ~0% stalls",
        ylim,
    )
    ax.set_xlabel("training batch index")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fulltrain_raw_perbatch.png", dpi=200)
    plt.close(fig)

    # combined two-panel
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, sharey=True)
    perbatch_panel(a1, bx, vx, C_XARR, "xarray loader   (2.34 h, p95 1306 ms, 7.7% stalls)", ylim)
    perbatch_panel(a2, br, vr, C_RAW, "raw-zarr loader   (1.32 h, p95 <220 ms, ~0% stalls)", ylim)
    a2.set_xlabel("training batch index")
    fig.suptitle("Full time range, merged store: removing xarray removes the stalls", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fulltrain_xarray_vs_raw_perbatch.png", dpi=200)
    plt.close(fig)

    # store-size / xarray-graph finding (single-thread median ms/sample)
    single = [1329, 5513]  # single-zarr, 66 GB / 435 GB
    merged = [174, 286]  # merged,      59 GB / 387 GB
    raw_big = 37.9  # merged_big_raw, size-independent (37-38 ms across 4 probes)
    xlab = ["small store\n(~60 GB, 16 time-chunks)", "big store\n(~400 GB, 115 time-chunks)"]
    x = np.arange(2)
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar(x - w / 2, single, w, color=C_SINGLE, label="single-zarr (xarray)")
    b2 = ax.bar(x + w / 2, merged, w, color=C_XARR, label="merged (xarray)")
    rawline = ax.axhline(
        raw_big, ls="--", lw=1.3, color=C_RAW, label="merged + raw loader (~38 ms)"
    )
    ax.set_yscale("log")
    ax.set_ylim(20, 9000)
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_ylabel("per-sample fetch, single thread (ms)")
    ax.set_title(
        "xarray per-sample cost grows with store size; raw loader is flat", loc="left", fontsize=11
    )
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.0f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.annotate("x4.15", (0.5, 2600), ha="center", color=C_SINGLE, fontsize=9, fontweight="bold")
    ax.annotate("x1.64", (0.5, 210), ha="center", color=C_XARR, fontsize=9, fontweight="bold")
    ax.legend(handles=[b1, b2, rawline], loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "xarray_store_size.png", dpi=200)
    plt.close(fig)

    print("\nwrote:")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(" ", path.relative_to(REPO.parent), f"({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
