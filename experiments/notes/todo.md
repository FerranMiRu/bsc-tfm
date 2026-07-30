# Long-term lever backlog

Updated 2026-06-14. Active-lever statuses live in `knowledge.md`
"Active levers" table — this file is the implementation backlog, ranked
by expected ROI.

For the immediate next-session handoff, see `next_session.md`.

## 1. Lever 1 — bypass xarray + dask in `MergedZarrDataset`

**Predicted ~4× per-sample on top of Lever 3** (~176 → ~35–45 ms).
Run 25 breakdown (job 41635364): dynamic 90 ms + static 60 ms median =
150 ms of xarray+dask compute work in `__getitem__`.

Plan:

- Replace each `.isel(...).to_numpy()` in `_read_dynamic_to_numpy` and
  `_process_static` with direct numpy slicing on the underlying zarr
  arrays via `zarr.open_array`.
- The raw zarr API releases the GIL during Blosc decode, so per-timestep
  work is threadable.
- Must hit **both** `_read_dynamic` (51% of budget) and `_process_static`
  (34% of budget). Skipping static throws away a third of the predicted
  gain.

Validation:

- Phase-1 single-thread `__getitem__` measurement on 100 samples (same
  framework as Run 24 + 25). Median should land at 35–45 ms.
- End-to-end DDP training comparison vs current merged at NW=12, 4r and
  8r (mirror Run 33 methodology). Determines whether the per-sample
  gain survives the all-reduce barrier.

Risk:

- If merged is already at 95% DDP efficiency and limited by raw GPFS
  bandwidth (waiting on Run 33 to confirm), the 4× per-sample gain
  could collapse to <2× end-to-end. Run a single-rank bypass vs
  baseline measurement first to gate the implementation cost.

## 2. Lever 4 — zarr v3 + `async.concurrency`

**Smaller gain than Lever 1; lower priority.** Parallelises per-chunk
metadata RPCs but does NOT remove the per-chunk xarray graph cost
(16 ms × ~6.75 = 108 ms) that dominates merged. Predicted ~1.2–1.4× on
merged Phase 1.

Code change is small:

- Bump `zarr` pin: `>=2.18.3,<3.0.0` → `>=3,<4`. Run `uv sync`.
- Switch codec imports: `from numcodecs import Blosc` (not from zarr).
- Drop `synchronizer=ProcessSynchronizer(...)` — not ported to v3. (Has
  0 effect anyway per Run 1.)
- Optionally `zarr.config.set({"async.concurrency": 32 or 64})` in
  dataset init.

Existing v2 stores remain readable by v3 — no recreation required.

Stack this on Lever 1 if it leaves residual I/O headroom; not a
substitute. Risk: transitive `zarr<3` pins in deps may surface during
`uv sync`.

## 3. Chunk-straddling reduction (deferred)

**Per-sample chunk count for merged is ~6.75 file opens** (Run 25
analysis): 2 dynamic timesteps × 2.25 + 1 static × 2.25.

Getting to the "3 files per sample" ideal would require constraining
sampling to chunk-aligned positions (changes training distribution).
Saves ~85 ms cold I/O, <20% of merged's 200 ms budget. Defer unless we
hit a wall elsewhere.

Alternative chunk sizes (1024², 2048²) all net negative — see
`knowledge.md` "Disproven / dead ends" → chunk-size analysis.

## Methodology improvements

- **Adopt `sps_wall` (samples/s from epoch wall time) as the canonical
  throughput metric** for all training-loop benchmarks. Median batch
  delta undercounts the tail; `sps_med` overstates by up to 41% at
  NW=8 (Run 31 + Run 33 verification). See `next_session.md` analysis
  script for the reference implementation.
- **Promote that snippet to a `scripts/analyze_training_runs.py`**
  once Run 33 is done and the formula is validated. Replaces ad-hoc
  re-implementations across the next levers.

## Items that are closed (do not re-open)

These were on earlier todo lists and are conclusively resolved.
Kept here briefly so future sessions don't re-investigate:

- **~100 ms DataLoader main-process overhead** — was a cold-cache
  artefact (Run 24); Run 26 showed worker-effective ≈ Phase-1
  single-thread, no structural overhead.
- **30 ms in-memory floor** — measured directly in Run 25 as
  `_build_continuous_sequence` ≈ 15 ms. No further investigation
  needed.
- **`profiling-singlezarr.yaml` rerun** — Phase 1 single mean 1411 ms
  matched Run-4 baseline 1067 ms within noise.
- **`ProcessSynchronizer` effect** — 0 effect on perf (Run 1, 2), 0
  effect on `sem_wait` traces (Run 7). Toggle removed from codebase.
- **DDP-aware worker sweep design validation** — Runs 29 + 30 + 31 +
  33 collectively settled this. The sweep methodology was confounded
  by per-rank variance and cross-store GPFS contention; real-training
  comparison (Run 33) is the authoritative measurement going forward.
