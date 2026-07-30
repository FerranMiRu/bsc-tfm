# Dataloader profiling — results

Running notebook of what we've measured, what we've concluded, and what we
still need to investigate. Each run gets a section appended as new data lands.

Section template:

```
## Run X — <one-line summary>

### Jobs
- <jobid> — <differentiating factor>

### Interesting values
...

### Quick conclusions

Short run-scoped takeaways only. Durable facts, hypotheses, and open
questions belong in `knowledge.md`.
```

---

## Historical training-loop runs (weeks 15–16, pre-standalone-profiler)

Eleven "BATCH" runs (numbering from the weekly journal) submitted before the standalone dataloader profiler existed. Each was the **full training loop** via `acc_training.sh` (model forward/backward + dataloader), originally 15 batches/epoch, later 50. Per-batch timings live in `batch_times.txt`. These runs are summarised together because they share the same experimental setup and progressively converge on the same finding.

### Jobs

| BATCH | Singlezarr | Preprocessed | Multizarr | Other                       | Note                                                                |
| ----- | ---------- | ------------ | --------- | --------------------------- | ------------------------------------------------------------------- |
| 1     | 41088388   | 41088390     | 41088508  | —                           | First runs; no per-batch logs captured                              |
| 2     | 41093449   | 41093451     | 41093456  | —                           | First clean per-batch log; cycle visible                            |
| 3     | 41094289   | 41094291     | 41094292  | —                           | Loss confirmed; pattern persists                                    |
| 4     | 41162051   | 41162052     | 41162053  | —                           | Re-confirm                                                          |
| 5     | 41164024   | 41164025     | 41164026  | —                           | Re-confirm                                                          |
| 6     | 41164619   | 41164620     | 41164621  | —                           | Stopped at batch 10 (mistake)                                       |
| 7     | 41165374   | 41165375     | 41165376  | —                           | Profiling window 6–10 → spike moves to batch 11 (profiler artefact) |
| 8     | 41166558   | 41166559     | 41166561  | —                           | Profiling off; batch-12 cycle reappears                             |
| 9     | 41171635   | 41171636     | 41171637  | 41172665 (preprocessed-big) | 50 batches/epoch; periodic cycle confirmed                          |
| 10    | —          | 41199507     | —         | 41199508 (preprocessed-big) | Store-size effect (x1.9 in spike)                                   |
| 11    | —          | 41201236     | —         | 41201237 (preprocessed-big) | Re-confirm size effect (x3 in spike, 41 s → 97 s)                   |

### Interesting values

Steady-state pattern (BATCHes 2–8, 15 batches each, `num_workers=12`):

|                          |        singlezarr |      preprocessed |         multizarr |
| ------------------------ | ----------------: | ----------------: | ----------------: |
| Typical fast batch       |            0.15 s |            0.15 s |            0.15 s |
| Spike batch (≈ batch 12) |            7–14 s |            7–17 s |           66–92 s |
| Cadence                  | every ~12 batches | every ~12 batches | every ~12 batches |

50-batch confirmation (BATCH 9, periodic cycle visible):

- **singlezarr** (41171635): spikes at batches 12 (9.0 s), 24 (8.5 s), 26 (6.2 s), 36 (12.7 s), 38 (6.9 s), 48 (3.1 s), 50 (8.1 s)
- **preprocessed** (41171636): spikes at batches 12 (11.0 s), 24 (7.7 s), 36 (16.5 s), 48 (20.7 s)
- **multizarr** (41171637): spikes 8–12× larger, at batches 12 (82.7 s), 20 (12.5 s), 24 (77.4 s), 32 (22.5 s), 36 (61.4 s), 44 (22.6 s), 48 (54.3 s)
- **preprocessed-big** (41172665): spikes at batches 12 (30.7 s), 16 (8.5 s), 24 (25.7 s), 28 (5.9 s), 36 (23.7 s), 40 (6.1 s), 48 (23.7 s)

Store-size effect (BATCHes 10–11, side-by-side preprocessed vs preprocessed-big):

|                                 | preprocessed (64 G) |  preprocessed-big (213 G) |
| ------------------------------- | ------------------: | ------------------------: |
| Mean spike magnitude (BATCH 10) |               8.0 s | 28.0 s (≈ x1.9 over base) |
| Mean spike magnitude (BATCH 11) |              10.3 s |             31.1 s (≈ x3) |

### Quick conclusions

- **Batch-12 cycle is periodic, not transient.** Confirmed by 50-batch runs (BATCH 9). Spike recurs every `num_workers` (12) batches — matches the prefetch-rotation hypothesis.
- **Multizarr spikes are ~8× larger** than singlezarr/preprocessed, but the cadence is identical.
- **Store size affects spike magnitude.** Preprocessed-big (213 G) has spikes 1.9–3× longer than preprocessed (64 G) under identical chunk layout. Mechanism still unexplained — see `knowledge.md`.
- **Bake-in preprocessing was _not_ a win in these runs either** — singlezarr and preprocessed show indistinguishable cycle behaviour. This survived re-measurement with the standalone profiler (Run 6).
- These runs predated the standalone profiler so they conflate model compute + dataloader spikes. Subsequent runs (Run 2 onwards) isolated the dataloader.

---

## Run 1 — Synchronizer toggle on full training

### Jobs

- `41318578` — `profiling-singlezarr`, `use_synchronizer=true`
- `41318658` — `profiling-singlezarr-no-synch`, `use_synchronizer=false`

### Interesting values

- Both runs trained 50 batches/epoch and produced **identical batch-time spike
  patterns**: ~10s stalls every 12 batches, with sub-second batches in between.
- Stall cadence matches `num_workers=12 × batch_size=8 = 96 samples ≈ 12
training batches per prefetch rotation`.

### Quick conclusions

- The Zarr `ProcessSynchronizer` toggle is **not** the bottleneck — same
  pattern, same magnitude with and without it.
- Toggle is kept as a config knob (`data.use_synchronizer`) for future
  experiments but won't be revisited as a perf lever.

---

## Run 2 — Standalone dataloader profiler v1

### Jobs

- `41321400` — `profiling-singlezarr`, `use_synchronizer=true`
- `41321507` — `profiling-singlezarr-no-synch`, `use_synchronizer=false`

Both ran the same two-phase profile (50 single-thread `__getitem__` then a
single DataLoader for 50 batches) with the same RNG seed.

### Interesting values

|                            | 41321400 (synch)       | 41321507 (no synch)    |
| -------------------------- | ---------------------- | ---------------------- |
| getitem mean               | 1177 ms                | 1178 ms                |
| getitem total (50 samples) | 58.85 s                | 58.88 s                |
| sample size                | 9.50 MB                | 9.50 MB                |
| Phase 2                    | hung 28 min, 0 batches | hung 28 min, 0 batches |

Single-thread effective throughput: **9.50 MB / 1.18 s ≈ 8 MB/s per worker**
— nowhere near what GPFS should provide.

### Quick conclusions

- Synchronizer truly has zero effect, even on cold-start single-thread runs.
- Phase 2 hung because the DataLoader was instantiated in the same Python
  process as the Hydra-initialized parent — fork-after-init issue. Will be
  fixed in v2 by running each loader in a `mp.spawn` child.
- Why are we capped at ~8 MB/s per single-thread reader on GPFS? Open.

---

## Run 3 — Profiler v2 (per-modality + sweep in subprocesses)

### Jobs

- `41384723` — `profiling-singlezarr`, `use_synchronizer=true`, 50 getitem
  samples + sweep `[0, 1, 4, 12]` × 10 batches each

### Interesting values

Phase 1 — per-modality breakdown (mean per call, 50 samples):

| Method                 | Calls |         Mean |   Median |       Max | % of total |
| ---------------------- | ----: | -----------: | -------: | --------: | ---------: |
| `_get_luh2_data`       |   100 | **427.9 ms** | 392.9 ms | 2991.0 ms |  **68.0%** |
| `_get_static_data`     |    50 |     105.1 ms |  98.9 ms |  313.9 ms |       8.4% |
| `_get_population_data` |   100 |      55.3 ms |  54.3 ms |  123.8 ms |       8.8% |
| `_get_hilda_target`    |    50 |      48.9 ms |  47.9 ms |   70.4 ms |       3.9% |
| `_get_hilda_prior`     |    50 |      47.8 ms |  47.5 ms |   72.3 ms |       3.8% |
| `_get_kg_data`         |   100 |      43.3 ms |  43.1 ms |   80.0 ms |       6.9% |

Phase 2 — `num_workers` sweep (10 batches per config, batch_size=8):

| num_workers |    Cold | Mean warm | Median warm |         Throughput |
| ----------: | ------: | --------: | ----------: | -----------------: |
|           0 | 11.04 s |   9660 ms |     9572 ms |      0.8 samples/s |
|           1 | 10.71 s |   9472 ms |     9545 ms |      0.8 samples/s |
|           4 | 11.00 s |   2233 ms |       47 ms |      3.6 samples/s |
|          12 | 11.71 s | **81 ms** |  **0.1 ms** | **98.7 samples/s** |

### Quick conclusions

- LUH2 is **two thirds** of per-sample time (~430 ms for ~7.5 MB of payload).
  It's also the biggest tensor (79% of sample bytes), so cost-per-byte is
  roughly even across modalities — but the absolute LUH2 time per call is the
  headline target.
- DataLoader is **not** wedged — running each loader in a clean spawn child
  resolves Run 2's hang.
- 100 calls per LUH2/pop/kg vs 50 for static/hilda is the expected pattern (2
  rollout steps in `timesteps_target`).
- **Q1 — Load vs preprocess.** Need to split each `_get_*_data` into raw zarr
  read vs in-memory processing (fillna/normalize/stack/transpose/cast). Done
  in v3 of the profiler via `TimedSingleZarrDataset`.
- **Q2 — Size dependence.** Different stores chunked identically
  (`time:1, latitude:512, longitude:512`, 256×256 patches) show very different
  per-sample times. Read amount should be store-size independent. Next: run
  the same profile against `profiling-preprocessed`,
  `profiling-preprocessed-big`, and `profiling-multizarr`.
- **Q3 — `num_workers=12` apparent throughput.** Expected ~10 samples/s
  (12 / 1.2), measured 98.7 samples/s. Likely the 10 batches we consumed are
  all queue drains (queue depth = 12 × 4 = 48 batches). Bump `num_batches` to
  80+ to drain past the queue and see steady state.

---

## Run 4 — Profiler v3 (load vs preprocess split + 80-batch sweep)

### Jobs

- `41385929` — `profiling-singlezarr`, `use_synchronizer=true`, 50 getitem
  samples in both phase 1 (merged) and phase 2 (load|prep split), sweep
  `[0, 1, 4, 12]` × 80 batches each (≥ prefetch queue depth so steady state
  surfaces)

### Interesting values

Phase 2 — load vs preprocess split (50 samples, 100/50 calls each):

| Modality       | Calls |    Load mean |  Load total (%) | Prep mean | Prep total (%) |
| -------------- | ----: | -----------: | --------------: | --------: | -------------: |
| `luh2`         |   100 | **314.4 ms** | 31.44 s (58.9%) |   33.4 ms |  3.34 s (6.3%) |
| `population`   |   100 |      52.2 ms |   5.22 s (9.8%) |    0.1 ms |  0.01 s (0.0%) |
| `kg`           |   100 |      42.2 ms |   4.22 s (7.9%) |    0.0 ms |  0.00 s (0.0%) |
| `static`       |    50 |      62.2 ms |   3.11 s (5.8%) |   19.0 ms |  0.95 s (1.8%) |
| `hilda_prior`  |    50 |      48.8 ms |   2.44 s (4.6%) |    0.2 ms |  0.01 s (0.0%) |
| `hilda_target` |    50 |      48.2 ms |   2.41 s (4.5%) |    0.4 ms |  0.02 s (0.0%) |

Totals: load 48.84 s (91.6% of getitem), prep 4.33 s (8.1%). Phase 2 getitem
mean was 1066.8 ms (Phase 1 merged was 1234 ms; ~14% delta likely from RNG
variance — Phase 2's `rng.integers` call advanced the same generator).

Phase 3 — `num_workers` sweep, 80 batches per config:

| num_workers |    Cold |                  Mean warm | Median warm |        Throughput |
| ----------: | ------: | -------------------------: | ----------: | ----------------: |
|           0 |       — |                    9548 ms |     9510 ms | **0.8 samples/s** |
|           1 |       — |                    9325 ms |     9266 ms | **0.9 samples/s** |
|           4 |       — |                    2428 ms |     1197 ms | **3.3 samples/s** |
|          12 | 11.47 s | **832 ms** (median 0.1 ms) |      0.1 ms | **9.6 samples/s** |

Per-batch trace for w=12 shows a clear cycle: 12 cached fast batches → 1 slow
batch (~9.5s) → 1 medium (~1.6s) → 12 cached → … exactly what 12 workers each
producing one batch per ~9.5s would yield (12 batches per 9.5s = 1.26
batches/s = 10 samples/s).

### Quick conclusions

- **Q1 answered — load, not preprocess.** Raw zarr reads are ~92% of getitem
  cost. In-memory work (fillna, normalize, stack, transpose, cast, remap,
  clip) is ~8%. Optimization effort should go into the zarr read path
  (chunk shape, codec, store layout, GPFS read patterns) — preprocessing
  optimization buys at most 8%.
- **Q3 answered — DataLoader scales as expected.** Steady-state throughput at
  ~0.8 samples/s/worker. 12 workers gives 9.6 samples/s, matching the model
  `num_workers / per-sample-time`. The 98.7 samples/s in Run 3 was pure
  prefetch-queue drain (10 batches vs 48-batch queue depth) — not real
  throughput.
- **Q2 still open** — need the same profile on `profiling-preprocessed`,
  `profiling-preprocessed-big`, `profiling-multizarr` to verify whether the
  store-size dependence holds. With load = 92% of the cost, this is now
  specifically about why a single 256×256 read from a single chunk
  (`time:1, latitude:512, longitude:512`) should depend on overall store
  size.

---

## Run 5 — Multizarr profile (one zarr store per modality)

### Jobs

- `41407784` — `profiling-multizarr`, `use_synchronizer=true`, 50 getitem +
  sweep `[0, 1, 4, 12]` × 80 batches

### Interesting values

Phase 1 — per-modality breakdown (multizarr):

| Method                 | Calls |          Mean |    Median |       Max | % of total |
| ---------------------- | ----: | ------------: | --------: | --------: | ---------: |
| `_get_luh2_data`       |   100 | **3297.3 ms** | 3253.9 ms | 6188.2 ms |  **82.4%** |
| `_get_population_data` |   100 |      291.4 ms |  285.7 ms |  613.1 ms |       7.3% |
| `_get_hilda_prior`     |    50 |      190.4 ms |  192.0 ms |  227.4 ms |       2.4% |
| `_get_hilda_target`    |    50 |      186.6 ms |  184.5 ms |  234.4 ms |       2.3% |
| `_get_kg_data`         |   100 |      185.6 ms |  185.3 ms |  213.9 ms |       4.6% |
| `_get_static_data`     |    50 |       73.4 ms |   72.2 ms |  110.8 ms |       0.9% |

Per-sample budget: `2 * (3297 + 291 + 186) + (73 + 190 + 187) ≈ 8.0 s`.

Phase 2 — skipped (timed subclass only implemented for `SingleZarrDataset`).

Phase 3 — every config (`[0, 1, 4, 12]`) **timed out at batch 1** because a
single batch_size=8 in single-thread takes ~64 s, far over the 30 s
per-batch timeout.

### Quick conclusions

- **Q2 partial answer — store layout dominates store contents.** Same chunks,
  same patches, but five separate zarr stores are **~8× slower** than the
  bundled single-zarr equivalent across every modality (except static,
  which is comparable or marginally faster — only difference is the
  modality is static so no time dim).
- The multizarr stalls in earlier `batch_times.txt` runs (70-90 s spikes)
  are exactly what an 8 s/sample × 8 batch_size load looks like under a
  partially saturated worker pool.
- Open: **what specifically makes multizarr slow?** Candidates: per-store
  open/index overhead, GPFS handle/directory cost, codec or chunk-shape
  differences between the per-modality stores (need to inspect each
  store's `encoding["chunks"]` and `compressor`), or coordinate alignment
  cost when joining across stores.
- Next sweep on multizarr should bump `batch_timeout_s` to ≥120 to get
  steady-state throughput numbers.

---

## Run 6 — Preprocessed profile (same singlezarr layout, fillna/norm baked in)

### Jobs

- `41409692` — `profiling-preprocessed`, `use_synchronizer=true`, 50 getitem +
  sweep `[0, 1, 4, 12]` × 80 batches

### Interesting values

Phase 1 — per-modality breakdown (preprocessed):

| Method                 | Calls |     Mean |   Median |       Max | % of total |
| ---------------------- | ----: | -------: | -------: | --------: | ---------: |
| `_get_luh2_data`       |   100 | 412.1 ms | 381.5 ms | 2702.8 ms |      68.8% |
| `_get_static_data`     |    50 |  92.0 ms |  92.5 ms |  140.7 ms |       7.7% |
| `_get_population_data` |   100 |  50.9 ms |  49.2 ms |  135.5 ms |       8.5% |
| `_get_hilda_target`    |    50 |  51.2 ms |  46.5 ms |  154.3 ms |       4.3% |
| `_get_hilda_prior`     |    50 |  44.0 ms |  43.7 ms |   72.3 ms |       3.7% |
| `_get_kg_data`         |   100 |  41.0 ms |  41.0 ms |   60.6 ms |       6.8% |

Phase 2 — load vs preprocess split:

| Modality       | Load mean |  Load total (%) | Prep mean | Prep total (%) |
| -------------- | --------: | --------------: | --------: | -------------: |
| `luh2`         |  313.8 ms | 31.38 s (61.2%) |   33.7 ms |  3.37 s (6.6%) |
| `population`   |   46.5 ms |   4.65 s (9.1%) |    0.1 ms |  0.01 s (0.0%) |
| `kg`           |   39.8 ms |   3.98 s (7.8%) |    0.0 ms |  0.00 s (0.0%) |
| `static`       |   48.6 ms |   2.43 s (4.7%) |   19.2 ms |  0.96 s (1.9%) |
| `hilda_prior`  |   41.6 ms |   2.08 s (4.1%) |    0.2 ms |  0.01 s (0.0%) |
| `hilda_target` |   44.0 ms |   2.20 s (4.3%) |    0.4 ms |  0.02 s (0.0%) |

Phase 3 — `num_workers` sweep (80 batches each):

| num_workers |    Cold | Mean warm | Median warm |        Throughput |
| ----------: | ------: | --------: | ----------: | ----------------: |
|           0 | 10.19 s |   9270 ms |     9215 ms |     0.9 samples/s |
|           1 | 10.15 s |   8912 ms |     8850 ms |     0.9 samples/s |
|           4 | 10.57 s |   2317 ms |      0.2 ms |     3.5 samples/s |
|          12 | 11.09 s |    834 ms |      0.1 ms | **9.6 samples/s** |

### Quick conclusions

- **Q2 fully answered — preprocessed bake-in does ~nothing for the bundled
  layout.** Side-by-side vs Run 4 (`profiling-singlezarr`, non-preprocessed):

  |                      | Run 4 (singlezarr) | Run 6 (preprocessed) |
  | -------------------- | -----------------: | -------------------: |
  | getitem Phase 1 mean |            1234 ms |              1199 ms |
  | getitem Phase 2 mean |            1067 ms |              1026 ms |
  | LUH2 load mean       |           314.4 ms |             313.8 ms |
  | LUH2 prep mean       |            33.4 ms |              33.7 ms |
  | w=12 throughput      |      9.6 samples/s |        9.6 samples/s |

  Same time, same throughput, same load/prep split. The preprocessed store
  saves ~0 because (a) prep was already only ~8% of cost (Run 4's Q1
  conclusion), and (b) the LUH2 "prep" cost is the **to_stacked_array +
  transpose + reshape**, not fillna/normalize — bake-in doesn't remove
  those. The codec/chunk layout of the LUH2 array is what's expensive.

- **Two singlezarr stores of comparable size (Runs 4 and 6) are
  indistinguishable.** This does NOT contradict the earlier observation
  that per-sample time scales with underlying store size — that was
  established with stores of meaningfully different sizes, and is still
  open.
- **Updated open question — Q2': what about layout matters?**
  - LUH2 owns 60-68% of getitem time and is dominated by raw read.
  - Hypotheses to test (in this order):
    1. **Chunk shape.** If chunks are `time:1, lat:512, lon:512` we read
       4× more bytes than needed per 256² patch. Re-chunking to
       `time:1, lat:256, lon:256` (or pre-stacking vars into a single
       data variable with chunks `(vars, time, lat, lon)` chunked
       `(14, 1, 256, 256)`) would cut amplification.
    2. **Per-variable fetch cost.** 14 separate `.isel` calls × 14
       separate chunk reads × 14 separate decompressions, all serial.
       Stacking into 1 variable cuts to 1 chunk per read.
    3. **Codec.** zstd vs blosc-lz4 trade speed for ratio; current codec
       unknown. Worth `xr.open_zarr(store).c3ann.encoding` to check.

---

## Run 7 — T0: nsys trace of current singlezarr dataloader (sem_wait verification)

Re-profile the _current_ singlezarr setup (synchronizer ON, w=12) under nsys to see what
DataLoader worker threads are actually doing during steady-state. Goal: settle whether the
historical "sem_wait everywhere" pattern is (a) idle queue wait, (b) GPFS daemon futex /
Blosc internal sync inside `__getitem__`, or (c) something else.

### Jobs

- `41436043`, `41436050` — cancelled before completion. Both were launched concurrently against
  the same store; GPFS contention would have distorted the timings. Rule now codified in
  `AGENTS.md`.
- `41436408` — `profiling-singlezarr`, `use_synchronizer=true`, `workers_sweep=[12]`,
  `num_batches=20`, `num_getitem=5`, nsys osrt trace via
  `scripts/launch_profile_dataloader_nsys.sh`. Synch-off counterpart will be submitted only
  after this one finishes.
- `41452882` — `profiling-singlezarr-no-synch`, `use_synchronizer=false`, same sweep/batch
  settings as 41436408. nsys osrt trace.

### Interesting values

**Job 41436408 (synchronizer on)**

_Phase 1 — getitem + per-modality merged (5 samples)_

```
--- getitem | n=5 | mean= 2703.9 ms | median= 2043.8 ms | p95= 5250.4 ms | max= 5671.9 ms | sum=  13.52 s
--- per-modality (load + preprocess merged) ---
  _get_luh2_data        : calls=  10 | mean= 1015.5 ms | median=  409.6 ms | max= 3353.3 ms | total= 10.15 s (75.1%)
  _get_population_data  : calls=  10 | mean=  198.0 ms | median=   50.8 ms | max= 1521.7 ms | total=  1.98 s (14.6%)
  _get_kg_data          : calls=  10 | mean=   44.1 ms | median=   41.7 ms | max=   69.1 ms | total=  0.44 s ( 3.3%)
  _get_static_data      : calls=   5 | mean=   95.5 ms | median=   95.8 ms | max=  107.4 ms | total=  0.48 s ( 3.5%)
  _get_hilda_prior      : calls=   5 | mean=   47.4 ms | median=   49.8 ms | max=   52.6 ms | total=  0.24 s ( 1.8%)
  _get_hilda_target     : calls=   5 | mean=   42.7 ms | median=   43.2 ms | max=   49.5 ms | total=  0.21 s ( 1.6%)
```

_Phase 2 — getitem + load vs preprocess split (5 samples)_

```
--- getitem | n=5 | mean= 4282.1 ms | median= 1126.1 ms | p95=10930.2 ms | max=12100.7 ms | sum=  21.41 s
--- per-modality (load | preprocess split) ---
  luh2          : calls=  10 | load mean= 1937.4 ms total= 19.37 s (90.5%) | prep mean=   33.5 ms total=  0.33 s ( 1.6%)
  population    : calls=  10 | load mean=   44.8 ms total=  0.45 s ( 2.1%) | prep mean=    0.1 ms total=  0.00 s ( 0.0%)
  kg            : calls=  10 | load mean=   36.1 ms total=  0.36 s ( 1.7%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=   5 | load mean=   47.5 ms total=  0.24 s ( 1.1%) | prep mean=   19.4 ms total=  0.10 s ( 0.5%)
  hilda_prior   : calls=   5 | load mean=   60.4 ms total=  0.30 s ( 1.4%) | prep mean=    0.2 ms total=  0.00 s ( 0.0%)
  hilda_target  : calls=   5 | load mean=   46.8 ms total=  0.23 s ( 1.1%) | prep mean=    0.5 ms total=  0.00 s ( 0.0%)
```

_Phase 3 — num_workers=12, 20 batches_

```
cold batch: 21887.9 ms (includes worker startup)
--- warm batches | n=19 | mean= 3957.5 ms | median=    0.2 ms | p95=28702.1 ms | max=48712.0 ms | sum=  75.19 s
--- throughput |    2.0 samples/s |   0.25 batches/s

  batch   1/20   21887.9 ms   76.00 MB
  batch   2/20       0.2 ms   76.00 MB
  batch   3/20       0.1 ms   76.00 MB
  batch   4/20   48712.0 ms   76.00 MB
  batch   5/20       0.1 ms   76.00 MB
  batch   6/20       0.2 ms   76.00 MB
  batch   7/20       0.2 ms   76.00 MB
  batch   8/20       0.0 ms   76.00 MB
  batch   9/20       0.2 ms   76.00 MB
  batch  10/20       0.2 ms   76.00 MB
  batch  11/20       0.0 ms   76.00 MB
  batch  12/20       0.0 ms   76.00 MB
  batch  13/20       0.2 ms   76.00 MB
  batch  14/20       0.2 ms   76.00 MB
  batch  15/20       0.0 ms   76.00 MB
  batch  16/20   26478.8 ms   76.00 MB
  batch  17/20       0.2 ms   76.00 MB
  batch  18/20       0.0 ms   76.00 MB
  batch  19/20       0.0 ms   76.00 MB
  batch  20/20       0.2 ms   76.00 MB
```

_nsys osrt summary (top 10)_

```
 Time (%)  Total Time (ns)  Num Calls     Avg (ns)        Med (ns)       Min (ns)      Max (ns)     StdDev (ns)            Name
 --------  ---------------  ---------  --------------  --------------  ------------  ------------  -------------  ----------------------
     89.2   32674553873551     249269     131081497.8       1794396.0          5534  164012944941   1754153972.7  sem_wait
      4.9    1794772149267     173428      10348802.7          6948.0          1000   99789067203    314307032.0  read
      2.0     725249474563        220    3296588520.7          9325.0          2215   49431404240   8576264916.8  accept4
      1.5     533555087564       1415     377070733.3     100120634.0          1063  163576124414   4414561292.6  poll
      0.7     242800973131          1  242800973131.0  242800973131.0  242800973131  242800973131            0.0  pthread_join
      0.7     242588066716          9   26954229635.1         36658.0         13420  242562428514  80853074958.6  epoll_wait
      0.6     228919779468     638256        358664.5        211459.0          1000    1210250321      4657521.6  pthread_cond_timedwait
      0.3      97072476017         60    1617874600.3    1404077751.0          9518    5000062048   1567230204.3  sem_timedwait
      0.1      38238053293     180411        211949.7          7502.0          1000    1209463660     14527886.0  pthread_mutex_lock
      0.1      23136941712       2080      11123529.7          2474.5          1001    2084806267    146078209.3  write
```

**Job 41452882 (synchronizer off)**

Note: profiler output was emitted to stderr (not stdout); facts extracted from `41452882.err`.

_Phase 1 — getitem + per-modality merged (5 samples)_

```
--- getitem | n=5 | mean=16500.0 ms | median=12282.6 ms | p95=34621.8 ms | max=40168.5 ms | sum=  82.50 s
--- per-modality (load + preprocess merged) ---
  _get_luh2_data        : calls=  10 | mean= 5912.3 ms | median= 3238.3 ms | max=31364.6 ms | total= 59.12 s (71.7%)
  _get_population_data  : calls=  10 | mean=  192.9 ms | median=   74.5 ms | max= 1289.3 ms | total=  1.93 s ( 2.3%)
  _get_kg_data          : calls=  10 | mean=  512.7 ms | median=   56.1 ms | max= 2090.2 ms | total=  5.13 s ( 6.2%)
  _get_static_data      : calls=   5 | mean= 1659.8 ms | median= 1414.6 ms | max= 2257.6 ms | total=  8.30 s (10.1%)
  _get_hilda_prior      : calls=   5 | mean=  474.1 ms | median=   70.1 ms | max= 2118.9 ms | total=  2.37 s ( 2.9%)
  _get_hilda_target     : calls=   5 | mean= 1127.0 ms | median= 1454.1 ms | max= 2050.3 ms | total=  5.63 s ( 6.8%)
```

_Phase 2 — getitem + load vs preprocess split (5 samples)_

```
--- getitem | n=5 | mean= 7454.5 ms | median= 9574.8 ms | p95=10522.8 ms | max=10630.8 ms | sum=  37.27 s
--- per-modality (load | preprocess split) ---
  luh2          : calls=  10 | load mean= 1797.3 ms total= 17.97 s (48.2%) | prep mean=   33.2 ms total=  0.33 s ( 0.9%)
  population    : calls=  10 | load mean=  427.0 ms total=  4.27 s (11.5%) | prep mean=    0.1 ms total=  0.00 s ( 0.0%)
  kg            : calls=  10 | load mean=  213.1 ms total=  2.13 s ( 5.7%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=   5 | load mean= 1674.5 ms total=  8.37 s (22.5%) | prep mean=   18.9 ms total=  0.09 s ( 0.3%)
  hilda_prior   : calls=   5 | load mean=   48.1 ms total=  0.24 s ( 0.6%) | prep mean=    0.2 ms total=  0.00 s ( 0.0%)
  hilda_target  : calls=   5 | load mean=  767.3 ms total=  3.84 s (10.3%) | prep mean=    0.5 ms total=  0.00 s ( 0.0%)
```

_Phase 3 — num_workers=12, 20 batches_

```
num_workers = 12
batch 1 exceeded 60s per-batch timeout — aborted. Stopped at batch 1/20.
(No warm-batch stats or throughput line available.)
```

_nsys osrt summary (top 10)_

```
 Time (%)  Total Time (ns)  Num Calls     Avg (ns)        Med (ns)       Min (ns)      Max (ns)     StdDev (ns)             Name
 --------  ---------------  ---------  --------------  --------------  ------------  ------------  --------------  ----------------------
     82.9   26236864448184      70473     372296687.4       1094426.0         13030  144046007877    2703338564.9  sem_wait
     10.9    3437987960621     104497      32900350.8          4126.0          1005   80757891576     361339675.8  read
      2.3     737727400318          1  737727400318.0  737727400318.0  737727400318  737727400318             0.0  pthread_join
      2.3     713405402237          9   79267266915.2         34280.0         13700  713374054451  237790045521.8  epoll_wait
      1.2     368786219323         69    5344727816.3    5004685049.0          1350  138215571354   16371865039.0  poll
      0.2      59998237757         12    4999853146.4    5000060434.0    4997568702    5000067021        719421.0  sem_timedwait
      0.1      29668246257      63299        468700.1          8323.0          1066    3771787912      31175971.1  open64
      0.1      25670935071     120952        212240.7         90689.0          1000       5062396        322400.1  pthread_cond_timedwait
      0.1      22505463965        244       92235508.1          3317.5          1004    1927940055     406434853.8  write
      0.0      12146189551         14      867584967.9        133443.0          1255   10000048039    2688722576.0  futex
```

### Quick conclusions

- **`ProcessSynchronizer` ruled out as the source of `sem_wait` dominance.** Both nsys runs show
  `sem_wait` at the top of osrt at near-identical share (89.2% synch-on vs 82.9% synch-off). The
  signature is independent of the synchronizer's `fcntl` lock. Confirms what zarr v2 docs and
  source say: `_chunk_getitem` does not acquire the synchronizer — it's a no-op on read paths.
- **The 12-batch producer/consumer cycle survives nsys** (slow refills at batches 4 and 16 in
  synch-on Phase 3, 12 apart), so the cycle is structural and not measurement-dependent.
- **nsys numbers from this run look heavily inflated vs the unprofiled baseline** (LUH2 load
  314 → 1937 ms, w=12 throughput 9.6 → 2.0 samples/s, slow refills 9.5 → 26-49 s; synch-off
  Phase 3 timed out at batch 1). Single data point — not enough to conclude these are intrinsic
  nsys overhead vs a transient bad run on GPFS. **Not propagated to `knowledge.md`.** Re-verify
  if/when we run nsys again.
- **`--stats=true` osrt can't attribute `sem_wait` to PIDs**, so we still can't tell idle
  queue-wait apart from in-`__getitem__` blocking. Per-PID breakdown of
  `nsys_41436408.sqlite` (or nsys-ui) is the resolution path — deferred since `ProcessSynchronizer`
  is now off the suspect list and Lever A (parallel LUH2 var reads) is the next obvious move.

---

## Run 8 — Lever A: ThreadPoolExecutor over LUH2 var loads

### Jobs

- `41466184` (hydra override error, replaced) / `41467036` — `profiling-singlezarr`, `luh2_load_threads=1` (baseline reproduce)
- `41506187` — `profiling-singlezarr`, `luh2_load_threads=2`
- `41507170` — `profiling-singlezarr`, `luh2_load_threads=4`

### Interesting values

**Job 41467036 (luh2_load_threads=1)**

_Phase 1 — getitem + per-modality merged (50 samples)_

```
--- getitem | n=50 | mean= 1221.5 ms | median= 1127.2 ms | p95= 1351.9 ms | max= 3473.2 ms | sum=  61.08 s
--- per-modality (load + preprocess merged) ---
  _get_luh2_data        : calls= 100 | mean=  411.8 ms | median=  382.4 ms | max= 2676.2 ms | total= 41.18 s (67.4%)
  _get_population_data  : calls= 100 | mean=   69.2 ms | median=   46.4 ms | max= 2201.0 ms | total=  6.92 s (11.3%)
  _get_kg_data          : calls= 100 | mean=   41.2 ms | median=   39.2 ms | max=  135.2 ms | total=  4.12 s ( 6.8%)
  _get_static_data      : calls=  50 | mean=   87.0 ms | median=   84.5 ms | max=  136.2 ms | total=  4.35 s ( 7.1%)
  _get_hilda_prior      : calls=  50 | mean=   41.7 ms | median=   41.8 ms | max=   64.2 ms | total=  2.09 s ( 3.4%)
  _get_hilda_target     : calls=  50 | mean=   45.1 ms | median=   43.9 ms | max=   90.6 ms | total=  2.25 s ( 3.7%)
```

_Phase 2 — getitem + load vs preprocess split (50 samples)_

```
--- getitem | n=50 | mean= 1011.5 ms | median= 1009.9 ms | p95= 1137.4 ms | max= 1180.3 ms | sum=  50.58 s
--- per-modality (load | preprocess split) ---
  luh2          : calls= 100 | load mean=  307.7 ms total= 30.77 s (60.8%) | prep mean=   33.6 ms total=  3.36 s ( 6.6%)
  population    : calls= 100 | load mean=   46.5 ms total=  4.65 s ( 9.2%) | prep mean=    0.1 ms total=  0.01 s ( 0.0%)
  kg            : calls= 100 | load mean=   39.0 ms total=  3.90 s ( 7.7%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=  50 | load mean=   46.8 ms total=  2.34 s ( 4.6%) | prep mean=   18.9 ms total=  0.95 s ( 1.9%)
  hilda_prior   : calls=  50 | load mean=   43.5 ms total=  2.18 s ( 4.3%) | prep mean=    0.2 ms total=  0.01 s ( 0.0%)
  hilda_target  : calls=  50 | load mean=   43.7 ms total=  2.19 s ( 4.3%) | prep mean=    0.4 ms total=  0.02 s ( 0.0%)
```

_Phase 3 — num_workers sweep (80 batches per config)_

```
num_workers = 0
cold batch: 9791.4 ms (includes worker startup)
--- warm batches | n=79 | mean= 9247.7 ms | median= 9073.0 ms | p95=11229.3 ms | max=11825.2 ms | sum= 730.57 s
--- throughput |    0.9 samples/s |   0.11 batches/s

num_workers = 1
cold batch: 10022.0 ms (includes worker startup)
--- warm batches | n=79 | mean= 9140.8 ms | median= 8941.7 ms | p95=11127.3 ms | max=13005.7 ms | sum= 722.12 s
--- throughput |    0.9 samples/s |   0.11 batches/s

num_workers = 4
cold batch: 10105.8 ms (includes worker startup)
--- warm batches | n=79 | mean= 2392.9 ms | median=  233.1 ms | p95= 9280.5 ms | max=10900.1 ms | sum= 189.04 s
--- throughput |    3.3 samples/s |   0.42 batches/s

num_workers = 12
cold batch: 10797.3 ms (includes worker startup)
--- warm batches | n=79 | mean=  938.4 ms | median=    0.1 ms | p95= 8615.9 ms | max=11741.3 ms | sum=  74.13 s
--- throughput |    8.5 samples/s |   1.07 batches/s
```

**Job 41506187 (luh2_load_threads=2)**

_Phase 1 — getitem + per-modality merged (50 samples)_

```
--- getitem | n=50 | mean= 1222.5 ms | median= 1167.2 ms | p95= 1403.5 ms | max= 3202.4 ms | sum=  61.13 s
--- per-modality (load + preprocess merged) ---
  _get_luh2_data        : calls= 100 | mean=  444.0 ms | median=  414.9 ms | max= 2403.4 ms | total= 44.40 s (72.6%)
  _get_population_data  : calls= 100 | mean=   42.4 ms | median=   40.4 ms | max=   88.6 ms | total=  4.24 s ( 6.9%)
  _get_kg_data          : calls= 100 | mean=   39.0 ms | median=   38.4 ms | max=   62.7 ms | total=  3.90 s ( 6.4%)
  _get_static_data      : calls=  50 | mean=   82.1 ms | median=   82.3 ms | max=  114.1 ms | total=  4.10 s ( 6.7%)
  _get_hilda_prior      : calls=  50 | mean=   42.3 ms | median=   40.9 ms | max=  127.1 ms | total=  2.11 s ( 3.5%)
  _get_hilda_target     : calls=  50 | mean=   43.9 ms | median=   42.3 ms | max=   67.5 ms | total=  2.19 s ( 3.6%)
```

_Phase 2 — getitem + load vs preprocess split (50 samples)_

```
--- getitem | n=50 | mean= 1122.6 ms | median= 1114.6 ms | p95= 1273.1 ms | max= 1363.1 ms | sum=  56.13 s
--- per-modality (load | preprocess split) ---
  luh2          : calls= 100 | load mean=  380.1 ms total= 38.01 s (67.7%) | prep mean=   31.9 ms total=  3.19 s ( 5.7%)
  population    : calls= 100 | load mean=   40.1 ms total=  4.01 s ( 7.1%) | prep mean=    0.1 ms total=  0.01 s ( 0.0%)
  kg            : calls= 100 | load mean=   35.6 ms total=  3.56 s ( 6.3%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=  50 | load mean=   42.7 ms total=  2.14 s ( 3.8%) | prep mean=   18.6 ms total=  0.93 s ( 1.7%)
  hilda_prior   : calls=  50 | load mean=   41.7 ms total=  2.08 s ( 3.7%) | prep mean=    0.2 ms total=  0.01 s ( 0.0%)
  hilda_target  : calls=  50 | load mean=   40.3 ms total=  2.01 s ( 3.6%) | prep mean=    0.4 ms total=  0.02 s ( 0.0%)
```

_Phase 3 — num_workers sweep (80 batches per config)_

```
num_workers = 0
cold batch: 9870.0 ms (includes worker startup)
--- warm batches | n=79 | mean= 9024.0 ms | median= 9093.4 ms | p95= 9571.2 ms | max= 9755.4 ms | sum= 712.90 s
--- throughput |    0.9 samples/s |   0.11 batches/s

num_workers = 1
cold batch: 9318.9 ms (includes worker startup)
--- warm batches | n=79 | mean= 8960.1 ms | median= 8901.1 ms | p95= 9897.6 ms | max=13483.3 ms | sum= 707.85 s
--- throughput |    0.9 samples/s |   0.11 batches/s

num_workers = 4
cold batch: 9824.6 ms (includes worker startup)
--- warm batches | n=79 | mean= 2401.7 ms | median=    0.2 ms | p95= 9771.6 ms | max=10228.9 ms | sum= 189.74 s
--- throughput |    3.3 samples/s |   0.42 batches/s

num_workers = 12
cold batch: 11932.2 ms (includes worker startup)
--- warm batches | n=79 | mean=  885.4 ms | median=    0.1 ms | p95=10375.6 ms | max=11055.3 ms | sum=  69.95 s
--- throughput |    9.0 samples/s |   1.13 batches/s
```

**Job 41507170 (luh2_load_threads=4)**

_Phase 1 — getitem + per-modality merged (50 samples)_

```
--- getitem | n=50 | mean= 1104.7 ms | median= 1054.8 ms | p95= 1227.0 ms | max= 3043.4 ms | sum=  55.23 s
--- per-modality (load + preprocess merged) ---
  _get_luh2_data        : calls= 100 | mean=  384.3 ms | median=  351.4 ms | max= 2310.5 ms | total= 38.43 s (69.6%)
  _get_population_data  : calls= 100 | mean=   41.6 ms | median=   40.5 ms | max=   66.4 ms | total=  4.16 s ( 7.5%)
  _get_kg_data          : calls= 100 | mean=   40.5 ms | median=   38.7 ms | max=   86.1 ms | total=  4.05 s ( 7.3%)
  _get_static_data      : calls=  50 | mean=   84.9 ms | median=   82.3 ms | max=  152.6 ms | total=  4.24 s ( 7.7%)
  _get_hilda_prior      : calls=  50 | mean=   42.1 ms | median=   41.0 ms | max=  103.4 ms | total=  2.10 s ( 3.8%)
  _get_hilda_target     : calls=  50 | mean=   41.6 ms | median=   41.2 ms | max=   51.9 ms | total=  2.08 s ( 3.8%)
```

_Phase 2 — getitem + load vs preprocess split (50 samples)_

```
--- getitem | n=50 | mean= 1015.1 ms | median= 1006.4 ms | p95= 1141.1 ms | max= 1235.0 ms | sum=  50.75 s
--- per-modality (load | preprocess split) ---
  luh2          : calls= 100 | load mean=  322.7 ms total= 32.27 s (63.6%) | prep mean=   32.9 ms total=  3.29 s ( 6.5%)
  population    : calls= 100 | load mean=   41.0 ms total=  4.10 s ( 8.1%) | prep mean=    0.1 ms total=  0.01 s ( 0.0%)
  kg            : calls= 100 | load mean=   37.1 ms total=  3.71 s ( 7.3%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=  50 | load mean=   43.3 ms total=  2.16 s ( 4.3%) | prep mean=   18.8 ms total=  0.94 s ( 1.9%)
  hilda_prior   : calls=  50 | load mean=   40.7 ms total=  2.04 s ( 4.0%) | prep mean=    0.2 ms total=  0.01 s ( 0.0%)
  hilda_target  : calls=  50 | load mean=   40.6 ms total=  2.03 s ( 4.0%) | prep mean=    0.4 ms total=  0.02 s ( 0.0%)
```

_Phase 3 — num_workers sweep (80 batches per config)_

```
num_workers = 0
cold batch: 9380.0 ms (includes worker startup)
--- warm batches | n=79 | mean= 8401.1 ms | median= 8377.7 ms | p95= 8963.9 ms | max= 9079.6 ms | sum= 663.69 s
--- throughput |    1.0 samples/s |   0.12 batches/s

num_workers = 1
cold batch: 8840.4 ms (includes worker startup)
--- warm batches | n=79 | mean= 8118.0 ms | median= 8142.1 ms | p95= 8698.8 ms | max= 8897.8 ms | sum= 641.32 s
--- throughput |    1.0 samples/s |   0.12 batches/s

num_workers = 4
cold batch: 9681.0 ms (includes worker startup)
--- warm batches | n=79 | mean= 2392.2 ms | median=  460.1 ms | p95= 8856.9 ms | max= 9362.6 ms | sum= 188.98 s
--- throughput |    3.3 samples/s |   0.42 batches/s

num_workers = 12
cold batch: 11288.3 ms (includes worker startup)
--- warm batches | n=79 | mean=  875.1 ms | median=    0.1 ms | p95= 9917.1 ms | max=10413.0 ms | sum=  69.13 s
--- throughput |    9.1 samples/s |   1.14 batches/s
```

### Quick conclusions

- **Lever A as implemented gave no measurable benefit** at threads ∈ {1, 2, 4}.
  Per-LUH2 load (Phase 2): 307.7 → 380.1 → 322.7 ms (non-monotonic). w=12 throughput
  (Phase 3): 8.5 → 9.0 → 9.1 samples/s — flat within ~10% GPFS run-to-run noise
  (Run-4 baseline on the same store was 9.6 s/s; the threads=1 reproduce here is
  itself 0.9 s/s below, so do not over-interpret the 0.6 s/s gain from threads=1
  to threads=4).
- **The threads=2 regression (+23%, 308 → 380 ms) is the most informative single
  datapoint.** Pure I/O bottlenecks do not get worse when you add a thread. The
  signature is consistent with **CPU/lock contention in a shared resource** — most
  likely Blosc internal state (matches numcodecs #239 quantitatively), but could
  also be GIL churn from concurrent xarray/dask Python work.
- **This run did NOT localize the underlying bottleneck.** The test design
  conflated three things that all happen inside `_get_luh2_data`: (a) attempting
  I/O parallelism, (b) Blosc-in-threadpool contention, (c) per-`__getitem__`
  ThreadPoolExecutor creation overhead. The null end-to-end result is consistent
  with any of "I/O is the ceiling per-worker", "decompress is GIL/lock-contended",
  or "per-call Python overhead dominates" — we can't tell which without
  isolating each variable.
- **Run-to-run GPFS variance matters.** Run 4 baseline (same store, same code
  path as threads=1) showed 9.6 s/s; Run 8 threads=1 showed 8.5 s/s — ~10%
  delta from variance alone. Future comparisons against historical runs should
  include a fresh baseline arm or accept that noise floor.
- **Lever A propagated to `knowledge.md`** under "Disproven / dead ends" (in
  current form) and the candidate-levers table (struck through). `data.luh2_load_threads`
  knob kept in place; default 1.
- **Next:** diagnostic suite to localize I/O vs decompress vs Python overhead.
  Plan in `next_session_plan.md`: T1 cold/warm same-chunk, T2 NVMe staging,
  T3 workers sweep up to 19, T5 `strace -c -f` syscall summary.

---

## Run 9 — GPFS baseline + cold/warm + workers sweep [12,16,19]

### Jobs

- `41523471` — `profiling-singlezarr`, T1 cold/warm phase (50 pairs) + T3 extended workers sweep [12,16,19]

### Interesting values

Phase 1 — per-modality merged (50 samples):

```
--- getitem | n=50 | mean= 1156.0 ms | median= 1074.2 ms | p95= 1331.6 ms | max= 2718.4 ms | sum=  57.80 s
  _get_luh2_data        : calls= 100 | mean=  405.6 ms | median=  367.3 ms | max= 1971.1 ms | total= 40.56 s (70.2%)
  _get_population_data  : calls= 100 | mean=   47.2 ms | median=   45.0 ms | max=  133.9 ms | total=  4.72 s ( 8.2%)
  _get_kg_data          : calls= 100 | mean=   40.3 ms | median=   39.0 ms | max=  125.0 ms | total=  4.03 s ( 7.0%)
  _get_static_data      : calls=  50 | mean=   81.1 ms | median=   79.4 ms | max=  102.1 ms | total=  4.05 s ( 7.0%)
  _get_hilda_prior      : calls=  50 | mean=   42.7 ms | median=   41.3 ms | max=   79.6 ms | total=  2.14 s ( 3.7%)
  _get_hilda_target     : calls=  50 | mean=   42.9 ms | median=   41.5 ms | max=   70.2 ms | total=  2.14 s ( 3.7%)
```

Phase 2 — load vs preprocess split (50 samples):

```
--- getitem | n=50 | mean=  993.7 ms | median=  975.8 ms | p95= 1158.2 ms | max= 1211.1 ms | sum=  49.68 s
  luh2          : calls= 100 | load mean=  306.7 ms total= 30.67 s (61.7%) | prep mean=   33.4 ms total=  3.34 s ( 6.7%)
  population    : calls= 100 | load mean=   44.3 ms total=  4.43 s ( 8.9%) | prep mean=    0.1 ms total=  0.01 s ( 0.0%)
  kg            : calls= 100 | load mean=   37.7 ms total=  3.77 s ( 7.6%) | prep mean=    0.0 ms total=  0.00 s ( 0.0%)
  static        : calls=  50 | load mean=   41.8 ms total=  2.09 s ( 4.2%) | prep mean=   18.8 ms total=  0.94 s ( 1.9%)
  hilda_prior   : calls=  50 | load mean=   44.3 ms total=  2.22 s ( 4.5%) | prep mean=    0.2 ms total=  0.01 s ( 0.0%)
  hilda_target  : calls=  50 | load mean=   39.4 ms total=  1.97 s ( 4.0%) | prep mean=    0.4 ms total=  0.02 s ( 0.0%)
```

Phase 2b — cold/warm LUH2 (50 pairs):

```
--- cold/warm LUH2 | n=50 pairs ---
  cold  | mean =    462.4 ms | median =    387.6 ms
  warm  | mean =    355.7 ms | median =    346.8 ms
  delta | mean =    106.7 ms ( 23.1% of cold)
```

Phase 3 — num_workers sweep (80 batches each):

```
num_workers = 12
cold batch: 11847.3 ms (includes worker startup)
--- warm batches | n=79 | mean=  848.8 ms | median=    0.1 ms | p95= 9048.9 ms | max=10578.6 ms | sum=  67.06 s
--- throughput |    9.4 samples/s |   1.18 batches/s

num_workers = 16
cold batch: 11448.9 ms (includes worker startup)
--- warm batches | n=79 | mean=  661.1 ms | median=    0.1 ms | p95= 4217.9 ms | max= 9018.4 ms | sum=  52.23 s
--- throughput |   12.1 samples/s |   1.51 batches/s

num_workers = 19
cold batch: 13802.1 ms (includes worker startup)
--- warm batches | n=79 | mean=  644.1 ms | median=    0.1 ms | p95= 1945.2 ms | max=13910.1 ms | sum=  50.89 s
--- throughput |   12.4 samples/s |   1.55 batches/s
```

### Quick conclusions

(reserved for smarter model after all three jobs)

---

## Run 10 — NVMe staging attempt (cancelled — time limit)

### Jobs

- `41524127` — CANCELLED at 2-hour wall limit; rsync of 66 G zarr store (many-small-files)
  over GPFS did not complete. T2 (NVMe vs GPFS) deferred; not needed given T1 result from Run 9.

---

## Run 11 — T5 strace syscall summary (w=12, short capture)

### Jobs

- `41536968` — FAILED: `sem_wait` is not a Linux syscall; strace rejected it and exited immediately.
- `41542527` — `profiling-singlezarr`, strace -c -f (sem_wait removed from filter), num_getitem=5, num_batches=30, workers_sweep=[12], num_coldwarm=0

### Interesting values

(fill in after /hpc-pull)

### Quick conclusions

(reserved for smarter model after combined analysis)

## Run 12 — Chunk-size probe: 100 random chunks per variable, all vars (size-only)

### Jobs

- `41548701` — `profile_chunk_sizes.py` over `AI4LAND_NON_preprocessed-data-2000-2015.zarr`; 100 random chunk files per variable (all subdirs, not just LUH2), `os.path.getsize` only, CSV output to `jobs-map/41548701/chunk_sizes.csv`.

### Interesting values

Raw CSV: `jobs-map/41548701/chunk_sizes.csv` (2303 rows = 100 × 23 vars).
Box plot: `jobs-map/41548701/chunk_sizes.png` (log-y, all vars except `time`, `latitude`, `longitude`).

Per-variable distribution (n=100 unless noted; bytes):

| var               | n   | min  | median | mean   | p95     | max     | total    |
| ----------------- | --- | ---- | ------ | ------ | ------- | ------- | -------- |
| ASPECT            | 100 | 4248 | 4248   | 199836 | 882364  | 888697  | 19983601 |
| LULC_states       | 100 | 4248 | 4291   | 26650  | 113772  | 117719  | 2664971  |
| SLOPE             | 100 | 4248 | 5202   | 310495 | 887464  | 891256  | 31049470 |
| c3ann             | 100 | 4248 | 4248   | 116802 | 779684  | 806803  | 11680176 |
| c3nfx             | 100 | 4248 | 4248   | 84545  | 709914  | 811468  | 8454509  |
| c3per             | 100 | 4248 | 4248   | 74485  | 703285  | 770635  | 7448516  |
| c4ann             | 100 | 4248 | 4248   | 162788 | 760041  | 800586  | 16278813 |
| c4per             | 100 | 4248 | 4248   | 62098  | 561814  | 731188  | 6209750  |
| elevation         | 100 | 3899 | 297152 | 262764 | 345209  | 368320  | 26276416 |
| kg_class          | 100 | 4248 | 4258   | 16123  | 63650   | 99368   | 1612276  |
| latitude          | 1   | 6743 | 6743   | 6743   | 6743    | 6743    | 6743     |
| longitude         | 1   | 8634 | 8634   | 8634   | 8634    | 8634    | 8634     |
| number_of_people  | 100 | 4248 | 4248   | 109695 | 598683  | 667159  | 10969491 |
| pastr             | 100 | 4248 | 4248   | 77415  | 464050  | 715426  | 7741505  |
| primf             | 100 | 4248 | 4248   | 70200  | 546213  | 749166  | 7019954  |
| primn             | 100 | 4248 | 4248   | 75312  | 578231  | 731727  | 7531180  |
| range             | 100 | 4248 | 4248   | 90552  | 682598  | 733003  | 9055196  |
| secdf             | 100 | 4248 | 4248   | 90223  | 648553  | 760534  | 9022269  |
| secdn             | 100 | 4248 | 4248   | 68793  | 462280  | 649022  | 6879338  |
| secma             | 100 | 4248 | 4248   | 123749 | 717810  | 742677  | 12374909 |
| secmb             | 100 | 4248 | 4248   | 130895 | 622613  | 678268  | 13089545 |
| time              | 1   | 72   | 72     | 72     | 72      | 72      | 72       |
| urban             | 100 | 4248 | 4248   | 42198  | 178419  | 570994  | 4219798  |
| weighted_ORGANIC  | 100 | 8472 | 8472   | 329210 | 1531294 | 1547769 | 32921006 |
| weighted_PCT_CLAY | 100 | 8472 | 8564   | 374137 | 1534175 | 1548580 | 37413673 |
| weighted_PCT_SAND | 100 | 8472 | 8694   | 358749 | 1518020 | 1637199 | 35874937 |

Raw observations (no interpretation):

- 18/20 sampled 3-D variables show **median == minimum == 4248 B** (or 8472 B for `weighted_*`). Mean is 6–90× the median.
- LUH2 means range **42 KB (urban)** → **163 KB (c4ann)**.
- `LULC_states` mean = 26.6 KB, p95 = 113.8 KB; smallest of the 3-D vars.
- `kg_class` mean = 16.1 KB, p95 = 63.6 KB.
- Static `weighted_*` and `ASPECT/SLOPE/elevation` are the largest: means 200–370 KB, p95 ≈ 0.88–1.55 MiB.
- `time`, `latitude`, `longitude` are coord arrays with a single chunk each (72 B, 6.7 KB, 8.6 KB).

### Quick conclusions

- **Distribution is heavily bimodal for every 3-D variable.** Median = minimum compression floor (4248 B for most, 8472 B for `weighted_*`) — these are the all-fill / ocean chunks that Blosc compresses to nothing. Means are 6–90× the medians, p95 reaches 100 KB – 1.5 MiB — these are the land chunks where the actual data lives.
- **LUH2 means span only ~4× across vars** (42 KB `urban` → 163 KB `c4ann`). Variable choice is not the bytes differentiator; spatial position (land vs ocean) dominates.
- **Per-LUH2-call payload (14 vars × 2 timesteps) at means: ≈ 2.5 MiB; at p95 land chunks: ≈ 19 MiB.** Either way, far under GPFS single-stream bandwidth (556 MB/s → 19 MiB in 34 ms). The observed 314 ms LUH2 baseline is **9× more** than even the worst-case byte transfer time — reinforces metadata + Python overhead as the bottleneck, not bytes.
- **Static `weighted_*` vars are the largest** (means 330–374 KB, p95 ≈ 1.5 MiB). Static loads happen 1× per sample, not 2× like LUH2, but still worth knowing.
- **`kg_class` (mean 16 KB, p95 64 KB) and `LULC_states` (mean 27 KB, p95 114 KB) are tiny** — these are essentially metadata-cost only, byte transfer negligible.

## Run 13 — File-count × file-size sweep on GPFS (synthetic files)

### Jobs

- `41549468` — `profile_file_count.py`; writes synthetic files for the matrix
  SIZES ∈ {10 KB, 100 KB, 1 MB, 10 MB, 100 MB} × COUNTS ∈ {1, 10, 100, 1000, 10000}
  (cells > 10 GiB skipped), then sequentially `open().read()` each cell and times
  wall ms. CSV output to `jobs-map/41549468/file_count.csv`.
  Caveat: page cache is warm (write→read in same job), so absolute ms underestimates
  cold reads — cross-cell scaling shape is the signal.

### Interesting values

Raw CSV: `jobs-map/41549468/file_count.csv` (22 cells).

**Wall ms per cell:**

| file_size \ count | n=1   | n=10   | n=100   | n=1000   | n=10000  |
| ----------------- | ----- | ------ | ------- | -------- | -------- |
| 10 KB             | 0.21  | 1.45   | 11.24   | 110.93   | 1081.52  |
| 100 KB            | 0.13  | 1.19   | 12.12   | 121.27   | 1226.33  |
| 1 MB              | 2.08  | 17.80  | 145.95  | 1874.92  | 16690.39 |
| 10 MB             | 4.05  | 42.95  | 1291.87 | 19583.76 | —        |
| 100 MB            | 44.46 | 697.88 | 6052.98 | —        | —        |

**Per-file ms (= wall_ms / count):**

| file_size \ count | n=1   | n=10  | n=100 | n=1000 | n=10000 |
| ----------------- | ----- | ----- | ----- | ------ | ------- |
| 10 KB             | 0.21  | 0.145 | 0.112 | 0.111  | 0.108   |
| 100 KB            | 0.13  | 0.119 | 0.121 | 0.121  | 0.123   |
| 1 MB              | 2.08  | 1.78  | 1.46  | 1.87   | 1.67    |
| 10 MB             | 4.05  | 4.30  | 12.9  | 19.6   | —       |
| 100 MB            | 44.46 | 69.8  | 60.5  | —      | —       |

**Constant-bytes anti-diagonal (≈ 100 MB total):**

| split         | wall ms | × vs 1-file baseline |
| ------------- | ------- | -------------------- |
| 1 × 100 MB    | 44      | 1.0×                 |
| 10 × 10 MB    | 43      | 1.0×                 |
| 100 × 1 MB    | 146     | 3.3×                 |
| 1000 × 100 KB | 121     | 2.7×                 |
| 10000 × 10 KB | 1082    | **24.6×**            |

### Quick conclusions

- **Warm-cache per-`open()` cost on GPFS is ~0.11–0.12 ms** for small files (≤ 100 KB), independent of file count and ~constant across the 10 KB ↔ 100 KB range — so bytes are noise in that regime; cost is **pure metadata**.
- **Linear file-count scaling at fixed file size for small files**: 1.45 → 11.24 → 110.93 → 1081.5 ms across n=10 → 100 → 1000 → 10000 (10 KB). Each 10× more files = 10× more wall. No super-linearity at small sizes.
- **At constant total bytes (100 MB), splitting it into many small files is dramatically slower**: 1 × 100 MB = 44 ms vs **10000 × 10 KB = 1082 ms (24.6×)**. This is the cleanest confirmation that **file count, not bytes, drives cost** in the size regime that matches zarr chunks.
- **At larger files (10 MB+) per-file cost grows with count** (4.3 → 12.9 → 19.6 ms at 10 MB): payload exits page cache so reads start hitting disk. Not relevant to our 10–150 KB zarr-chunk regime.
- **Bytes start mattering above ~1 MB**: per-file 1.7 ms at 1 MB ≈ 600 MB/s page-cache throughput — and that's still 14× the small-file metadata cost.
- **Cold-cache scaling is implied but not measured.** profile_layers showed cold per-chunk = 19–20 ms (zarr chunks, real store); warm here = 0.12 ms. The ~160× gap is the GPFS metadata-server round-trip on first open. Cold-cache penalty for many files should be **dramatically worse** than the 24.6× we see here.
- **Implication for the LUH2 call** (14 chunks × ~100 KB): 14 × 0.12 ms = 1.7 ms warm = trivial; but cold = 14 × 19 ms ≈ 270 ms — matches the ~107 ms cold-I/O share of the 314 ms baseline plus GPFS server-side prefetch absorbing some. **Merging the 14 vars into 1 chunk file** would collapse 14 metadata round-trips to 1 — expected cold savings ≈ 13 × 19 ms ≈ **250 ms** (≈ 80% of the LUH2 baseline).
- **Does not directly test "larger directory → slower per-open"** (the GPFS metadata-cache-miss-rate hypothesis for store-size scaling). Warm cache bypasses the metadata server, so this question remains open. Pending the recreated large singlezarr.

## Run 14 — Directory-size effect: 100K vs 1M files in one dir (10 KB each, random reads)

### Jobs

- `41551638` — `profile_file_count.py` (reduced matrix); two cells in one tmp dir:
  pool A = 100K × 10 KB (1 GB), pool B = 1M × 10 KB (10 GB). Reads 10 000 random files
  per pool, **shuffled interleaved order across both pools** so cache state is symmetric.
  Output: per-file wall µs to `jobs-map/41551638/file_count.csv`. Tests whether per-`open()`
  cost grows with directory size (directory-size / metadata-cache-miss-rate hypothesis).

### Interesting values

Per-file wall µs distribution (10 000 random reads per pool, interleaved shuffled order):

| pool       | n      | mean    | p50     | p90     | p95     | p99     | max     |
| ---------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- |
| 100K files | 10 000 | 8.42 ms | 1.33 ms | 20.4 ms | 22.9 ms | 33.6 ms | 72.7 ms |
| 1M files   | 10 000 | 7.78 ms | 0.93 ms | 20.1 ms | 23.0 ms | 32.2 ms | 70.4 ms |

Raw CSV: `jobs-map/41551638/file_count.csv` (20 000 rows).

### Quick conclusions

- **Directory size (100K vs 1M files) does NOT inflate per-`open()` cost.** Distributions are within 1–8% of each other across all percentiles; the 1M-pool mean is actually 8% _faster_ (cache shuffle residue, not signal). **The "larger store → slower per-sample time" empirical finding is NOT explained by per-open cost scaling with files-per-directory.**
- **Cold metadata reads are much more expensive than Run 13's warm-cache numbers suggested.** Run 13 measured 0.11 ms/file warm; here median = 0.9–1.3 ms and p95 = 23 ms. The **p95 ≈ 23 ms matches profile_layers' cold `raw_read` of 19–20 ms** — these are real GPFS metadata-server round-trips.
- **Heavily bimodal**: median ~1 ms (warm metadata cache hit) vs p90+ at ≥ 20 ms (cold metadata miss). Suggests roughly half of reads land warm, half cold under this access pattern. Real workloads with persistent workers / re-reads would shift further warm.
- **Refined estimate for the LUH2 cost in production**: 14 chunks × p95 23 ms ≈ **322 ms** — matches the observed 314 ms LUH2 baseline almost exactly. This strongly suggests the LUH2 baseline is dominated by **mostly-cold metadata round-trips**, not mixed warm/cold.
- **Refined "merge 14 vars into 1" savings (Lever 3) estimate**: 14 × ~22 ms cold metadata = ~308 ms collapsing to ~22 ms = **~280 ms saved per LUH2 call (~89% of baseline)**. Tightens the previous range estimate of 100–250 ms.
- **Open question left intact**: why does the 213 G store run ~1.9–3× slower than the 64 G one (Run 6)? Not files-per-directory. Candidate mechanisms still on the table: GPFS server-side contention across all node workers, larger total chunk count → more total opens during a training epoch, or something else. The recreated large store will provide the direct comparison.

## Run 15 — Full file_count matrix re-run with random reads + cross-cell shuffle

### Jobs

- `41595712` — `profile_file_count.py` restored to full Run 13 matrix
  (SIZES ∈ {10 KB, 100 KB, 1 MB, 10 MB, 100 MB} × COUNTS ∈ {1, 10, 100, 1000, 10000},
  cells > 10 GiB skipped). Reads up to `min(count, 10000)` random files per cell;
  reads from all cells shuffled into one interleaved sequence. Per-file wall µs
  output to `jobs-map/41595712/file_count.csv`. Goal: re-validate Run 13 against
  the new realistic access pattern + characterize the warm-cache regime per-file.

### Interesting values

22 cells, 34 555 per-file rows. Per-cell wall µs distribution:

| size   | count | n     | mean µs | p50 µs | p90 µs | p95 µs | p99 µs | max µs      |
| ------ | ----- | ----- | ------- | ------ | ------ | ------ | ------ | ----------- |
| 10 KB  | 1     | 1     | 107     | 107    | 107    | 107    | 107    | 107         |
| 10 KB  | 10    | 10    | 138     | 107    | 404    | 404    | 404    | 404         |
| 10 KB  | 100   | 100   | 121     | 108    | 144    | 255    | 370    | 370         |
| 10 KB  | 1000  | 1000  | 117     | 106    | 127    | 219    | 336    | 475         |
| 10 KB  | 10000 | 10000 | 118     | 105    | 130    | 225    | 339    | 2972        |
| 100 KB | 1     | 1     | 128     | 128    | 128    | 128    | 128    | 128         |
| 100 KB | 10    | 10    | 120     | 122    | 129    | 129    | 129    | 129         |
| 100 KB | 100   | 100   | 163     | 123    | 225    | 293    | 1775   | 1775        |
| 100 KB | 1000  | 1000  | 134     | 122    | 148    | 237    | 373    | 698         |
| 100 KB | 10000 | 10000 | 134     | 121    | 150    | 242    | 352    | 881         |
| 1 MB   | 1     | 1     | 826     | 826    | 826    | 826    | 826    | 826         |
| 1 MB   | 10    | 10    | 1932    | 878    | 9715   | 9715   | 9715   | 9715        |
| 1 MB   | 100   | 100   | 5403    | 850    | 4134   | 5863   | 388765 | 388765      |
| 1 MB   | 1000  | 1000  | 1942    | 843    | 4564   | 6210   | 16220  | 35051       |
| 1 MB   | 10000 | 10000 | 2199    | 833    | 4152   | 6207   | 11098  | **2131536** |
| 10 MB  | 1     | 1     | 8177    | 8177   | 8177   | 8177   | 8177   | 8177        |
| 10 MB  | 10    | 10    | 11015   | 4206   | 66939  | 66939  | 66939  | 66939       |
| 10 MB  | 100   | 100   | 9891    | 4334   | 28819  | 42441  | 62124  | 62124       |
| 10 MB  | 1000  | 1000  | 9116    | 4327   | 21921  | 30875  | 66345  | 140457      |
| 100 MB | 1     | 1     | 31718   | 31718  | 31718  | 31718  | 31718  | 31718       |
| 100 MB | 10    | 10    | 51237   | 35110  | 181135 | 181135 | 181135 | 181135      |
| 100 MB | 100   | 100   | 48257   | 34253  | 93831  | 124290 | 675596 | 675596      |

Per-cell **mean per-file ms vs Run 13's sequential cell aggregate** (10 KB × 10000): Run 13 → 0.108 ms/file; Run 15 → 0.118 ms/file (+9%). 1 MB × 1000: Run 13 → 1.87 ms/file; Run 15 → 1.94 ms/file (+4%). **Random vs sequential matches within ~10%** at this working-set size.

### Quick conclusions

- **Working set fits in dentry cache → reads are mostly WARM.** Total inodes across all 22 cells ≈ 55 K, well under the ~500 K Linux dentry cache ceiling. Median per-file at 10 KB is **105 µs**, vs Run 14's **930 µs** at the same file size (10× slower). The variable that flipped Run 14 into the cold regime was pool size (≥ 100 K inodes), not random vs sequential access.
- **Metadata floor ≈ 100 µs per warm `open()`**, constant from 10 KB → 100 KB (medians 105 → 122 µs, +16% — pure metadata). Above ~100 KB, byte cost emerges: median 1 MB = 843 µs, 10 MB = 4.3 ms, 100 MB = 33 ms.
- **Effective per-stream throughput from page cache grows with file size**: 770 MB/s at 100 KB → 1.2 GB/s at 1 MB → 2.3 GB/s at 10 MB → 3.1 GB/s at 100 MB. **4–5× faster than cold-disk single-stream** (556 MB/s, Run 12). GPFS page cache is doing real work.
- **Confirms Run 13's cell aggregates** (sequential read): mean per-file at 10 KB × 10000 = 0.118 ms (Run 13 was 0.108 ms); 1 MB × 1000 = 1.94 ms (Run 13: 1.87 ms). Random+shuffled access doesn't change the warm-cache totals — only the regime does (cold vs warm).
- **Tail spikes are real but rare.** p99 stays orderly (≤ 339 µs at 10 KB; ≤ 16 ms at 1 MB), but isolated max values explode: **2.13 sec at (1 MB, 10000)**, 388 ms at (1 MB, 100), 181 ms at (100 MB, 10). Likely GPFS server-side stalls (lock contention or another tenant's I/O). These appear in every regime and bias means upward — distrust mean, trust median.
- **For modelling production LUH2 cost**: this run is the warm-cache lower bound (≈ 14 × 100 µs = 1.4 ms metadata + bytes), Run 14 is the cold-cache upper bound (≈ 14 × 23 ms = 322 ms). Observed 314 ms baseline sits right at Run 14's prediction — production is operating in the **cold-metadata regime**, not warm.
- **Implication for the "larger store → slower" question**: Real LUH2 variable directories have ~40 K (small store) or ~140 K (large store) chunks. **40 K just barely fits in dentry cache; 140 K does not.** That's the likely mechanism: cache-fit at the smaller store (mostly warm reads), cache-miss at the larger one (mostly cold reads, 10× per-open penalty). This is the "transition" Run 14's pool sizes already crossed — both 100 K and 1 M pools were past the cache threshold. A targeted test sweeping pool size 10 K → 1 M would pin the threshold exactly.

## Run 16 — Production-shape simulation: 23-subdir vs 1-subdir, 12-process concurrency

### Jobs

- `41596212` — `profile_dataloader_sim.py`. Two synthetic stores at
  `/gpfs/scratch/bsc32/bsc096444/dataloader_sim/`:
  - **Store A** (current layout): 23 subdirs × 13 000 files (~300 K inodes total),
    sizes sampled with replacement from Run 12's 3-D-var chunk distribution.
  - **Store B** (merged layout): 1 subdir × 13 000 files, each file = byte-sum of
    23 sampled small chunks (~2.3 MB each).
  - Both stores roughly the same total bytes; differ in inode count (300 K vs 13 K).
  - Access pattern: 12 `multiprocessing.Pool` workers (one per DataLoader worker).
    100 samples per worker. 1 sample in A = 40 random file opens across random
    subdirs; 1 sample in B = 2 random file opens from the single subdir.
  - Output CSV: `jobs-map/41596212/dataloader_sim.csv` with columns
    `store,worker,sample,wall_ms`.
  - Stores are kept across runs (manual `rm -rf` to clean up); `BASE_DIR` lives
    under user scratch, not the repo tmp.
  - Caveat: A is run first → its 48 K reads evict Store B's creation-residue page
    cache before B's measurement, biasing B slightly cold. Re-run script for a
    second pass to get warm-state numbers if needed.

### Interesting values

1 200 samples per store (12 workers × 100 samples). Per-sample wall ms:

| store | n     | mean  | p50   | p90   | p95   | p99    | max    | total_job_s |
| ----- | ----- | ----- | ----- | ----- | ----- | ------ | ------ | ----------- |
| A     | 1 200 | 369.4 | 324.3 | 442.1 | 500.6 | 2038.3 | 3795.7 | 443         |
| B     | 1 200 | 11.85 | 5.78  | 31.63 | 40.05 | 55.39  | 76.23  | 14          |

**Mean ratio A/B = 31.2×, p95 ratio = 12.5×, total-time ratio = 31.6×.**

Per-worker mean spread (across the 12 workers): A 348–404 ms (1.16×), B 10–16 ms (1.60×).

Per-open implied cost (sample_ms / files_per_sample):

- A: 369.4 ms / 40 = **9.2 ms/open** (matches Run 14's 8.4 ms cold-metadata mean)
- B: 11.85 ms / 2 = **5.9 ms/open** (for ~2 MB files — partly bytes, partly metadata)

Raw CSV: `jobs-map/41596212/dataloader_sim.csv` (2 400 rows).

### Quick conclusions

- **Lever 3 (variables merging) is confirmed at production-like concurrency: ~30× speedup on per-sample read time.** Store B (1 subdir × 2 MB files, 2 opens/sample) reads in 12 ms mean vs Store A's 369 ms (40 opens/sample). The extra factor on top of 40→2 = 20× comes from larger files getting more page-cache benefit per byte.
- **Store A's 369 ms / 40 = 9.2 ms per open** lines up with Run 14's 8.4 ms cold mean — confirms the production layout sits firmly in the cold-metadata regime even at this 30 GB synthetic store size, despite being well under node RAM (512 GB). Concurrency from 12 workers adds metadata-server pressure that keeps reads cold.
- **Store A's heavy tail** (p99 2.0 s, max 3.8 s) is significant. Even with median 324 ms, occasional 2–4 s samples explain training-loop "stall" patterns better than the mean does. 12-worker concurrency on a single metadata server amplifies any contention.
- **Implication for production LUH2 cost (314 ms baseline)**: Store A's per-open 9.2 ms × 14 chunks ≈ 129 ms is _less_ than 314 ms because (a) profile_layers measured single-stream cold (≈ 19 ms) whereas concurrent workers get some parallelism on the metadata server, and (b) xarray/dask Python overhead (~224 ms) isn't in this synthetic test. Adding 224 ms of fixed Python cost to either Store A or B is the right mental model.
- **Refined expected savings from Lever 3 on production**: 314 ms LUH2 baseline → if reads collapse to 2 large opens at ~12 ms total + 224 ms Python overhead = ~236 ms (~25% savings). If we also bypass xarray (Lever 1) — combined: ~12 ms metadata/bytes + lower Python overhead = **could push LUH2 cost below 50 ms (≈ 85% reduction)**.
- **Page cache likely warm for B during measurement.** B's 13 K × ~2 MB ≈ 26 GB pool fits in node RAM; after creation, most files are still page-cached during read. A's reads (48 K opens, partly cold) likely evicted some of B's pages but not all. So B's 12 ms is a _warm-leaning_ estimate. A more rigorous B measurement would re-run on a cold node — but the directional answer (huge speedup) is clear.

## Run 17 — Synthetic 1M × 10 KB pool vs real zarr store, 1000 random reads each

### Jobs

- `41596254` — `profile_synth_vs_real.py`. Creates 1M × 10 KB files in
  `/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/tmp/synth_vs_real/synth`, enumerates
  every chunk file in `/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr`,
  picks 1 000 random files from each, reads in shuffled interleaved order.
  Output CSV: `jobs-map/41596254/synth_vs_real.csv` with columns
  `source,path,wall_us,file_bytes`. Tests whether the synthetic-pool numbers
  (Run 14) generalize to real zarr chunks.

### Interesting values

1 000 random reads per source (2 000 total, interleaved shuffled). Per-file wall µs:

| source | n     | mean µs | p50    | p90    | p95    | p99    | max    |
| ------ | ----- | ------- | ------ | ------ | ------ | ------ | ------ |
| synth  | 1 000 | 9 686   | 6 909  | 21 371 | 24 040 | 34 403 | 46 233 |
| real   | 1 000 | 19 425  | 18 624 | 26 603 | 29 710 | 41 669 | 71 079 |

Ratios real/synth: mean 2.0×, median 2.7×, p95 1.24×, p99 1.21×.

File bytes:

| source | mean   | p50    | p95     | max     |
| ------ | ------ | ------ | ------- | ------- |
| synth  | 10 240 | 10 240 | 10 240  | 10 240  |
| real   | 92 442 | 4 248  | 660 695 | 807 236 |

Synth fixed at 10 KB; real bimodal — median 4248 B compression floor (ocean), tail to 800 KB (land).

### Quick conclusions

- **Conclusions deferred** — synth bytes (fixed 10 KB) are too different from real bytes (median 4 KB, mean 92 KB, p95 660 KB) to attribute the wall-µs gap to anything specific. Re-test (Run 18) will draw synth sizes from the real chunk distribution.

## Run 18 — Synth 1M files (sizes from real chunk distribution) vs real zarr, 1000 reads each

### Jobs

- `41598365` — `profile_synth_vs_real.py` updated: same 1M file count as Run 17,
  but each synth file's size is sampled (with replacement) from the Run 12
  chunk-size distribution (3-D vars only). Total synth pool ≈ 92 GB. Same
  read pattern as Run 17: 1 000 random reads from each (synth + real), shuffled
  interleaved order, per-file wall µs + file_bytes captured.
  CSV: `jobs-map/41598365/synth_vs_real.csv`. Eliminates the bytes confound
  between synth and real; if median wall still differs, cause is real-store-
  specific (shared FS contention, cross-mount metadata server, etc.).

### Interesting values

| source | n     | mean µs | p50    | p90    | p95    | p99    | max    |
| ------ | ----- | ------- | ------ | ------ | ------ | ------ | ------ |
| synth  | 1 000 | 15 763  | 16 870 | 27 369 | 32 246 | 44 097 | 64 224 |
| real   | 1 000 | 21 151  | 19 995 | 30 073 | 34 005 | 42 615 | 65 074 |

Byte distributions (now well-matched):

| source | mean    | p50   | p95     | max       |
| ------ | ------- | ----- | ------- | --------- |
| synth  | 144 767 | 4 258 | 806 803 | 1 637 199 |
| real   | 92 442  | 4 248 | 660 695 | 807 236   |

Ratios real/synth: **mean 1.34×, p50 1.19×, p95 1.05×**. Compare with Run 17 (fixed-10 KB synth): mean 2.0×, p50 2.7×, p95 1.24×.

### Quick conclusions

- **Synthetic is a valid proxy for real-store reads** once bytes match: the gap shrinks from Run 17's 2.7× median to **1.19× median, 1.05× p95**. Most of the Run 17 gap was bytes, not real-store-specific overhead. Remaining ~20–30% real-side surcharge is consistent with shared-filesystem noise / cross-allocation mount.
- **Per-open cost in the cold regime is ~20 ms p50 / 32–34 ms p95**, regardless of whether the pool is synthetic or real zarr layout. Any conclusion drawn from synthetic benchmarks at this scale transfers within ~20% to production.

## Run 19 — Rerun Run 16 stores in ABA pattern (cache/drift check)

### Jobs

- `41598870` — `profile_dataloader_sim_aba.py`. Reuses stores at
  `/gpfs/scratch/bsc32/bsc096444/dataloader_sim/` (Run 16's, no recreation).
  Reads A1 → B → A2; different RNG seed per pass (so A2 reads different random
  files than A1, similar working-set fraction). Output CSV columns:
  `store,pass,worker,sample,wall_ms`. **A1 vs A2 isolates within-job drift**
  (GPFS load fluctuation, cache warming on accidental overlaps). A1 vs A2 ≈
  → Run 16's 31× ratio is reproducible; A2 ≪ A1 → drift was real and the ratio
  is partly cache-dominated.

### Interesting values

Per-sample wall ms (12 workers × 100 samples each pass):

| store | pass | n     | mean   | p50    | p90    | p95    | p99    | max     |
| ----- | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------- |
| A     | 1    | 1 200 | 669.27 | 646.70 | 805.93 | 844.91 | 924.93 | 5 437.6 |
| B     | 1    | 1 200 | 93.51  | 84.47  | 125.00 | 141.56 | 174.89 | 2 249.0 |
| A     | 2    | 1 200 | 524.16 | 520.94 | 647.61 | 684.27 | 753.25 | 783.1   |

**Ratios**: A1/A2 = 1.28× (within-job drift), A1/B = **7.16×**, A2/B = 5.61×, A1 p95 / B p95 = 5.97×.

Comparison vs Run 16 (same stores, fresh job, no creation in this run):

| metric    | Run 16  | Run 19 (A1) | change |
| --------- | ------- | ----------- | ------ |
| A mean ms | 369     | 669         | 1.81×  |
| B mean ms | 12      | 94          | 7.8×   |
| A/B ratio | **31×** | **7.16×**   | 0.23×  |

### Quick conclusions

- **Run 16's 31× speedup was heavily cache-fit-contaminated.** When both stores are read in a fresh job (no creation residue in page cache, dentry cache state degraded since Run 16's creation), B's mean explodes from 12 ms → 94 ms (7.8× slower) and A only doubles (1.8×). **The "true" cold-cache speedup is 7×**, not 31×. Run 16's B was being measured warm because its 26 GB pool was still in page cache from creation seconds earlier; that benefit doesn't survive in production where B is created once and read many times across days.
- **A1 → A2 drift (1.28×) is small** → within-job GPFS load / cache state is stable enough that A's measurement is robust. Most A reads stay cold even after a previous A pass exercised the same pool, because random sampling of 48 K out of 299 K inodes still gives mostly fresh reads.
- **B's residual speedup (7×) still includes a working-set fit advantage**: B's 13 K-inode pool fits in dentry cache; A's 299 K does not. So Run 19's 7× is partly file-count-per-sample (A reads 40, B reads 2) and partly cache-fit (B's smaller pool). Run 20 below isolates the file-count effect at the fits-in-cache scale.
- **Production extrapolation** (213 G real store, ~140 K chunks/var in current layout vs ~140 K total in merged): both pools well beyond dentry cache (in cold regime), so Run 14 says per-open cost converges (~23 ms p95). Speedup at production scale ≈ 41 opens × 23 ms vs 2 opens × 23 ms = **~20× file-count-only**.

## Run 20 — Same-shape simulation at "fits-in-cache" scale (31K vs 1,350 inodes)

### Jobs

- `41598871` — `profile_dataloader_sim_small.py`. Two new stores at
  `/gpfs/scratch/bsc32/bsc096444/dataloader_sim_small/`:
  - Store A: 23 subdirs × 1 350 files = **31 050 inodes** (~2.8 GB total)
  - Store B: 1 subdir × 1 350 files (each = byte-sum of 23 small chunks ≈ 2 MB) = ~2.7 GB
  - Sizes sampled from Run 12 chunk distribution.
  - 12 workers × 100 samples each, 40 opens/sample (A) vs 2 opens/sample (B).
  - **ABA read pattern** (A1 → B → A2).
  - **At this scale, both pools fit trivially in dentry+page cache.** If the
    A/B speedup ratio at Run 16's scale (31×) collapses here to something near
    1× or 2×, the original 31× was largely _cache-fit-dominated_. If the speedup
    remains large (≥ 10×), the per-sample file-count effect drives most of it
    even in the warm regime — supporting Lever 3 at production scale.
  - CSV: `jobs-map/41598871/dataloader_sim_small.csv`, same columns as Run 19.

### Interesting values

Per-sample wall ms (12 workers × 100 samples each pass):

| store | pass | n     | mean  | p50   | p90   | p95   | p99    | max     |
| ----- | ---- | ----- | ----- | ----- | ----- | ----- | ------ | ------- |
| A     | 1    | 1 200 | 38.88 | 28.26 | 59.06 | 71.49 | 109.58 | 1 434.7 |
| B     | 1    | 1 200 | 15.75 | 6.46  | 40.03 | 50.40 | 86.16  | 1 124.8 |
| A     | 2    | 1 200 | 8.57  | 6.72  | 18.53 | 22.84 | 35.41  | 51.5    |

**Ratios**: A1/A2 = **4.54× (huge drift — A warmed fast)**, A1/B = 2.47×, **A2/B = 0.54× (A faster than B!)**.

Per-open implied cost (sample_ms / files_per_sample):

| pass      | A1 mean | A2 mean | B mean |
| --------- | ------- | ------- | ------ |
| ms / open | 0.97    | 0.21    | 7.88   |

### Quick conclusions

- **At fits-in-cache scale, the A/B "merge advantage" collapses.** A2 (warm) is _faster_ than B by ~0.54×. The ~30× and ~7× ratios from Runs 16 / 19 were both substantially **cache-effects of pool-size difference**, not pure file-count-per-sample.
- **In the warm regime, A's per-open is ~0.21 ms** (pure metadata-cache hit) and **B's per-open is ~7.9 ms** (byte-bound: 2 MB at ~250 MB/s effective). Result: small files win when their metadata is cached because byte transfer becomes the dominant cost for big files.
- **A1 → A2 = 4.54× warming** shows just how aggressively cache fills a 31 K-inode pool. Within a single job, A goes from cold-ish to warm.
- **B is roughly stable across passes** (15.75 ms — only one pass measured, but consistent with the warm regime starting state). B's 1.35 K-inode pool was warm from the start because creation just happened.
- **Production implication is now nuanced**:
  - At production scale (213 G store), _neither_ layout fits in cache → both stay in the cold regime
  - In the cold regime, per-open cost is ~9–23 ms (synth vs real, Runs 14/16/18) regardless of pool size or layout
  - File count per sample drops from 41 → 2 with merging → **~20× pure file-count-driven speedup at production scale**
  - This 20× is _less_ than Run 16's 31× (which had cache-fit bonus) but _more_ than Run 19's 7× (which had remaining cache-fit penalty for the larger A pool)
- **Lever 3 (variables merging) remains a strong win at production scale — but the realistic expected speedup is ~15–20×, not 30×.** The dominant savings come from collapsing 41 cold metadata round-trips → 2, not from any pool-size cache effect.

## Run 21 — ABAB on Run 16 stores (B drift check)

### Jobs

- `41599727` — `profile_dataloader_sim_abab.py`. Reuses Run 16's stores at
  `/gpfs/scratch/bsc32/bsc096444/dataloader_sim/`. Read pattern: A1 → B1 →
  A2 → B2, different seed per pass. The added B2 pass isolates whether
  Run 19's single B measurement was sampling a warming-up state.
  CSV: `jobs-map/41599727/dataloader_sim_abab.csv`.

### Interesting values

Per-sample wall ms (12 workers × 100 samples each pass):

| store | pass | n     | mean   | p50    | p90    | p95    | p99    | max     |
| ----- | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------- |
| A     | 1    | 1 200 | 668.03 | 650.30 | 790.14 | 833.46 | 922.78 | 2 822.7 |
| B     | 1    | 1 200 | 88.47  | 85.85  | 126.98 | 140.92 | 177.56 | 1 200.3 |
| A     | 2    | 1 200 | 536.07 | 521.20 | 643.22 | 685.56 | 792.27 | 2 553.5 |
| B     | 2    | 1 200 | 81.74  | 79.13  | 121.54 | 138.40 | 173.79 | 2 141.4 |

**Drift across passes**:

- A1 / A2 = **1.25×** (matches Run 19's 1.28× — A is cold-cache-stable)
- B1 / B2 = **1.08×** (NEW: **B is also stable — only 8% drift**)

**Ratios A/B**:

- A1 / B1 = 7.55×
- A2 / B2 = 6.56×
- A1 p95 / B1 p95 = 5.91×
- A2 p95 / B2 p95 = 4.95×

### Quick conclusions

- **B's measurement is stable across passes** (B1 → B2 = +8%). Run 19's single B reading was _not_ a warming-up artifact — B's cold-cache state is reached fast and then holds. So the 7× cold-cache ratio is robust, not a measurement glitch.
- **A's small drift (1.25×) is reproducible across Runs 19 + 21**. Random 4 % sampling per A pass means most reads stay cold; some accidentally hit warm files but not enough to disturb the regime.
- **Cold-cache A/B sits in the 5–8× range across all percentiles** (means 7.55× → 6.56×; p95 5.91× → 4.95×). Production speedup will be in this range or slightly higher because production's inode count (3.2 M for current, ~200 K for merged) keeps the per-open cost in the cold regime that Run 14 showed converges to ~22 ms p95 regardless of pool size.
- **Production estimate finalized**: 41 → 4 opens per sample × ~22 ms cold = **~810 ms saved on the I/O slice per sample (~5–7× speedup on read; ~3–4× on full `__getitem__` after counting xarray overhead reduction)**. Lever 3 commitment is justified.

## Run 22 — SingleZarr vs MergedZarr DataLoader throughput (2-group merged store)

### Jobs

- `41611662` — `profile_single_vs_merged.py` against the new 2-group merged
  store at `/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr`
  (dynamic = 17 vars + static = 6 vars, categoricals stored as float32 and
  cast back at read time). Both loaders run with `inputs/debug.yaml`,
  `batch_size=8`, `num_workers=12`, `prefetch_factor=4`, `shuffle=True`,
  100 batches each. Single runs first, then merged — caveat: merged may
  benefit from residual page cache from single's pass. CSV at
  `jobs-map/41611662/single_vs_merged.csv`. First end-to-end performance
  test of Lever 3 on a production-shape store.

### Interesting values

Warm-batch stats (batches 1–99, excludes cold batch 0):

| metric                  |      single |       merged | ratio (s/m) |
| ----------------------- | ----------: | -----------: | ----------: |
| Cold batch (0)          |   16 334 ms |     3 493 ms |       4.68× |
| Warm mean               |    1 548 ms |       360 ms |   **4.30×** |
| Warm median             |     0.13 ms |      0.12 ms |         ≈1× |
| Warm p95                |   10 041 ms |     2 315 ms |       4.34× |
| Warm p99                |   12 318 ms |     5 891 ms |       2.09× |
| Warm max                |   20 060 ms |     8 359 ms |       2.40× |
| Warm sum (99 batches)   |     153.3 s |       35.6 s |       4.30× |
| Throughput (warm)       | 5.17 samp/s | 22.23 samp/s |   **4.30×** |
| Fast batches (<10 ms)   |    75 of 99 |     82 of 99 |           — |
| Spike batches (>100 ms) |    24 of 99 |     17 of 99 |           — |
| Mean spike size         |    6 387 ms |     2 095 ms |       3.05× |

Spike indices in the first 60 batches:

- **single**: `[0, 1, 5, 12, 13, 14, 24, 25, 26, 29, 36, 37, 38, 41, 48, 50, 53]` —
  clear 12-batch cycle with consecutive-burst clusters at ~12, ~24, ~36, ~48.
- **merged**: `[0, 1, 2, 7, 8, 11, 15, 17, 19, 20, 32, 40, 44, 50]` —
  weaker cycle; bursts are shorter and more scattered.

### Quick conclusions

- **4.30× end-to-end throughput speedup on the bundled-vs-merged switch.**
  Lands in the 3–4× band predicted by Run 21 (cold per-open count drops
  from 41 → 2; the extra fraction over 4× likely from xarray-overhead
  reduction or residual page-cache benefit for merged — see caveat).
- **The 12-batch worker rotation is still present in both** (spikes cluster
  around multiples of 12 in single, less crisply in merged). Lever 3
  attacks spike _amplitude_, not the structural cycle — mean spike size
  6.4 s → 2.1 s (3.05×). Fast batches (<10 ms) remain 75–82/99 in both;
  the prefetch queue drains for the median, the real work happens at the
  spikes.
- **Run-order caveat.** Single runs first, merged runs second. Page-cache
  contamination across the two paths (`ehpc736/data/...` vs
  `bsc32/bsc096444/...`) should be ~zero — different filesets, different
  files — but merged's bytes may still be in inode/dentry cache from the
  preceding merge job (41610083) on this node. Swap-order rerun before
  propagating the 4.3× to `knowledge.md`.
- Cold-batch ratio (4.68×) tracks warm ratio (4.30×) closely → worker
  startup amortization isn't where the win comes from.

## Run 23 — Chunk-size parity: 512×512 vs 256×256 (SingleZarrDataset, no data change expected)

### Jobs

- `41605409` — `test_loader_parity_chunksize.py`. Same `SingleZarrDataset`,
  same `inputs/debug.yaml`, 32 random samples drawn against each store.
  - 512×512: `/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr`
  - 256×256: `/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-256x256-data-2000-2015.zarr`
  - CSV: `logs/tests/parity_chunksize_41605409.csv` (320 rows = 32 samples × 5 tensors × 2 stores).

### Interesting values

Per-sample bit-equality across all 5 tensors (`x_continuous`, `x_kg`,
`x_static`, `x_hilda_prior`, `hilda_target`):

- **1 120 stat-comparisons (32 samples × 5 tensors × 7 stats {mean, std, min,
  max, n_unique, p25, p75}), 0 mismatches** at 1e-10 tolerance.
- Averaged-across-samples summary identical between `chunk_512` and `chunk_256`
  for every tensor (mean, median, std, min-range, max-range all match).
- Categorical-tensor union sets identical:
  - `x_kg` → 29 distinct values (0..30 except 20, 24) in both stores
  - `x_hilda_prior`, `hilda_target` → {0..7} in both stores
- Sentinel-detection clean for both stores (no `100` leak in KG/population,
  no `>20` z-score outliers in static).

### Quick conclusions

- **The 256×256-chunked store serves bit-identical samples to the 512×512
  store** under `SingleZarrDataset`. The chunk-size move has zero
  data-level effect; both layouts are interchangeable inputs to any
  subsequent perf or training comparison.

## Run 24 — Single vs Merged: Phase 1 + Phase 2 sweep (proper methodology)

### Jobs

- `41618560` — `profile_single_vs_merged.py` rewritten to mirror
  `profile_dataloader.py`'s Phase 1 + Phase 3 shape (no
  `TimedSingleZarrDataset` subclass, no per-modality timers).
  - Phase 1: 50 single-thread `__getitem__` samples per dataset
  - Phase 2: workers ∈ {0, 1, 4, 12} × 80 batches per (dataset, workers),
    each in a spawn child with SIGALRM 30 s per-batch watchdog
  - Interleaved per worker count
  - Same stores as Run 22 (`debug.yaml`, merged store at
    `/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr`)
  - CSV: `jobs-map/41618560/single_vs_merged.csv`

### Interesting values

Phase 1 (single-thread `__getitem__`, 50 samples each):

| dataset |         mean |       median |       p95 |       max |
| ------- | -----------: | -----------: | --------: | --------: |
| single  |    1411.4 ms |    1253.1 ms | 1793.2 ms | 5329.9 ms |
| merged  | **201.1 ms** | **141.6 ms** |  210.2 ms | 2947.0 ms |

**Phase 1 median ratio (single/merged) = 8.85×; mean ratio = 7.02×.**

Phase 2 sweep (warm-batch throughput, excludes cold batch 0):

| num_workers | single (samp/s) | merged (samp/s) |     ratio |
| ----------: | --------------: | --------------: | --------: |
|           0 |            0.71 |            3.44 |     4.85× |
|           1 |            0.70 |            3.95 |     5.64× |
|           4 |            2.47 |           24.62 | **9.97×** |
|          12 |            8.46 |       **39.76** |     4.70× |

Per-config notes:

- Single w=0 ran only 28/80 batches before the 30 s per-batch timeout fired
  on the slow batches; w=1 ran 65/80. Throughput numbers above are over
  the batches that completed.
- Single w=12 reached 8.46 samp/s — within run-to-run variance of the
  9.6 samp/s historical Run 4 baseline (so single is NOT anomalously slow
  vs history; Run 22's 5.17 was just a noisy point).
- Merged w=12 cold batch 1655 ms vs single's 12876 ms (7.8× faster cold
  startup — workers initialize faster against the 2-group store).

Worker scaling (w=0 → w=12):

| dataset |  w=0 |  w=12 | scaling | of ideal 12× |
| ------- | ---: | ----: | ------: | -----------: |
| single  | 0.71 |  8.46 |  11.95× |        99.6% |
| merged  | 3.44 | 39.76 |  11.55× |        96.2% |

Effective per-sample work at w=12 (per-worker, derived from throughput):

| dataset | Phase-1 single-thread | w=12 per-worker effective | DataLoader overhead / sample |
| ------- | --------------------: | ------------------------: | ---------------------------: |
| single  |               1411 ms |                   1418 ms |                        ~7 ms |
| merged  |                201 ms |                    302 ms |                  **~101 ms** |

### Quick conclusions

- **The 7× per-sample speedup is real** and matches knowledge.md's "5-7×
  on read" prediction. Run 22's 4.30× was the **end-to-end DataLoader
  throughput** at w=12, which is compressed by a ~100 ms fixed
  per-sample DataLoader main-process overhead (collate + pin_memory). The
  same ~100 ms is 7% of single's 1411 ms budget but **50% of merged's
  201 ms budget**, so the throughput ratio compresses from 7× → ~4.7×.
- **Lever 3 delivers.** Merged at w=12 = **39.76 samp/s = 4.14× the
  historical 9.6 baseline**. This is the production headline.
- **`w=4` is the sweet spot for merged today**: 24.62 samp/s with only 4
  workers; going to w=12 buys 1.6× for 3× more workers, indicating
  DataLoader main-process overhead is starting to clip the gain.
- **Worker scaling is essentially perfect for both datasets** (99.6%,
  96.2% from w=0 → w=12). No GPFS-contention regime visible at 12
  concurrent workers.
- **The new bottleneck for merged is DataLoader main-process overhead**
  (~100 ms / sample), not the loader's `__getitem__`. To push merged
  past 40 samp/s we need to either lift per-sample work further (bypass
  xarray — could cut Phase-1 from 200 → 100 ms?) or attack the
  DataLoader main process (collate / pin_memory / queue).
- **H2 (single anomalously slow) debunked.** Phase 1 single mean 1411 ms
  matches the Run-4 baseline 1067 ms within run noise (Run 8 reproduce
  was 1222 ms). Single is not anomalous; Run 22 was just a noisier
  single point.

### Decomposition of the 201 ms merged Phase-1 floor (xarray-scheduler hypothesis)

Goal: explain where the 201 ms / 142 ms (mean / median) per `__getitem__`
goes, and predict the lower bound if we bypass xarray.

**Input measurements (all reused; not introduced here):**

| Quantity                                  |     Value | Source                                                                                                                                                        |
| ----------------------------------------- | --------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-chunk xarray graph cost (cold + warm) | **16 ms** | knowledge.md item 72 (profile_layers, jobs 41543368 / 41545383)                                                                                               |
| Per-chunk cold raw GPFS read              |  19–20 ms | same                                                                                                                                                          |
| Per-chunk blosc decode                    |    0.3 ms | same                                                                                                                                                          |
| Per-chunk zarr index (warm)               |     ~0 ms | same                                                                                                                                                          |
| Per-`.load()` "residual" (single's LUH2)  |     80 ms | knowledge.md item 72, attributed to `to_stacked_array + transpose + reshape + loop overhead` (i.e. **in-memory post-processing**, not dask scheduler kickoff) |
| Patch size                                |   256×256 | config                                                                                                                                                        |
| Dynamic spatial chunk size                |   512×512 | merge_zarr_variables.py                                                                                                                                       |
| Static spatial chunk size                 |   512×512 | merge_zarr_variables.py                                                                                                                                       |

**Chunk-straddling expectation** (uniform random `y0`, `x0` in valid range):

For a 256×256 patch in a 512×512 chunk grid:

- `P(no straddle in lat)` = `P(y0 mod 512 ≤ 256)` = **0.5**
- Same for lon → `P(patch fits in 1 chunk)` = **0.25**
- `P(2 chunks)` = 0.5; `P(4 chunks)` = 0.25
- `E[chunks per modality call]` = `0.25 × 1 + 0.5 × 2 + 0.25 × 4` = **2.25**

**Per-sample chunk count for merged (config: `timesteps_target=[0,1]`, `timesteps_dynamic=[0]`, `timesteps_hilda=[1]` ⇒ `_compute_dynamic_timesteps` = {t, t+1} = 2 timesteps):**

| Call                        | timesteps | spatial chunks | chunks per call |
| --------------------------- | --------: | -------------: | --------------: |
| `_read_dynamic`             |         2 |           2.25 |        **4.50** |
| `_process_static`           |         — |           2.25 |        **2.25** |
| **Total chunks per sample** |           |                |        **6.75** |

**Cost model fit to the 201 ms mean:**

| Component                                         |      ms | Calculation                                                                                                                                                                 |
| ------------------------------------------------- | ------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| xarray graph build (cache-invariant)              | **108** | 6.75 chunks × 16 ms                                                                                                                                                         |
| Cold raw I/O (with GPFS prefetch)                 |  **54** | 6.75 chunks × ~8 ms (server-side prefetch absorbs the per-open delta; 8 ms is the empirical effective rate from knowledge.md's "107 ms / 14 chunks" decomposition for LUH2) |
| In-memory work (cast + slice + stack + transpose) | **~30** | extrapolated from single's 80 ms residual (merged does less because vars are already stacked)                                                                               |
| Blosc decode                                      |  **~2** | 6.75 chunks × 0.3 ms                                                                                                                                                        |
| **Modeled total**                                 | **194** |                                                                                                                                                                             |
| **Observed mean (Run 24 Phase 1)**                | **201** |                                                                                                                                                                             |
| **Observed median (Run 24 Phase 1)**              | **142** |                                                                                                                                                                             |

Modeled 194 ms vs observed 201 ms (mean) — agreement within ~4%. Median
142 ms is below the model because the median patch hits fewer chunks
than the 2.25 expectation (the distribution is bimodal: 25% of patches
hit 1 chunk per modality).

**Where the xarray floor name comes from:**

Of the 194 ms modeled total, **108 ms (56%) is per-chunk xarray graph
build** — pure CPU under the GIL, cache-invariant (verified by
profile_layers showing the `(d-c)` delta is the same warm and cold).
This 108 ms is what we'd eliminate by bypassing xarray and using the raw
zarr API (which releases the GIL during Blosc decode).

**Prediction if we bypass xarray:**

| Component            | bypass-xarray model |
| -------------------- | ------------------: |
| Raw zarr read (cold) |  ~54 ms (unchanged) |
| Blosc decode         |   ~2 ms (unchanged) |
| In-memory work       |  ~30 ms (unchanged) |
| **Bypass total**     | **~86 ms / sample** |

With Phase-1 → ~86 ms / sample and the observed 96.2% worker-scaling
efficiency, a w=12 throughput projection: `12 / (0.086 + 0.100 DataLoader overhead) ≈ 65 samp/s`
(vs current 39.76 samp/s) — provided the DataLoader main-process
overhead doesn't grow.

**This decomposition is a hypothesis, not a measurement.** The 16 ms
per-chunk xarray cost is measured (profile_layers); the 8 ms effective
cold I/O is measured; the 30 ms in-memory estimate is extrapolated. To
turn this from hypothesis to measurement, instrument `_read_dynamic`
with `time.perf_counter` around (a) `isel`, (b) `.load()`, and (c)
post-processing — or run an A/B with `chunks=None` to switch xarray
from dask-backed to numpy-backed and see if the floor drops to ~86 ms.

**Update from Run 25 (job 41635364) breakdown measurement.** The cost
model above is correct at the totals level (~200 ms) but the
per-component split was rough. Direct measurement shows the per-sample
budget actually splits as:

| component                                         | median ms |        % |
| ------------------------------------------------- | --------: | -------: |
| `_read_dynamic_to_numpy` (compute + I/O + xarray) |        90 |      51% |
| `_process_static` (compute + I/O + xarray)        |        60 |      34% |
| `_build_continuous_sequence` (pure in-memory)     |        15 |       9% |
| `_read_dynamic_isel` (lazy graph build only)      |       1.4 |       1% |
| other extracts / parse                            |        <1 |      <1% |
| **total**                                         |  **~169** | **~96%** |

Surprises:

- `_read_dynamic_isel` is **essentially free** (1.4 ms median). The
  xarray-scheduler-kickoff component of the floor isn't in the lazy
  graph build — it's inside `.to_numpy()`, where I/O + dask compute +
  the cache-invariant 16 ms-per-chunk xarray-graph work all happen.
- `_process_static` is **34% of the per-sample budget** (60 ms median).
  It's actual xarray + I/O work, not pure in-memory; the earlier
  decomposition's "30 ms in-memory" lumped this into the wrong bucket.
- The actual in-memory floor is just `_build_continuous_sequence` ≈
  **15 ms** (not 30 ms as extrapolated).

Implication for Lever 1 (bypass xarray): it must attack **both**
`_read_dynamic` and `_process_static` — leaving static on xarray gives
back 34% of the budget. With raw zarr on both, expected savings ≈
(132 + 65) × ~70% ≈ ~138 ms → merged `__getitem__` could land at
**~35-45 ms** (vs 175 ms median today). That would be ~4× per-sample
gain on top of Lever 3, not the ~2.3× the Run 24 model predicted.

## Run 25 — Merged `__getitem__` per-method breakdown

### Jobs

- `41635364` — `profile_merged_breakdown.py`. Wraps every helper method
  with `time.perf_counter`; splits `_read_dynamic` into the `isel`
  (lazy graph build) step and the `.to_numpy()` (compute + I/O +
  materialize) step. 100 samples, single-thread, `debug.yaml`. CSV at
  `jobs-map/41635364/merged_breakdown.csv`.

### Interesting values

Per-phase wall ms across 100 samples:

| phase                        |  mean | median |   p95 |    max |
| ---------------------------- | ----: | -----: | ----: | -----: |
| `__getitem__` (whole)        | 215.4 |  175.9 | 303.1 | 3070.3 |
| `_parse_index`               |  0.01 |   0.01 |  0.01 |   0.01 |
| `_read_dynamic_isel`         |  1.43 |   1.41 |  1.56 |   2.33 |
| `_read_dynamic_to_numpy`     | 132.5 |   90.4 | 177.4 | 2958.7 |
| `_process_static`            |  65.0 |   60.5 | 123.8 |  165.8 |
| `_build_continuous_sequence` |  14.9 |   14.9 |  15.8 |   16.1 |
| `_extract_kg_sequence`       |  0.08 |   0.08 |  0.09 |   0.09 |
| `_extract_hilda_prior`       |  0.27 |   0.26 |  0.29 |   0.40 |
| `_extract_hilda_target`      |  0.40 |   0.39 |  0.45 |   0.46 |

Sum-of-parts / total: **99.6% (mean)**, **95.5% (median)** — no
unaccounted overhead.

### Quick conclusions

- **`_read_dynamic_isel` is essentially free (1.4 ms).** The
  xarray-scheduler-kickoff hypothesis from Run 24 was wrong at the
  decomposition level — the expensive xarray work happens inside the
  compute step (`.to_numpy()`), not the lazy `isel`.
- **`_process_static` is 34% of the per-sample budget (60 ms median).**
  It's actual xarray + I/O work, not pure in-memory. **Lever 1 must
  target both `_read_dynamic` and `_process_static`** — leaving static
  on xarray gives back a third of the gain.
- **In-memory work is only ~15 ms** (`_build_continuous_sequence`), not
  ~30 ms as the Run 24 model extrapolated. All the other helpers
  combined are <1 ms.
- **Revised Lever 1 prediction**: bypass-xarray on both dynamic and
  static could land merged `__getitem__` at **~35-45 ms** (vs 176 ms
  median now) → **~4× per-sample gain** on top of Lever 3. Previous
  ~2.3× estimate was an underestimate because the model lumped static
  with "in-memory".
- The Run 24 cost model (108 ms xarray + 54 ms I/O + 30 ms in-memory +
  2 ms decode) is still correct at the **totals** level, but the
  per-component split is now: ~152 ms in two zarr/xarray reads (dynamic
  - static) + ~15 ms in-memory + ~1.5 ms in lazy graph build + others
    negligible.

## Run 26 — Single vs Merged: Phase 1 + Phase 2 rerun (verification of Run 24)

### Jobs

- `41636051` — `profile_single_vs_merged.py` rerun with identical
  config to Run 24 (debug.yaml, batch=8, prefetch=4, workers
  [0, 1, 4, 12], Phase 1 = 50 single-thread samples per dataset,
  Phase 2 = 80 batches per (dataset, workers)). CSV at
  `jobs-map/41636051/single_vs_merged.csv`.

### Interesting values

Phase 1 (single-thread `__getitem__`):

| dataset |         mean |       median |       p95 |       max |
| ------- | -----------: | -----------: | --------: | --------: |
| single  |    1385.0 ms |    1291.9 ms | 1817.7 ms | 4101.0 ms |
| merged  | **193.5 ms** | **178.8 ms** |  310.7 ms |  324.8 ms |

**Phase 1 median ratio (single/merged) = 7.22×; mean ratio = 7.16×** —
consistent with Run 24's 8.85× / 7.02× within run noise.

Phase 2 sweep (warm-batch throughput):

| num_workers | single (samp/s) | merged (samp/s) | ratio (Run 26) | ratio (Run 24) |
| ----------: | --------------: | --------------: | -------------: | -------------: |
|           0 |            0.75 |        **5.83** |          7.77× |          4.85× |
|           1 |            0.83 |        **5.92** |          7.13× |          5.64× |
|           4 |            3.25 |       **25.29** |          7.78× |          9.97× |
|          12 |            8.71 |       **70.47** |      **8.09×** |          4.70× |

Single is stable across runs (8.46 → 8.71 samp/s at w=12). **Merged at
w=12 jumped from 39.76 to 70.47 samp/s (+77%).**

Effective per-sample at w=12 (per-worker, derived from throughput):

| dataset | Phase 1 single-thread | w=12 per-worker effective |        gap |
| ------- | --------------------: | ------------------------: | ---------: |
| single  |               1385 ms |                   1378 ms |      −7 ms |
| merged  |                194 ms |                **170 ms** | **−24 ms** |

Workers are _faster_ per sample than Phase 1 single-thread (likely
inter-worker page-cache sharing). Worker scaling efficiency is
**essentially 100%** in both directions.

Cold batches (worker startup + first prefetch fill) are stable across
the two runs (single ~12 s, merged ~1.7 s). The change between Run 24
and Run 26 is **in the warm-batch spike sizes**: mean warm spike for
merged dropped from ~2095 ms (Run 24) to ~530 ms (Run 26). Workers
refill the prefetch queue faster, fewer queue-drain stalls.

### Quick conclusions

- **Lever 3 per-sample speedup is robust at ~7×** (Phase 1 ratio of
  7.22× this run, 8.85× last run). Independent of cache state.
- **Merged at w=12 = 70.47 samp/s here** — 8.09× the historical 9.6
  baseline, vs Run 24's 4.14×. The variance is in the throughput at
  saturated workers, not in the per-sample work.
- **The "~100 ms DataLoader main-process overhead" hypothesis from
  Run 24 was at least partly a cold-cache artifact.** With the merged
  store having been read multiple times in between (Run 25 +
  chunk-size parity), this run shows worker-effective-per-sample
  (170 ms) ≈ Phase 1 single-thread (194 ms), so the overhead is ≤ 0 ms
  (workers actually benefit from cache sharing). The `pin_memory=False`
  A/B probe is therefore deprioritised — there is no big invariant
  overhead to chase at the DataLoader-main layer.
- **Likely cause of the Run-24-vs-Run-26 throughput delta is GPFS /
  page-cache state.** The merged store has now been read enough times
  that its chunks are warm. Cold batches are unchanged; the spike
  batches got shorter.
- **The bypass-xarray prediction (~4× extra per-sample) stands** —
  Run 25's per-method breakdown is the basis for it, and that's a
  single-thread measurement that's robust to cache state at the
  worker-scaling level.

## Run 27 — Debug training comparison: single vs merged (3 reps each)

### Jobs

Six jobs, two parallel sequential streams (cannot run same-store concurrently per AGENTS.md). **First pass (41651883-41651888) cancelled after rep 1 to add `log_loss_every_batch: true` for per-batch timing. Resubmitted as 41652686-41652691.**

| job ID     | dataloader | rep | depends on |
| ---------- | ---------- | --- | ---------- |
| `41652686` | single     | 1   | —          |
| `41652687` | single     | 2   | 41652686   |
| `41652688` | single     | 3   | 41652687   |
| `41652689` | merged     | 1   | —          |
| `41652690` | merged     | 2   | 41652689   |
| `41652691` | merged     | 3   | 41652690   |

Configs `inputs/debug-{single,merged}.yaml`, identical except `dataset_type`
and `stores`. Both use 4-GPU DDP (`acc_training.sh`), `num_workers=12`,
`batch_size=8`, `prefetch_factor=4`, `epochs=1`,
`train_steps_per_epoch=15`, `val_steps_per_epoch=1` (matches
`profiling.yaml`). First end-to-end training comparison after Lever 3 —
includes model forward/backward and validates how much of the
dataloader-level speedup translates with the model in the loop.

### Interesting values

Per-job wall times (from unet training logs):

| job ID          |      rep | train wall |   val wall |       total |
| --------------- | -------: | ---------: | ---------: | ----------: |
| `41652686`      | single 1 |     83.4 s |     66.9 s |     150.3 s |
| `41652687`      | single 2 |     69.2 s |     55.4 s |     124.7 s |
| `41652688`      | single 3 |     49.4 s |     32.2 s |      81.6 s |
| `41652689`      | merged 1 |     18.1 s |     10.4 s |      28.5 s |
| `41652690`      | merged 2 |     15.1 s |      5.2 s |      20.4 s |
| `41652691`      | merged 3 |     10.3 s |      4.4 s |      14.8 s |
| **mean single** |          | **67.3 s** | **51.5 s** | **118.9 s** |
| **mean merged** |          | **14.5 s** |  **6.7 s** |  **21.2 s** |
| **ratio**       |          |  **4.64×** |  **7.69×** |   **5.61×** |

Per-batch deltas from per-batch log timestamps (`log_loss_every_batch:
true`). B1 (cold) excluded — shown as B2..B15 sequential step delta in
seconds. Bold = clear "refill spike":

single:

| batch |    rep 1 |    rep 2 |   rep 3 |
| ----: | -------: | -------: | ------: |
|     2 | **13.4** |     0.16 |    0.16 |
|     3 |     0.15 |     0.15 |    0.15 |
|     4 |  **6.4** |  **3.7** | **1.6** |
|     5 |     0.18 |     0.16 |    0.15 |
|     6 |     0.16 |     0.15 |    0.15 |
|     7 |     0.16 |     0.15 |    0.15 |
|     8 |     0.16 |     0.16 | **4.1** |
|     9 |     0.15 |     0.16 |    0.16 |
|    10 |     0.15 |     0.15 |    0.15 |
|    11 |     0.15 |     0.15 |    0.15 |
|    12 | **27.8** | **14.8** | **5.7** |
|    13 |     0.16 |     0.16 |    0.16 |
|    14 |     0.15 |     0.15 |    0.15 |
|    15 |     0.15 |     0.15 |    0.16 |

merged:

| batch | rep 1 | rep 2 |    rep 3 |
| ----: | ----: | ----: | -------: |
|     2 |  1.62 |  0.24 |     0.20 |
|     3 |  0.15 |  0.22 |     0.18 |
|     4 |  2.33 |  0.30 | **2.18** |
|     5 |  0.19 |  0.17 |     0.21 |
|     6 |  0.19 |  0.16 |     0.17 |
|     7 |  0.15 |  0.16 |     0.18 |
|     8 |  0.20 |  0.21 |     0.20 |
|     9 |  0.15 |  0.18 |     0.27 |
|    10 |  0.15 |  0.20 |     0.18 |
|    11 |  0.15 |  0.16 |     0.23 |
|    12 |  0.17 |  0.17 |     0.20 |
|    13 |  0.16 |  0.15 |     0.18 |
|    14 |  0.15 |  0.17 |     0.17 |
|    15 |  0.16 |  0.15 |     0.19 |

### Quick conclusions

- **End-to-end training with the model in the loop: ~4.64× faster
  training, ~7.69× faster validation, ~5.61× faster total wall.** Single
  vs merged with identical model / hyperparameters, `train_steps_per_epoch=15`,
  4-rank DDP via accelerate, 4-accumulation, batch=8, num_workers=12.
- **Per-batch logging makes the spike pattern unmistakable.** Single
  shows the worker-cycle refill at B12 in every rep; merged shows
  _no_ B12 spike in any rep. Lever 3 doesn't just speed the dataloader —
  it **eliminates the periodic stall pattern** that was the dominant
  cost of historical single-zarr training (per batch_times.txt and
  Runs 1-11).
- **Cache warming is a major factor.** Both dataloaders speed up
  ~1.85× from rep 1 → rep 3 because GPFS / page-cache state warms
  across consecutive jobs on the same store. The single-vs-merged
  ratio holds across reps (~5×), even though absolute numbers shift.
- **Merged spike magnitudes shrink to ≤ 2.3 s and only land at B4**
  (and not always). Single's B12 spike is 5.7-27.8 s across reps —
  shrinking with cache warm-up but still 30× bigger than any merged
  spike.
- **Training-loop reality is closer to the 5-8× per-sample dataloader
  ratio than to Run 24's 4.7×.** The model overlap with the dataloader
  helps merged turn its dataloader headroom into compute parallelism,
  while single just stalls on the refill spikes.
- **The model is _not_ yet the bottleneck for merged training at this
  config**, despite earlier suspicion. If it were, the train/val wall
  would be capped by model time and the ratio between single and
  merged would compress. Instead the ratio stays at ~5×, meaning the
  dataloader is still partly on the critical path for single but
  merged hides almost all of it.

## Run 28 — Debug training comparison at 200 steps × 30 val × 30 test (3 reps each)

### Jobs

Same dependency layout as Run 27 (two parallel sequential streams), with
`train_steps_per_epoch=200`, `val_steps_per_epoch=30`,
`test_steps_per_epoch=30` (vs 15 / 1 / 1 in Run 27). All other settings
identical: 4-rank DDP, accumulation=4, num_workers=12, batch=8,
prefetch=4, `log_loss_every_batch: true`. Bigger N to estimate the
steady-state per-batch behavior past the 12-batch worker-cycle period
(now ~16 cycles per epoch instead of ~1.25).

| job ID     | dataloader | rep | depends on |
| ---------- | ---------- | --- | ---------- |
| `41654672` | single     | 1   | —          |
| `41654673` | single     | 2   | 41654672   |
| `41654674` | single     | 3   | 41654673   |
| `41654675` | merged     | 1   | —          |
| `41654676` | merged     | 2   | 41654675   |
| `41654677` | merged     | 3   | 41654676   |

### Interesting values

Per-job wall times:

| job ID          |      rep |  train wall |   val wall |       total |
| --------------- | -------: | ----------: | ---------: | ----------: |
| `41654672`      | single 1 |     325.5 s |     62.3 s |     387.8 s |
| `41654673`      | single 2 |     260.7 s |     62.8 s |     323.4 s |
| `41654674`      | single 3 |     290.9 s |     60.2 s |     351.0 s |
| `41654675`      | merged 1 |      96.5 s |     22.8 s |     119.3 s |
| `41654676`      | merged 2 |      50.5 s |     11.0 s |      61.5 s |
| `41654677`      | merged 3 |      41.6 s |      7.2 s |      48.8 s |
| **mean single** |          | **292.4 s** | **61.8 s** | **354.1 s** |
| **mean merged** |          |  **62.9 s** | **13.7 s** |  **76.5 s** |
| **ratio**       |          |   **4.65×** |  **4.51×** |   **4.63×** |

Per-batch spike pattern (from per-batch log timestamps; "spike" = delta > 1 s):

| dataloader | rep | # spikes / 199 | max spike | spike cadence                                 |
| ---------- | --: | -------------: | --------: | --------------------------------------------- |
| single     |   1 |             17 |    34.4 s | exactly every 12 batches (B4, 16, 28, …, 196) |
| single     |   2 |            ~20 |    14.4 s | ~12-batch but with smaller scattered spikes   |
| single     |   3 |             17 |    20.8 s | exactly every 12 batches                      |
| merged     |   1 |              5 |    11.6 s | irregular: B33, 54, 89, 156, 160              |
| merged     |   2 |              3 |     4.5 s | irregular: B16, 40, 64                        |
| merged     |   3 |              1 |     2.7 s | only B4                                       |

Cache-warming across the 3 reps in each stream:

|                   | rep 1 | rep 2 | rep 3 |         rep 1 / rep 3 |
| ----------------- | ----: | ----: | ----: | --------------------: |
| single train wall | 325 s | 261 s | 291 s | 1.12× (non-monotonic) |
| merged train wall |  96 s |  51 s |  42 s |             **2.32×** |

### Quick conclusions

- **End-to-end speedup with 200 steps: 4.63× total (4.65× train, 4.51× val).** Smaller than Run 27's 5.61× because val is now properly amortized (30 steps vs 1 step). The Run-27 val ratio (7.69×) was inflated by single's prefetch-queue-fill cost. Run 28's ratio is the more honest end-to-end number.
- **Single's 12-batch worker-cycle is undeniable now.** With 200 steps, we see 17 cycles per epoch, and spikes are 10-34 s each (mean ~12 s). The cycle period is `num_workers × prefetch_factor / batch_size = 12 × 4 / 8 = 6`... wait, observed is 12. Likely `prefetch_factor` shifts the queue depth in the way that produces the 12-batch cadence we see. Either way, the pattern is **structural to the single-zarr dataloader**, not a one-off.
- **Merged eliminates the worker-cycle in real training.** Rep 1 has 5 sparse spikes; rep 3 has 1. With a warm cache, merged at this DDP config produces essentially smooth batch arrival.
- **Cache warming is dramatic for merged** (2.32× across 3 reps in ~5 minutes wall) but **noisy for single** (rep 2 faster than rep 3). Single's spikes are dominated by metadata-read cost which warms inconsistently across the 41-files-per-sample pattern.
- **Merged rep 3 is in the steady-state regime**: 42 s for 200 train steps × 4-rank DDP × accumulation=4 = 3200 dataloader batches. That's **~76 batches/s aggregate** (~19 batches/s/rank), comparable to merged at w=12 in Run 26 (70 samples/s = 8.75 batches/s/rank single rank). The 4-rank DDP gets ~2.2× over single-rank — sub-linear (~55%) which is expected with shared GPFS metadata.
- **The 4.6× total-wall ratio holds despite cache warming**, confirming that the per-sample dataloader work is the source of the gain, not just GPFS load.

## Run 29 — DDP-aware worker sweep, 1 node (4 ranks)

### Jobs

- `41654678` — `profile_ddp_sweep.py` via `launch_profile_ddp_sweep_1node.sh`.
  Depends on both single-3 (41654674) and merged-3 (41654677) so GPFS
  is clear of training contention. Single SLURM node, 4 GPUs, 80 cores,
  exclusive (matches `acc_training.sh`). 4 accelerate sub-processes,
  each runs a per-rank DataLoader sweep over `num_workers ∈ {1, 4, 8, 12, 20}`
  for `{single, merged}` × 40 batches/config. Per-rank CSV at
  `jobs-map/41654678/ddp_sweep_rank{0..3}.csv`.

### Interesting values

Per-rank throughput at each (dataset, num_workers); aggregate views
(optimistic = sum-of-rates, DDP-realistic = `min_rate × num_ranks`):

| dataset |  nw | per-rank rates (samp/s) | min thr |   DDP agg | optimistic sum | variance ratio |
| ------- | --: | ----------------------: | ------: | --------: | -------------: | -------------: |
| single  |   1 |         0.6 0.6 0.7 0.6 |     0.6 |       2.4 |            2.5 |           1.1× |
| single  |   4 |         2.1 1.8 2.4 2.2 |     1.8 |       7.3 |            8.5 |           1.3× |
| single  |   8 |         4.5 3.2 4.2 3.2 |     3.2 |      12.8 |           15.0 |           1.4× |
| single  |  12 |         5.5 4.8 4.2 7.1 |     4.2 |      17.0 |           21.7 |           1.7× |
| single  |  20 |       12.9 9.7 9.9 11.7 |     9.7 |      38.9 |           44.3 |           1.3× |
| merged  |   1 |         4.7 3.4 4.1 4.7 |     3.4 |      13.7 |           16.9 |           1.4× |
| merged  |   4 |     10.7 42.5 10.7 24.7 |    10.7 |      42.6 |           88.6 |           4.0× |
| merged  |   8 |     53.8 98.2 39.5 18.9 |    18.9 |      75.5 |          210.4 |           5.2× |
| merged  |  12 |   51.9 115.9 124.8 71.1 |    51.9 | **207.7** |          363.7 |           2.4× |
| merged  |  20 |    31.9 241.1 45.0 10.8 |    10.8 |      43.1 |          328.7 |      **22.4×** |

### Quick conclusions

- **Single is rock-uniform across ranks** (variance 1.1-1.7×) because
  41 files / sample averages out per-chunk variance.
- **Merged has substantial per-rank variance** (4-22×) — with ~6.75
  files / sample, individual cold-chunk hits dominate; random sample
  assignments differ enough to drive very different per-rank rates.
- **DDP-realistic best: merged w=12 at 207.7 samp/s** (single w=20 = 38.9
  → 5.3× advantage). At higher worker counts merged regresses
  (w=20 = 43.1) because the slowest rank gets unluckier on metadata.
- **Worker scaling for single is excellent**: 0.6 → 1.8 → 3.2 → 4.2 → 9.7
  samp/s min rate from w=1 → w=20.
- **For merged, w=12 is the sweet spot**: w=8 (75.5) → w=12 (207.7)
  jumps 2.7×; w=20 collapses to 43.1.

## Run 30 — DDP-aware worker sweep, 2 nodes (8 ranks)

### Jobs

- `41654679` — same script as Run 29 via `launch_profile_ddp_sweep_2node.sh`.
  Depends on 41654678 (1-node sweep must finish first to avoid GPFS
  contention). 2 nodes × 4 GPUs × 20 cores/GPU = 8 ranks total. Tests
  whether per-rank scaling holds when 8 ranks contend for GPFS across
  two nodes. Per-rank CSV at `jobs-map/41654679/ddp_sweep_rank{0..7}.csv`.

### Interesting values

| dataset |  nw | min thr | DDP agg (min × 8) | optimistic sum | variance ratio |
| ------- | --: | ------: | ----------------: | -------------: | -------------: |
| single  |   1 |     0.6 |               4.9 |            5.1 |           1.1× |
| single  |   4 |     3.0 |              24.0 |           24.8 |           1.1× |
| single  |   8 |     5.9 |              47.3 |           49.0 |           1.1× |
| single  |  12 |     6.9 |              55.5 |           62.1 |           1.2× |
| single  |  20 |    16.5 |         **132.2** |          142.1 |           1.2× |
| merged  |   1 |     4.0 |              32.0 |           43.4 |           1.8× |
| merged  |   4 |     9.1 |              72.6 |          172.4 |           4.6× |
| merged  |   8 |    10.3 |              82.5 |          233.9 |           9.3× |
| merged  |  12 |    19.5 |         **156.0** |          334.8 |           5.4× |
| merged  |  20 |    10.9 |              87.3 |          376.9 |          13.8× |

Per-rank wall (merged w=12) shows the dispersion: rank 0 takes 28.3 s
for 40 batches (max single batch 23.7 s) while rank 2 takes 4.0 s — a
7× wall-time spread across ranks. Outlier batches in slow ranks reach
10-23 s.

### Quick conclusions

- **Single scales near-perfectly to 8 ranks** (~93% DDP efficiency,
  super-linear vs 4-rank Run 29 because metadata-server bandwidth
  doubles across 2 nodes): per-rank min rises from 9.7 (1-node w=20)
  to 16.5 (2-node w=20).
- **Merged scales sub-linearly and REGRESSES at 2 nodes vs 1 node**:
  Run 29 1-node merged w=12 hit 207.7 samp/s DDP; Run 30 2-node merged
  w=12 only reaches 156.0. Adding more ranks creates more chances for
  one rank to land on cold-cache samples, and the slowest rank caps
  the whole step.
- **Single at 2-node w=20 (132.2) is now competitive with merged at
  2-node w=12 (156.0)** — the Lever-3 advantage at 8-rank shrinks to
  ~1.2×, down from ~5× at single-rank.
- **The cap on Lever 3 in production DDP is per-rank variance**, not
  the per-sample dataloader work. Bypass-xarray (Lever 1) is now extra
  motivating: shrinking per-chunk cost would also shrink the variance
  envelope.
- **Today's best production config (DDP-realistic): 1-node × merged ×
  w=12 = ~207 samp/s.** Going wider hurts unless we can kill the
  per-rank tail.

## Run 31 — Real-training scale-out validation: rank × workers × dataloader

### Jobs

Real DDP training (5 epochs × 300 train_steps × 10 val × 10 test) used
as a direct measurement of the scaling behavior that Runs 29 + 30
inferred from the synthetic sweep. Bypasses all sweep-methodology
concerns: cache contamination across configs, "min × N" approximation
of DDP throughput, sample-distribution coincidences. 18 jobs total: 2
dataloaders × {1 rank, 4 ranks, 8 ranks} × {8, 12, 20} num_workers.

Two parallel sequential streams (same-store concurrency forbidden):

| job ID     | dataloader | scale | workers | depends on |
| ---------- | ---------- | ----: | ------: | ---------- |
| `41788795` | single     |     1 |       8 | —          |
| `41788796` | single     |     1 |      12 | 41788795   |
| `41788797` | single     |     1 |      20 | 41788796   |
| `41788798` | single     |     4 |       8 | 41788797   |
| `41788799` | single     |     4 |      12 | 41788798   |
| `41788800` | single     |     4 |      20 | 41788799   |
| `41788801` | single     |     8 |       8 | 41788800   |
| `41788802` | single     |     8 |      12 | 41788801   |
| `41788803` | single     |     8 |      20 | 41788802   |
| `41788804` | merged     |     1 |       8 | —          |
| `41788805` | merged     |     1 |      12 | 41788804   |
| `41788806` | merged     |     1 |      20 | 41788805   |
| `41788807` | merged     |     4 |       8 | 41788806   |
| `41788808` | merged     |     4 |      12 | 41788807   |
| `41788809` | merged     |     4 |      20 | 41788808   |
| `41788810` | merged     |     8 |       8 | 41788809   |
| `41788811` | merged     |     8 |      12 | 41788810   |
| `41788812` | merged     |     8 |      20 | 41788811   |

Scripts: `scripts/launch_bench_{1rank,4rank,8rank}.sh` — mirrors
`acc_training.sh` (`--exclusive`, `accelerate launch --multi_gpu`,
production CPU layout) but accepts Hydra overrides for `num_workers`.

### Interesting values

**RUN INVALIDATED BY CROSS-STORE GPFS CONTENTION.** The single and
merged streams ran in parallel as two independent chains. We later
proved that this contaminates p95 batch latency even though the two
streams read different zarr stores. Only a subset of jobs were
"clean" (zero temporal overlap with the other stream):

| Job              | Config                    | Status                                                      |
| ---------------- | ------------------------- | ----------------------------------------------------------- |
| `41788795`       | single 1r×8               | CONTAMINATED — 58% overlap with merged stream               |
| `41788796`       | single 1r×12              | CONTAMINATED — 60% overlap                                  |
| `41788797`       | single 1r×20              | CONTAMINATED — 58% overlap                                  |
| `41788798`       | single 4r×8               | **CLEAN — 0% overlap, usable reference**                    |
| `41788799`       | single 4r×12              | **CLEAN — 0% overlap, usable reference**                    |
| `41788800`       | single 4r×20              | **CLEAN — 0% overlap, usable reference**                    |
| `41788801`       | single 8r×8               | CONTAMINATED — 11% overlap with merged Run 32               |
| `41788802`       | single 8r×12              | CONTAMINATED — 83% overlap with merged Run 32               |
| `41788803`       | single 8r×20              | CANCELLED mid-run after ~9% overlap with merged R32         |
| `41788804`–`812` | merged 1r/4r/8r × 8/12/20 | ALL CONTAMINATED (concurrent with single stream throughout) |

Merged-stream measurements I extracted before realising the
contamination (median over epochs 2–5; samples/s from wall time):

```
Job        cfg      b_med   b_mean  b_p95   sps_wall
41788804   1r×8     0.151   0.213   0.371    37.53
41788805   1r×12    0.152   0.163   0.161    49.09
41788806   1r×20    0.152   0.155   0.166    51.68
41788807   4r×8     0.154   0.209   0.317   153.04
41788808   4r×12    0.156   0.176   0.175   182.08
41788809   4r×20    0.157   0.167   0.180   192.16
41788810   8r×8     0.156   0.201   0.258   318.93
41788811   8r×12    0.157   0.173   0.204   369.83
41788812   8r×20    0.158   0.173   0.205   370.57
```

These numbers should be treated as **lower bounds** for the merged
loader (contention drags p95 up and wall throughput down). Don't
quote them as truth.

### Quick conclusions

- **Methodology lesson**: the AGENTS.md rule "Never run multiple jobs
  that read the same zarr store concurrently" was insufficient — even
  concurrent jobs on _different_ zarr stores contaminate each other via
  shared GPFS bandwidth. Run 33 (clean serialized intercalated sweep)
  is the corrected experiment.
- **Median batch delta is unreliable when the tail is heavy**. At NW=8,
  `samp/s_med` (= batch_size × ranks / b_med) overcounts wall-time
  throughput by 28–41% because the median ignores spikes. Default to
  `sps_wall = (300 × batch_size × ranks) / (epoch_train_time + b_mean)`
  for all future merged/single comparisons.
- **Clean baseline preserved**: Run 31 single 4r × {8, 12, 20}
  (`41788798`–`41788800`) are usable references for cross-checking
  Run 33's 4r-single results.

## Run 32 — Merged sweep reproducibility check (9 jobs)

Re-run the 9 merged jobs from Run 31 to validate reproducibility of the
wall-time-based throughput numbers. Same configs, same scripts, same
zarr store. If reproducible (±5%), Run 31's merged conclusions stand;
if not, GPFS state on a single launch is the dominant noise.

| Job ID     | Ranks | Workers | Depends on |
| ---------- | ----: | ------: | ---------- |
| `41798193` |     1 |       8 | —          |
| `41798194` |     1 |      12 | 41798193   |
| `41798195` |     1 |      20 | 41798194   |
| `41798196` |     4 |       8 | 41798195   |
| `41798197` |     4 |      12 | 41798196   |
| `41798198` |     4 |      20 | 41798197   |
| `41798199` |     8 |       8 | 41798198   |
| `41798200` |     8 |      12 | 41798199   |
| `41798201` |     8 |      20 | 41798200   |

All sequential to avoid concurrent reads on the same zarr store.

### Interesting values

**RUN CANCELLED MID-FLIGHT.** Only 4 of 9 jobs completed before we
detected the contamination and cancelled the rest:

| Job                   | Config                     | Status                           |
| --------------------- | -------------------------- | -------------------------------- |
| `41798193`            | merged 1r×8                | Completed                        |
| `41798194`            | merged 1r×12               | Completed                        |
| `41798195`            | merged 1r×20               | Completed                        |
| `41798196`            | merged 4r×8                | Completed                        |
| `41798197`            | merged 4r×12               | CANCELLED — completed CG cleanup |
| `41798198`–`41798201` | merged 4r×20, 8r×{8,12,20} | CANCELLED before start           |

Run 32 was launched while Run 31's single stream was still working
through its 4r and 8r jobs, which itself was the second source of
cross-store GPFS contention (Run 32 jobs were running concurrently
with single 41788801 / 802 / 803).

Pair-wise vs Run 31 merged equivalents (drop epoch 1, epochs 2–5
batch-delta statistics; `samp/s_wall` from epoch wall time):

```
cfg         metric       Run31      Run32       Δ%
1r×8        b_median     0.151      0.151       +0.0%
            b_mean       0.213      0.204       -4.2%
            b_p95        0.371      0.166      -55.3%
            sps_wall    37.53      39.16       +4.3%

1r×12       b_median     0.152      0.150       -1.3%
            b_mean       0.163      0.942     +477.8%
            b_p95        0.161      6.383    +3864.6%
            sps_wall    49.09       8.50      -82.7%

1r×20       b_median     0.152      0.152       +0.0%
            b_mean       0.155      0.178      +15.3%
            b_p95        0.166      0.165       -0.6%
            sps_wall    51.68      44.83      -13.3%

4r×8        b_median     0.154      0.153       -0.6%
            b_mean       0.209      0.414      +98.0%
            b_p95        0.317      2.063     +550.8%
            sps_wall   153.04      77.28      -49.5%
```

### Quick conclusions

- **Cross-store GPFS contention has a bimodal failure mode**: most
  median batch deltas are unchanged (0.15s steady state survives), but
  the **tail blows up** by 1.5–40× and that crushes wall-time
  throughput. The 1r×12 case dropped from 49 → 8.5 samp/s — workers
  stalling 6.4s on individual batches.
- **Per-rank node assignment by SLURM does NOT save us.** Even with
  `--exclusive` ensuring each job runs on dedicated nodes, the GPFS
  metadata path and shared bandwidth still serialise across the
  cluster.
- **All Run 32 measurements are discarded.** Run 33 replaces Run 31 +
  Run 32 with one clean serialized sweep.

## Run 33 — Clean serialized intercalated sweep (18 jobs)

Run 31 and Run 32 had concurrent-job GPFS contention which inflated p95
batch times and depressed throughput. All 18 jobs are now serialized
into one dependency chain. Order alternates single ↔ merged with each
side independently randomized (seed=42), so no two consecutive jobs
share a zarr store and no cache state persists across data-store
switches. Run 31's single 4r jobs (41788798–800) and 8r×20 (41788803,
borderline) were the only clean readings from the earlier streams; this
run replaces all 18 with one apples-to-apples set.

| Step | Job ID   | Dataset | Ranks | Workers |
| ---: | -------- | :-----: | ----: | ------: |
|    1 | 41799685 | single  |     4 |       8 |
|    2 | 41799686 | merged  |     8 |      12 |
|    3 | 41799687 | single  |     8 |       8 |
|    4 | 41799688 | merged  |     1 |      20 |
|    5 | 41799689 | single  |     8 |      12 |
|    6 | 41799690 | merged  |     4 |      20 |
|    7 | 41799691 | single  |     4 |      12 |
|    8 | 41799692 | merged  |     8 |       8 |
|    9 | 41799693 | single  |     8 |      20 |
|   10 | 41799694 | merged  |     1 |       8 |
|   11 | 41799695 | single  |     1 |      20 |
|   12 | 41799696 | merged  |     4 |       8 |
|   13 | 41799697 | single  |     4 |      20 |
|   14 | 41799698 | merged  |     4 |      12 |
|   15 | 41799699 | single  |     1 |       8 |
|   16 | 41799700 | merged  |     1 |      12 |
|   17 | 41799701 | single  |     1 |      12 |
|   18 | 41799702 | merged  |     8 |      20 |

Analysis convention: drop epoch 1 (cold-cache warm-up), aggregate batch
deltas across epochs 2–5, derive samples/s from epoch wall time (not
median batch delta — median undercounts the tail).

### Interesting values

All 18 jobs reached "Training complete." cleanly. Chain ran 14 Jun 22:55
UTC → 15 Jun 07:01 UTC (≈ 8 h end-to-end wall).

Per-job throughput (epochs 2–5, BS=8, 300 batches/epoch,
`sps_wall = 300 × BS × ranks / (epoch_train_t + b_mean)`):

| Job      | DS     |  r × w | b_med | b_mean |  b_p95 | sps_wall |
| -------- | ------ | -----: | ----: | -----: | -----: | -------: |
| 41799699 | single |  1 × 8 | 0.151 |  1.341 |  8.900 |     5.97 |
| 41799701 | single | 1 × 12 | 0.151 |  0.982 |  8.631 |     8.15 |
| 41799695 | single | 1 × 20 | 0.151 |  0.663 |  4.247 |    12.07 |
| 41799685 | single |  4 × 8 | 0.153 |  2.075 | 12.579 |    15.42 |
| 41799691 | single | 4 × 12 | 0.154 |  1.519 | 11.449 |    21.07 |
| 41799697 | single | 4 × 20 | 0.155 |  1.055 |  7.939 |    30.34 |
| 41799687 | single |  8 × 8 | 0.153 |  1.909 | 11.992 |    33.52 |
| 41799689 | single | 8 × 12 | 0.154 |  1.670 | 11.847 |    38.33 |
| 41799693 | single | 8 × 20 | 0.155 |  1.224 |  8.338 |    52.29 |
| 41799694 | merged |  1 × 8 | 0.151 |  0.199 |  0.164 |    40.18 |
| 41799700 | merged | 1 × 12 | 0.151 |  0.161 |  0.163 |    49.81 |
| 41799688 | merged | 1 × 20 | 0.152 |  0.163 |  0.163 |    49.14 |
| 41799696 | merged |  4 × 8 | 0.153 |  0.299 |  1.232 |   107.00 |
| 41799698 | merged | 4 × 12 | 0.155 |  0.221 |  0.239 |   144.74 |
| 41799690 | merged | 4 × 20 | 0.155 |  0.251 |  0.183 |   127.73 |
| 41799692 | merged |  8 × 8 | 0.153 |  0.359 |  1.670 |   178.27 |
| 41799686 | merged | 8 × 12 | 0.154 |  0.364 |  0.719 |   175.60 |
| 41799702 | merged | 8 × 20 | 0.156 |  0.200 |  0.207 |   319.98 |

(`b_med`, `b_mean`, `b_p95` in seconds; `sps_wall` in samples/s aggregate
across all ranks.)

Run 31 cross-check (clean 4r-single triplet vs Run 33 counterparts):

| Config  | R31 job  | R33 job  | R31 sps_wall | R33 sps_wall |  delta |
| ------- | -------- | -------- | -----------: | -----------: | -----: |
| 4r × 8  | 41788798 | 41799685 |        13.74 |        15.42 | +12.2% |
| 4r × 12 | 41788799 | 41799691 |        20.04 |        21.07 |  +5.1% |
| 4r × 20 | 41788800 | 41799697 |        19.43 |        30.34 | +56.1% |

Per-rank merged throughput (sps_wall / ranks):

| ranks ↓ / NW → |    8 |   12 |   20 |
| -------------- | ---: | ---: | ---: |
| 1              | 40.2 | 49.8 | 49.1 |
| 4              | 26.8 | 36.2 | 31.9 |
| 8              | 22.3 | 21.9 | 40.0 |

DDP scaling efficiency on merged at each config's best NW:

- 1r → 4r: 144.74 / 49.81 = **2.91× (73% of ideal 4×)** at NW=12
- 1r → 8r: 319.98 / 49.14 = **6.51× (81% of ideal 8×)** at NW=20
- 4r → 8r: 319.98 / 127.73 = **2.51× (125% — superlinear)** at NW=20

### Quick conclusions

- **All 18 jobs completed; no in-run failures.** The serialized chain
  delivered a fully apples-to-apples dataset, eliminating Run 31/32's
  cross-store GPFS contention.
- **No 2-node merged regression observed.** At every NW, 8r merged ≥ 4r
  merged in aggregate throughput. The 2-node regression suggested by
  Run 30 (synthetic sweep) does NOT survive when measured under a real
  DDP training loop with all-reduce in the critical path.
- **Lever 3 (variables merging) holds up in real DDP training.** Best
  merged config (8r × 20, 320 sps_wall) is **6.1× faster** than the
  best single config (8r × 20, 52 sps_wall). At matched 1r the gap
  is **4.1×** (49.8 vs 12.1); at matched 4r it is **4.8×** (144.7 vs
  30.3); at matched 8r it is **6.1×**. Lever 3's per-rank gain
  amplifies with rank count.
- **Optimal `num_workers` depends on rank count, contrary to the Run 31
  NW=12 hypothesis.**
  - **merged 1r:** NW=12 ≈ NW=20 (~49 sps), NW=8 lags (40 sps).
  - **merged 4r:** NW=12 best (145 sps) > NW=20 (128) > NW=8 (107).
  - **merged 8r:** NW=20 decisively best (320 sps) >> NW=12 (176) ≈
    NW=8 (178).
  - **single:** NW=20 best at every rank count (lowest b_p95, highest
    sps_wall); NW=8 worst. The structural 12-batch worker stalls
    (b_p95 ≈ 8–13 s) survive across all NW.
- **The 8r × 20 merged result (320 sps_wall) is anomalously high vs
  4r × 20 (128) and 8r × 12 (176).** This was the LAST job in the
  chain (07:01 UTC, ~7 h after start) so the merged store would have
  had maximum prior cache warming. Single-data-point — cannot
  separate "true 8r×20 sweet spot" from "warmest-cache job in the
  chain". Worth confirming.
- **Run 31 cross-check is partly clean (4r×8, 4r×12 within ±5–12%)
  but 4r×20 is +56% in Run 33.** Per `next_session.md` rule, flag
  this. Two plausible explanations: (a) GPFS state on a single launch
  is genuinely the dominant noise floor at NW=20; (b) NW=20 was
  itself the unreliable config in Run 31's noisier environment. Does
  not invalidate the conclusion about no 2-node regression, since
  that conclusion is from internal Run 33 comparisons, not cross-run.

---

## Run 34 — NW=20 anomaly re-verification (3-job chain)

Targeted rerun of the three Run-33 configs that flagged as suspicious:
the anomalously high merged 8r×20 (last-in-chain, max cache warming),
the merged 4r×20 < 4r×12 inversion, and the single 4r×20 that came in
+56% vs Run 31. Serialized chain to keep AGENTS.md no-concurrent-store
rule in force; intercalated single/merged so cache state doesn't
persist across each job.

### Jobs

- `41811414` — merged 8r×20 (re-verify 319.98 sps_wall)
- `41811415` — single 4r×20 (resolve R31 vs R33 +56% mismatch)
- `41811416` — merged 4r×20 (verify NW=12 > NW=20 at 4r ranks)

Chain order: `41811414 → 41811415 → 41811416`, each
`--dependency=afterany`.

### Interesting values

Only `41811414` (merged 8r×20) completed; `41811415` + `41811416` were
cancelled mid-flight (never produced results).

### Quick conclusions

**Discarded — superseded by Run 35.** Per `next_session.md`, all three Run-34
jobs are dropped; the 90-job 5-rep chain (Run 35) is the authoritative
re-verification. Do not use Run 34 numbers.

## Run 35 — 5-rep validation of the 18-config DDP sweep (90-job chain)

The statistically-defensible rerun of the Run-33 sweep. 5 seeds (42–46), each
independently shuffling the 9 `(ranks, num_workers)` configs from
`{1,4,8} × {8,12,20}` per dataset, alternating merged ↔ single so no two
consecutive jobs read the same store. Single `--dependency=afterany` chain,
jobs `41825117`–`41825214`, tag file `jobs-map/chain_1781521500_tags.tsv`.
Config: `debug-{single,merged}.yaml` (epochs=5, batch=8, accumulation=4,
train/val/test steps = 300/10/10, patch 256). Built with
`scripts/build_run_chain.py`; analysed with `scripts/build_run_csv.py` +
`scripts/compare_runs.py`. Canonical metric: `sps_wall` (samples/s from
per-batch deltas + per-epoch train time).

### Jobs

- `41825117`–`41825214` — 90-job chain (18 configs × 5 seeds), single afterany chain.
- **86 of 90 usable.** 4 dropped:

  | Job        | Config       | Seed | State           | Elapsed | Disposition                                                                               |
  | ---------- | ------------ | ---- | --------------- | ------- | ----------------------------------------------------------------------------------------- |
  | `41825176` | merged 8r×8  | 44   | FAILED (sig 15) | 4:29    | ran 2 epochs then sibling-rank death; **excluded** (corrupt `wall_s`, optimistic b-stats) |
  | `41825198` | merged 8r×12 | 46   | FAILED (sig 15) | 21:48   | 0-byte `.out`; absent from CSV                                                            |
  | `41825199` | single 4r×12 | 46   | TIMEOUT         | 2:00:26 | ran 2 epochs, sps_wall 3.45 vs ~27 (8× slow node); **excluded**                           |
  | `41825207` | merged 4r×12 | 46   | TIMEOUT         | 2:00:14 | 0-byte `.out` (hung 2 h); absent from CSV                                                 |

  Net rep counts: 14 configs at n=5; merged {4r×12, 8r×8, 8r×12} and single 4r×12 at n=4.
  3 of 4 failures cluster in seed 46's window (2026-06-16 eve → 06-17 morning) →
  infrastructure flakiness, not a structural merged fault (two were fast merged
  configs that hung).

### Interesting values

**Naive 5-rep `sps_wall` (mean ± std) and CV (std/mean):**

```
 cfg          n   sps_wall ± std    CV       cfg          n   sps_wall ± std    CV
 merged 1r×8  5    49.37 ± 3.69     7.5%     single 1r×8  5     5.82 ± 0.31     5.3%
 merged 1r×12 5    51.50 ± 0.78     1.5%     single 1r×12 5     8.67 ± 0.17     2.0%
 merged 1r×20 5    51.91 ± 0.59     1.1%     single 1r×20 5    12.81 ± 0.95     7.4%
 merged 4r×8  5   153.91 ± 51.35   33.4% F   single 4r×8  5    19.28 ± 1.08     5.6%
 merged 4r×12 4   170.78 ± 52.77   30.9% F   single 4r×12 4    27.29 ± 1.18     4.3%
 merged 4r×20 5   185.69 ± 15.46    8.3%     single 4r×20 5    35.16 ± 4.56    13.0% F
 merged 8r×8  4   273.74 ± 94.85   34.6% F   single 8r×8  5    36.94 ± 3.22     8.7%
 merged 8r×12 4   296.25 ± 82.51   27.9% F   single 8r×12 5    51.06 ± 6.13    12.0% F
 merged 8r×20 5   333.33 ± 62.83   18.8% F   single 8r×20 5    71.23 ± 1.91     2.7%
```

(F = CV > 10% acceptance bar.)

**Per-seed global warming (each config's `sps_wall` normalised to its own
max-across-seeds, then averaged over configs):**

```
 seed  n_cfgs  mean(sps/cfgmax)  mean(normrank, 1=fastest)
  42     18        0.766              0.278   <- globally coldest
  43     18        0.918              0.583
  44     17        0.992              0.791   <- warm plateau
  45     18        0.992              0.883
  46     15        0.938              0.477   <- dips (failure-window degradation)
```

Same config across seeds, e.g. merged 8r×20: 223 / 343 / 371 / 365 / 364
(seed 42 → 46); merged 4r×8: 76 / 126 / 187 / 189 / 191; single 4r×20:
27 / 35 / 38 / 38 / 37.

**Warm-only (seeds 43–46, cold rep 42 dropped) `sps_wall` (mean ± std):**

```
 merged 1r×8  50.11±3.81   merged 4r×8 173.40±31.37   merged 8r×8 314.08±61.04
 merged 1r×12 51.65±0.81   merged 4r×12 197.16±1.58   merged 8r×12 327.81±65.07
 merged 1r×20 52.17±0.13   merged 4r×20 192.60±0.80   merged 8r×20 361.00±12.66
 single 1r×8  5.91±0.29    single 4r×8 19.36±1.22     single 8r×8 38.20±1.79
 single 1r×12 8.68±0.20    single 4r×12 27.87±0.20    single 8r×12 53.72±1.73
 single 1r×20 13.18±0.58   single 4r×20 37.13±1.39    single 8r×20 71.66±1.90
```

**Plateau-only (seeds 44–45) `sps_wall` + CV — the steady-state noise floor:**

```
 merged 1r: 52.09/52.34/52.22 (8/12/20)   CV 0.2/0.0/0.1%
 merged 4r: 187.91/197.94/192.60          CV 1.0/0.6/0.3%
 merged 8r: 367.07/364.12/368.49          CV 0.0/6.5/1.2%
 single 1r: 6.12/8.82/13.61               CV 0.0/2.0/0.1%
 single 4r: 20.23/27.77/38.08             CV 1.6/0.4/1.1%
 single 8r: 39.75/54.53/72.63             CV 0.6/0.7/1.0%
```

**Trimmed mean (drop the single fastest + slowest rep per config) — the
variance is removable, not intrinsic:**

```
 cfg          n | full_mean full_CV | trim_mean trim_CV | dropped lo / hi
 merged 1r×8  5 |    49.37    7.5%  |    50.11    6.4%  | lo s46 / hi s45
 merged 1r×12 5 |    51.50    1.5%  |    51.46    1.5%  | lo s46 / hi s44
 merged 1r×20 5 |    51.91    1.1%  |    52.14    0.3%  | lo s42 / hi s46
 merged 4r×8  5 |   153.91   33.4%  |   167.42   21.2%  | lo s42 / hi s46
 merged 4r×12 4 |   170.78   30.9%  |   196.37    0.6%  | lo s42 / hi s45
 merged 4r×20 5 |   185.69    8.3%  |   192.30    0.3%  | lo s42 / hi s43
 merged 8r×8  4 |   273.74   34.6%  |   287.59   19.8%  | lo s42 / hi s45
 merged 8r×12 4 |   296.25   27.9%  |   301.31   21.6%  | lo s42 / hi s44
 merged 8r×20 5 |   333.33   18.8%  |   357.51    3.6%  | lo s42 / hi s44
 single 1r×8  5 |     5.82    5.4%  |     5.84    5.3%  | lo s42 / hi s45
 single 1r×12 5 |     8.67    2.0%  |     8.63    0.8%  | lo s43 / hi s45
 single 1r×20 5 |    12.81    7.4%  |    13.03    4.7%  | lo s42 / hi s44
 single 4r×8  5 |    19.28    5.6%  |    19.42    2.8%  | lo s46 / hi s45
 single 4r×12 4 |    27.29    4.3%  |    27.77    0.4%  | lo s42 / hi s43
 single 4r×20 5 |    35.16   13.0%  |    36.71    3.7%  | lo s42 / hi s45
 single 8r×8  5 |    36.94    8.7%  |    37.62    4.5%  | lo s42 / hi s45
 single 8r×12 5 |    51.06   12.0%  |    53.36    3.6%  | lo s42 / hi s45
 single 8r×20 5 |    71.23    2.7%  |    71.38    2.3%  | lo s46 / hi s44
```

n=5 → trim leaves 3 reps; n=4 → trim leaves 2 (merged 4r×12 / 8r×8 / 8r×12 +
single 4r×12 trim_CV is a 2-sample spread). Dropped _bottom_ = seed 42 in
**13/18** configs. Both flagged anomalies fall under the 10 % bar: m8r×20
18.8 → 3.6 %, s4r×20 13.0 → 3.7 %. The three that stay > 19 % (merged 4r×8,
8r×8, 8r×12) have a **2-rep** cold ramp (seed 43 also cold), so a single trim
still leaves a cold value in.

**Run 33 cross-check (the two flagged anomalies):** merged 8r×20 = 319.98
(Run 33) vs warm plateau ~365–368 (Run 35); single 4r×20 = +56% vs Run 31
(Run 33) vs plateau CV 1.1% (Run 35).

### Quick conclusions

- **Q1 — merged 8r×20 ≈ 320 reproducible? YES, and it's the genuine top merged
  config.** Warm plateau = **~365–368 sps_wall** (seeds 44–45, CV 1.2%). Run 33's
  320 was a _cooler_-chain reading, not a cache-warmed inflation — the earlier
  "last-in-chain artifact" worry is inverted. Naive 5-rep mean (333) is dragged
  down only by the cold seed-42 (223).
- **Q2 — `sps_wall` noise floor is ~1–2% CV at steady state.** The naive 10–35%
  CV is almost entirely the **cold first rep** (seed 42 at 77% of plateau,
  seed 43 at 92%, seeds 44–45 at 99%). Once warm, the benchmark is highly
  reproducible. Run 33's single 4r×20 "+56%" is explained the same way (cold/
  contended state), not intrinsic metric noise — plateau CV here is 1.1%.
  Confirmed independently by trimming the fastest + slowest rep: m8r×20 CV
  18.8 → 3.6%, s4r×20 13.0 → 3.7%, dropped bottom = seed 42 in 13/18 configs.
- **Merged is worker-insensitive; single is worker-hungry.** Warm merged 8r is
  flat across workers (367/364/369 at NW 8/12/20) — 8 workers already saturate
  the GPU feed. Single scales ~linearly with workers (8r: 40/55/73). **This
  revises the Run 33 "NW=12 wins at 4r" claim** — it was cold-rep noise; warm,
  merged is flat and single prefers NW=20 everywhere.
- **DDP scaling (warm, ×20): merged ~88–92% efficient, single ~67–70%.** merged
  1r→4r→8r = 52.2 → 192.6 (92%) → 368.5 (88%); single = 13.6 → 38.1 (70%) →
  72.6 (67%). **No 2-node (8-rank) merged regression** — confirmed across reps.
- **Lever 3 (merged/single, matched config, warm): 3.8× → 9.2×**, widest at low
  worker counts (8r×8 = 9.2×; 8r×20 = 5.1×). Best absolute throughput:
  merged 8r×20 ≈ 368 sps_wall vs single best 8r×20 ≈ 72.

## Run 36 — Empirical file-open count per sample: single vs merged

First direct measurement of zarr chunk-file opens per `__getitem__`. Monkey-patches
`DirectoryStore.__getitem__` to count chunk-key fetches bucketed by variable,
then iterates 1000 random land-pixel indices through `SingleZarrDataset` and
`MergedZarrDataset` (single-process, no DataLoader). Goal: verify the
theoretical 2.25 chunks-per-modality-call from uniform-straddling, and
reconcile the single-zarr "41 isel calls" vs the actual file-open count.

### Jobs

- `41930447` — single + merged file-open count probe (`scripts/launch_profile_file_opens.sh`,
  `--dependency=afterany:41825191..41825214` — tail of the 5-rep validation chain).
  Completed; `jobs-map/41930447/file_opens.csv` (25 000 rows, 1000 samples × each dataset).

### Interesting values

**Per-sample total chunk-file opens (mean over 1000 samples):**

```
 dataset   n_samples   mean_total_opens   min   max   distinct_vars(logged)
 merged       1000          9.05           4    16          2
 single       1000         92.74          41   164         23
```

**Mean chunks per modality-call (= spatial-straddle base × n_timesteps):**

- Spatial-straddle base = **2.262 chunks/access** (theory: 2.25). Identical for
  every variable, both datasets.
- merged: `dynamic/data` = 6.786 (= 3 × 2.262), `static/data` = 2.262 (= 1 ×).
  Total 9.05.
- single per-var: static (aspect, slope, elevation, weighted\_\*) = 2.262 (1 ts);
  LUH2 14 vars + kg_class + number_of_people = 4.524 (2 ts); LULC_states = 6.786
  (3 ts). 23 vars → total 92.74.

**Straddle distribution (chunks per single-timestep access):** merged static
`{1,2,4}` = 25.0 / 49.4 / 25.6 %; merged dynamic `{3,6,12}` = same 25/49/26
pattern scaled ×3. Matches the theoretical 25 % × 1 / 50 % × 2 / 25 % × 4 exactly.

### Quick conclusions

- **Theoretical 2.25 chunks-per-access confirmed: empirical 2.262.** Land-pixel
  sampling is _not_ biased toward chunk-aligned positions (distribution is the
  exact 25/50/25 uniform-straddle pattern).
- **Single = 92.74 opens/sample, confirming the corrected count** (41 isel calls
  × 2.262 straddle ≈ 92.7), **not the bare 41** quoted in the old Lever-3
  estimate. The 41 is the _isel-call_ count, not the chunk-open count.
- **Merged = 9.05 opens/sample, higher than the 6.75 estimate.** The dynamic
  group is accessed at **3 effective timesteps** (6.786 = 3 × 2.262), not the 2
  assumed in the 6.75 figure; static adds 1 × 2.262.
- **Lever 3 file-open reduction = ~10.2×** (92.74 → 9.05), not the 6× (41→6.75)
  or 14× (92→6.75) previously framed. Minor open item: confirm _why_ merged
  dynamic reads 3 timesteps (autoregressive continuous-sequence construction).

---

## Run 37 — Merged-store chunk-file size distribution

Chunk-size probe (`profile_chunk_sizes.py`) rewritten to walk the store
recursively (handles the merged `dynamic/data` + `static/data` nested groups,
not just flat per-var dirs) and to stat **all** chunk files rather than a
100-sample subset. Goal: get the per-chunk byte distribution for the merged
store, to feed the cache-occupancy term of the cold/warm cost model (compare
against the single-store distribution already in `jobs-map/41548701`).

### Jobs

- `42036891` — chunk sizes of
  `/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr`
  (`scripts/launch_profile_chunk_sizes.sh`). Output:
  `jobs-map/42036891/chunk_sizes.csv`.

### Interesting values

**Merged store chunk-file size distribution (all chunk files stat'd):**

```
 array_dir          n        min     median       mean        p95         max   total_MiB
 dynamic/data   40896      71960      71980    1411499    8019810    10267903     55050.5
 static/data     2556      51917     388007    1874845    7139933     7366102      4570.1
 (coordinate arrays: dynamic/{time,variable,latitude,longitude}, static/{...} — 1 chunk each, <10 KB)
```

- **Merged total: 43,459 chunk files, 58.22 GiB, mean chunk 1.40 MiB.**
- `dynamic/data` median = 72 KB (≈ 17 × the single-store ~4 KB ocean floor — all
  17 vars stacked into one chunk), mean 1.4 MB, p95 8.0 MB, max 10.3 MB.
- `static/data` median 388 KB, mean 1.87 MB, p95 7.1 MB (6 vars stacked, float64).

**Chunk geometry (from `.zarray`):**

- Single `c3ann`: shape `[16, 18000, 36000]`, chunks `[1, 512, 512]` → spatial grid
  `ceil(18000/512) × ceil(36000/512)` = 36 × 71 = **2556 spatial chunks**, 16 time
  chunks → **40,896 chunk files per 3-D var**.
- Merged `dynamic/data`: shape `[17, 16, 18000, 36000]`, chunks `[17, 1, 512, 512]`
  → variable dim is one chunk → 1 × 16 × 2556 = **40,896 files** (same spatial/time
  grid, all 17 vars folded into each chunk).
- Merged `static/data`: chunks `[6, 512, 512]` → 1 × 2556 = **2556 files**.
- Both stores: Blosc(lz4, clevel=5, shuffle=1), identical codec.

**Derived total chunk-file (inode) counts — same grid, same 2000–2015 data:**

| Store                         | 3-D vars |      files/3-D var | static files | total chunk files |
| ----------------------------- | -------: | -----------------: | -----------: | ----------------: |
| merged (measured)             |     1×17 |             40,896 |        2,556 |        **43,459** |
| single 66 G (`c3ann` ×17 + 6) |       17 |             40,896 |      6×2,556 |     **≈ 710,568** |
| single 213 G (1960–2015, ×56) |       17 | 56×2,556 = 143,136 |      6×2,556 |   **≈ 2,448,648** |

### Quick conclusions

- **Same data, 16× fewer inodes when merged.** Merged folds 17 dynamic vars into one
  chunk along the `variable` dim: 43,459 chunk files vs ~710,568 for the single 66 G
  store, for the same 58–66 GiB of bytes. The 213 G store (3.5× the years) is ~2.45 M.
- **The merged win has two distinct multiplicative mechanisms, not one.** Run 36 gave
  the per-sample open reduction (92.7 → 9.05, ~10×); Run 37 gives the _total inode_
  reduction (~711 K → 43 K, ~16×). Both follow from variable-stacking.
- **For the store-size question, the binding cache is metadata, not data bytes.** The
  merged store (58 GiB) and the single 66 G store both fit comfortably in 512 GB node
  RAM, so the _data_ page cache is not what overflows. What differs by 16× is the
  chunk-file (inode/dentry) count: merged 43 K ≪ the ~500 K dentry-cache threshold
  (Run 14); single 66 G ≈ 711 K is just past it; single 213 G ≈ 2.45 M is ~5× past.
  Since the cold-open cost is itself a GPFS _metadata_ round-trip (~20 ms), the
  metadata cache is exactly what governs cold-vs-warm — consistent with the 1.9–3×
  size-effect spike (BATCH 10–11) tracking inode count, not byte count.
- This sharpens the cold/warm verification: model the inode/dentry cache (capacity
  ≈ files, not bytes) in the LRU simulation, with `m` = inode-miss rate.

---

## Run 38 — Per-chunk cold/warm latency trace (cold-miss-rate theory)

First direct, per-open measurement of the cold-vs-warm split on the real
dataloader. `profile_cold_warm.py` monkey-patches the zarr store `__getitem__`
to time every chunk-data read, then replays `NUM_SAMPLES=80` random land-pixel
samples through three arms. Goal: test whether bigger working set → more cold
opens → longer per-sample time, and whether an LRU cache sim reproduces the
measured miss rate `m`. (The 213 G single store that would have given a clean
same-layout size A/B was deleted; this uses the remaining multizarr stores.)

Arms (all seed 42, same land index):

- `single_match` — bundled single store, span `[2001,2014]` (small working set).
- `multi_match` — 5-store multizarr, span `[2001,2014]` (same span, 5-store layout).
- `multi_full` — 5-store multizarr, span `[1902,2014]` (**same layout as
  multi_match, ~8× the year span** → isolates working set with zero layout
  confound; the cleanest size signal in the run).

multizarr LUH2 store geometry: `[166, 18000, 36000]` chunks `[1,512,512]` →
424 K chunk files/var × 14 ≈ 6 M inodes (vs ~573 K for bundled-single LUH2).

### Jobs

- `42041162` — `scripts/launch_profile_cold_warm.sh`, `--exclusive` (clean cold
  cache), COMPLETED 29 min. Output `jobs-map/42041162/cold_warm.csv` (20 224 reads).
  Arms run sequentially single_match → multi_match → multi_full (so later arms see
  a *warmer* node — works against any multi_full penalty).

### Interesting values

**Per-arm (cold threshold 1 ms):**

```
 arm            reads    m      cold_mean  warm_mean  per-sample_med   first-q -> last-q
 single_match    6724  0.997    25.07 ms   0.399 ms      1807 ms       2127 ->  2321 ms (x0.92)
 multi_match     6481  0.969    24.87 ms   0.289 ms      1593 ms       2030 ->  2109 ms (x0.96)
 multi_full      7018  0.968    39.93 ms   0.399 ms      1780 ms       7539 ->  2100 ms (x3.59)
```

**LUH2-only per-open (single bundled store vs multizarr 606 G luh2 store):**

```
 single_match : reads=4592  m=1.000  cold_mean=25.86 ms  p50=25092 us
 multi_match  : reads=4349  m=1.000  cold_mean=25.87 ms  p50=24873 us
 multi_full   : reads=4721  m=1.000  cold_mean=48.47 ms  p50=25317 us
```

**Latency histogram** (all arms): warm `<100 us` ~0.1 %, valley `500–2000 us`
0.6–2.4 %, **cold `10–50 ms` bucket = 90–95 %**, `>50 ms` 1.1–1.5 %. Clean bimodal.

**LRU miss-rate sweep:** `m_sim` flat at **0.982–0.986 across every capacity
10 K → 2 M files** for all arms; distinct files touched ≈ total reads
(single 6627/6724, multi_full 6890/7018) → **almost zero chunk reuse at K=80**.

### Quick conclusions

- **Bimodal cold/warm confirmed on the real dataloader:** warm ≈ 0.3 ms, cold ≈
  20–25 ms, clean valley at 0.5–2 ms. 90–95 % of reads are cold ~20 ms opens.
- **At K=80 random sampling the cache-capacity / miss-rate axis is NOT exercised.**
  Distinct files ≈ total reads (≈ zero reuse), so `m ≈ 1` for every arm and the LRU
  sim is flat across all capacities. Testing the data-page miss-rate story needs ≫80
  samples (reuse must emerge) or a locality-controlled pattern. This run sits entirely
  in the "everything cold" regime — exactly what we want for measuring the *cold cost
  itself*.
- **The size effect appears in the per-cold-open COST, not the miss rate.** LUH2 cold
  open over a 166-yr span (multi_full, 48.5 ms) is **1.87× a 14-yr span of the same
  store** (multi_match, 25.9 ms), which equals the bundled single store (25.9 ms).
  Matched-span multizarr ≈ single → it is the **access breadth, not the 5-store
  layout**. Both arms miss the data page-cache (`m≈1`); the extra cost is inside the
  cold open → consistent with **GPFS metadata-lookup cost growing as the distinct-inode
  working set grows** (multi_full ~283 K LUH2 inodes vs ~36 K for multi_match).
- **Cold-start transient localizes the historical 1.9–3× spike.** multi_full's
  first-quartile 7.5 s/sample → last-quartile 2.1 s (warming 3.6×); narrow-span arms
  flat. The broad-span metadata working set warms over the run. The 3.6× cold-start
  ratio brackets the historical 64 G-vs-213 G spike magnitude (1.9–3×, BATCH 10–11),
  which were cold/spike batches — i.e. this same regime.
- **Caveat on "static 20 ms cold cost":** this run says the per-open cost is *not*
  constant — ~25 ms for a narrow working set, ~48 ms when the distinct-inode working
  set is ~8× larger. **Held back from `knowledge.md` pending the Run 39 replication.**
  Ordering makes it conservative (multi_full ran last, warmest node, still slowest).

---

## Run 39 — Replication: cold-open cost vs access span (monotonic sweep)

Focused replication of the Run 38 finding (per-cold-open cost grows with the
distinct-inode working set). Same probe mechanism, but the arms sweep ONLY the
time-window width on the SAME multizarr stores (layout fixed), at four widths, to
test for a *monotonic* trend rather than two points. `profile_cold_span.py`, same
seed (42) and `NUM_SAMPLES=80` as Run 38, narrow → wide order (later/wider arms
run on a warmer node → conservative). The 14-yr and 113-yr endpoints should
reproduce Run 38's multi_match (~26 ms) and multi_full (~48 ms) LUH2 cold means.

Arms: `single_14` [2001,2014] (matched-span control) · `multi_14` [2001,2014] ·
`multi_40` [1975,2014] · `multi_66` [1948,2014] · `multi_113` [1902,2014].

### Jobs

- `42095663` — `scripts/launch_profile_cold_span.sh`, `--exclusive`. Output
  `jobs-map/42095663/cold_warm.csv`. Analyze with `scripts/analyze_cold_warm.py`
  (set `CSV_PATH` to this job); read the LUH2-only per-arm `cold_mean` trend.

**Confirmation criterion:** LUH2 `cold_mean` rises monotonically with span, with
14-yr ≈ 26 ms and 113-yr ≈ 48 ms reproducing Run 38. If it holds, promote the
cold-cost-vs-working-set finding to `knowledge.md`.

### Interesting values

Job `42095663`, COMPLETED 51 min, empty err. 34 506 reads.

**LUH2-only per-open cold cost by span (mean | median):**

```
 arm         span(yr)   cold_mean   p50        first-q -> last-q (per-sample)
 single_14      14       27.53 ms   25.4 ms    2248 -> 2466 ms (x0.91)
 multi_14       14       27.39 ms   26.1 ms    2034 -> 2394 ms (x0.85)
 multi_40       40       31.09 ms   26.6 ms    2463 -> 3457 ms (x0.71)
 multi_66       66       30.02 ms   30.0 ms    2011 -> 2219 ms (x0.91)
 multi_113     113       30.42 ms   30.3 ms    2574 -> 2406 ms (x1.07)
```

**Run 38 endpoints, side by side:**

```
                  multi_14/match(14yr)   multi_113/full(113yr)
 Run 38 LUH2 cold_mean   25.87 ms              48.47 ms       (1.87x)
 Run 39 LUH2 cold_mean   27.39 ms              30.42 ms       (1.11x)
 Run 38 cold-start (1st-q/last-q)              7539->2100 ms (x3.59)
 Run 39 cold-start (1st-q/last-q)              2574->2406 ms (x1.07)
```

All arms still `m≈0.97–1.0` (everything cold, no reuse at K=80); LRU sim flat at
0.98 across all capacities — same as Run 38.

### Quick conclusions

- **Run 38's strong finding did NOT replicate.** multi_113 LUH2 cold_mean = 30.4 ms
  here vs **48.5 ms** in Run 38 (multi_full); cold-start transient **1.07×** here vs
  **3.59×** in Run 38. The 48 ms / 7.5 s-per-sample cold start in Run 38 was a
  **transient GPFS condition** (metadata-server load / node state on that run), not a
  property of the access span.
- **Cold-open cost is approximately constant ≈ 25–31 ms across 14–113-yr spans.** The
  *median* rises weakly and monotonically (25.4 → 26.1 → 26.6 → 30.0 → 30.3 ms,
  **~1.19×** over an 8× span increase); the *mean* is noisy and non-monotonic
  (multi_40 31.1 > multi_113 30.4). Run-to-run noise (~±2–3 ms) is comparable to the
  whole span signal. So at most a weak working-set dependence, **far** from Run 38's
  1.87×.
- **This vindicates the original "roughly static cold cost" premise.** Cold ≈ 25–30 ms,
  warm ≈ 0.3 ms, bimodal — reproducible across Run 38 + 39.
- **Therefore the historical 1.9–3× store-size effect is NOT explained by per-open cost
  growth.** With per-open cost ~constant, the size effect must live in the **miss rate**
  (how *often* a read is cold), which is still untested — K=80 random sampling produces
  ~zero reuse so `m≈1` everywhere. Next: a long / locality-controlled run that actually
  exercises cache reuse.
- **Do NOT promote the "cold cost grows with working set" claim to `knowledge.md`** —
  it failed replication.

---

## Run 40 — Multizarr + single-small concrete chunk-file counts

Direct chunk-file walk (count only, no stat) of the 5 multizarr stores and the
single-small 66 G store, to replace the geometry-derived single estimate
(Run 37 ≈ 710,568) with a measured value and get the first concrete inode count
for the historical multizarr layout. `profile_store_file_count.py` reuses Run 37's
`is_chunk_file` walk but skips `getsize` so it scales to the multi-million-file
LUH2 cube. From the single-small measured count we extrapolate the deleted
single-big (213 G, 1960–2015) by the 56/16 year ratio.

### Jobs

- `42158395` — chunk-file counts of all 5 multizarr stores + the single-small
  store (`scripts/launch_profile_store_file_count.sh`). Output:
  `jobs-map/42158395/store_file_count.csv`.

### Interesting values

Job `42158395` walked all 6 stores in **11 s** (count-only, no `getsize`).

**Per-store chunk-file counts (data arrays; coordinate arrays are 1 chunk each):**

| Store                                     | data chunk files | data vars | span (yr) |
| ----------------------------------------- | ---------------: | --------: | --------: |
| multizarr LUH2 (1850–2015)                |    **5,798,010** | 14        | 166       |
| multizarr population (1850–2020)          |          437,076 | 1         | 171       |
| multizarr HILDA (1899–2020)               |          309,276 | 1         | 122       |
| multizarr köppen-geiger (1901–2020)       |          172,560 | 1         | 120       |
| multizarr static                          |           15,336 | 6         | static    |
| **multizarr total (5 stores)**            |    **6,732,258** | 23        | mixed     |
| single-small NON_preprocessed (2000–2015) |      **710,568** | 17 dyn + 6 static | 16 |

(Coordinate arrays add 14 more 1-chunk files across the 5 multizarr stores and 3
to single-small — negligible.)

**Per-var density (measured count vs dense `years × 2556` ceiling):**

- single-small: **every** 3-D var = exactly `40,896 = 16 × 2556`; every static var
  = exactly `2,556`. Store is **fully dense** — no fill-chunk omission.
- multizarr population `number_of_people` = `437,076 = 171 × 2556` exactly (dense).
- multizarr LUH2 vars span `401,267` (`c4per`) → `424,296` (`secma`); `secma` =
  `166 × 2556` exactly (dense), the rest 95–100 % dense (a few all-fill chunks omitted).
- multizarr köppen `kg_class` = `172,560` = **56 %** of dense `120 × 2556 = 306,720`
  (land-only field → ~44 % all-ocean chunks omitted).
- multizarr HILDA `hilda_labels` = `309,276` vs dense `122 × 2556 = 311,832` (99 %).

**Single-big (213 G, 1960–2015, deleted) extrapolation from the dense single-small:**
17 dynamic × (56 × 2556) + 6 static × 2556 = `2,433,312 + 15,336` = **2,448,648**.

### Quick conclusions

- **Single-small measured 710,568 chunk files = Run 37's geometry-derived 710,568
  to the file.** The single store is fully dense, so the derivation rests on a
  measured fact now, not pure `.zarray` geometry. Single-big (213 G) extrapolates
  cleanly to **2,448,648** by the 56/16 year ratio on the dynamic vars only (static
  is year-invariant) — likewise confirming Run 37's 2,448,648.
- **The historical multizarr layout is ~6.73 M chunk files on disk** — ~9.5× the
  single-small (711 K) and **~155× the merged store** (43,459, Run 37). It is the
  worst layout by far for inode/dentry-cache pressure. Caveat: not span-matched —
  LUH2/population cover 166/171 yr vs single-small's 16 yr, so most of the gap is
  extra years, not pure layout. LUH2 alone (5.8 M) dominates the multizarr total.
- **Per-var density differs by store, not uniformly.** The single preprocessed
  store materializes every chunk (dense); the historical multizarr cubes omit
  all-fill chunks, heavily for land-only fields (köppen 56 % dense) and lightly for
  LUH2/HILDA (95–100 %). So a dense extrapolation overcounts sparse multizarr vars
  but is exact for the single store.

---

## Run 41 — Loader parity: all consolidated stores vs the multizarr baseline

### Jobs

- 42328823 — `MultiZarrDataset` (5 CONCERTO stores) vs 6 candidate stores
  (single small/big/256, merged small/big, preprocessed), 64 seeded patches,
  element-wise diff per tensor, `time_range [2001,2010]`.

### Interesting values

- All 6 candidates produce **identical** diffs vs multizarr. `single_big` and
  `single_256` are **bit-identical** to `single_small`; `merged_*` and
  `preprocessed` match `single` to **float32 ε** (≤ 2.4e-7; only the exact-match
  count flips on `x_static`).
- Per-tensor vs multizarr: `x_hilda_prior` exact (0); `hilda_target` values exact
  (dtype-only diff, int64 vs uint8); `x_kg` diff **31.0** on 22/64 patches;
  `x_continuous` **18.4207**; `x_static` **3.297**.

### Quick conclusions

- New-store family is internally consistent (single == merged == preprocessed) and
  data-faithful; baked preprocessing == read-time normalization.
- Divergence from legacy multizarr is loader **recipe**, not data: population
  `log1p` (single/merged) vs `log(x+1e-8)` + `nan→0` (multi) → `ln(1e-8)=−18.4207`
  at zero/ocean cells; KG ocean fill `0` (single) vs `31` (multi); static
  read-time z-score/cos (single) vs pre-baked `static_cube_processed` read as-is
  (multi); `secma`/`secmb` use `_full_time` (single) vs `_full_time_train` (multi).

---

## Run 42 — Store-size mechanism: xarray/dask graph overhead, not cache-miss rate

### Jobs

- 42344285 — load timing, 100 samples single-thread: single_small/big + multi
- 42344286 — cache-miss probe: single_small vs single_big, matched 14-yr window
- 42344287 — chunk-file counts: single/merged small + big
- 42354183 — load timing, 5 arms (adds merged_small, merged_big)
- 42354184 — xarray-overhead probe (warm decomposition + identical-slice)
- 42354185 — store size on disk (`du -sh`)

### Interesting values

- Per-sample (single-thread, `[2001,2010]`, median ms): single_small **1329**,
  single_big **5513** (4.15×), merged_small **174**, merged_big **286** (1.64×),
  multi **7970**. Merged vs single: **7.6× small, 19.3× big**.
- Cache-miss matched window (single_small vs single_big, `[2001,2014]`): miss rate
  m **0.994 vs 0.994** (ratio 1.00); cold-open cost **25.25 vs 25.25 ms**;
  per-sample chunk-read I/O 1843 vs 1859 ms (1.01×); files/sample 90.5 both. LRU
  sim m flat ~0.97 across capacities 10K→2M (random sampling → no chunk reuse).
- Xarray probe (warm): Part 1 (full `__getitem__` decomp) I/O ratio big/small
  **0.99**, non_io (xarray+decode) ratio **6.72**; Part 2 (identical warm
  `c3ann.isel(time=0,...).to_numpy()`, I/O ≈ 0.3 ms) total ratio **6.87×** — pure
  dask graph build/cull, ∝ 115/16 time-chunks.
- Chunk files: single_small **710,571** (= Run 40), single_big **5,012,319**
  (7.05×), merged_small **43,459**, merged_big **296,503**. Size on disk:
  single_small 66G, single_big 435G, merged_small 59G, merged_big 387G,
  multizarr ~680G (luh2 alone 606G).

### Quick conclusions

- **The store-size penalty is xarray/dask graph overhead** (build + cull), scaling
  with the array's total chunk count (= time extent), **not** inode/dentry
  cache-miss rate or I/O. The matched-window control shows m and cold-open cost
  identical small vs big (refutes the line-281 `m_big/m_small ≈ 1.9–3` prediction);
  the warm identical-slice isolates the graph at 6.87× with I/O ≈ 0.
- Merging neutralizes most of it: merged_big is only **1.64×** merged_small (vs
  single's 4.15×) because merged reads ~9 chunks/sample from 1 stacked array → a
  tiny graph base. **The merge benefit grows with store size** (7.6× → 19.3×).
- Overhead ∝ the **store's** time-extent, not the training window — the graph
  carries the full 1901–2015 chunk structure even when training on `[1960,2000]`.
  Fixes that cut the graph: `chunks=None` (drop dask) or bypass xarray (Lever 1).

---

## Run 43 — Production final training (merged-big) + worker scaling: equivalent model in 2.34 h

### Jobs

- 42401736 — **12w baseline**, full 10-epoch training (merged-big, `[1960,2000]`, 1 node / 4 ranks)
- 42410729 — eval of 42401736 `best_model.pth` (2001–2014, 14-yr recurrent rollout)
- 42413747 — **20w**, 1 node / 4 ranks (running); chain 42413748 (eval) → 42413749 (4node) → 42413750 (eval)
- 42417946 — 100-sample single-thread load-timing re-confirm (single/merged small+big + multi)
- 42417947–42417958 — 2 reps each of {12w, 20w, 4node-20w} + intercalated evals (GPFS variance)

### Interesting values

Per-batch timing (4-rank DDP, batch=8, from `batch_times.csv`):

|                    | 12w (42401736, full) | 20w (42413747, ~6.5 ep) |
| ------------------ | -------------------: | ----------------------: |
| p50 (GPU floor)    |   154 ms = 51.9 sps  |     157 ms = 51.0 sps   |
| mean (effective)   | **272 ms = 29.4 sps**| **200 ms = 40.0 sps**   |
| p95                |             1306 ms  |              **215 ms** |
| stalls > 1 s       |          7.7%        |              **1.8%**   |
| stall cadence      |          every 12    |              every 20   |

- Total wall (12w): **8420 s = 2.34 h** for 10 epochs vs **~55 h** legacy multizarr reference (~23×).
- Convergence: train 1.034 → 0.852 monotone; **best val 0.596 @ epoch 8** (`best_model.pth`). Val < train
  throughout because the train task is harder (prior masked 0.6→0.99, teacher forcing 1→0, noise 0.1);
  val uses the full unmasked prior, TF=0. Late-epoch val rise (0.728 @ e10) is that task mismatch +
  constant LR + noisy 500-batch val subset, not overfitting.
- Eval (`best_model.pth`, 2001–2014): Year0 acc **0.874** / macroF1 0.757 / IoU 0.639 → Year13 0.792 /
  0.579 / 0.466 (recurrent drift). Class acc: ocean .9999, forest .96, pasture .85, otherland .69,
  cropland .66, water .39, urban .37, grass/shrub .13.

### Quick conclusions

- **Merged-big production training delivers an equivalent model in 2.34 h vs ~55 h reference (~23×).**
- **20 workers nearly removes the stall tax:** effective throughput 29.4 → 40.0 sps/GPU, p95 1306 →
  215 ms, stalls 7.7% → 1.8%. The fast-batch floor (~155 ms = 51 sps/GPU) is GPU-compute-bound and
  unchanged by workers — so 53 sps/GPU is not reachable by adding workers alone at batch=8.
- **The periodic stall is structural at every-`num_workers` batches** (12 → every-12, 20 → every-20):
  the whole worker pool refills together. More workers = more outstanding I/O hides it, doesn't remove it.
- Eval path validated end-to-end (fixed `setup_model` `cfg.model.model_type`→`type`; `test_unet`
  `rollout.timesteps_target`→`timesteps_target` and dropped `mask_value=` from `RecurrentUNet`).
- Reps (42417947–58) + 100-sample probe (42417946) queued serially to quantify GPFS run-to-run
  variance and re-confirm single-big » multizarr (per AGENTS.md: no concurrent jobs on one store).

---

## Run 44 — Raw-zarr loader (no xarray): bit-identical parity; speed preliminary (contended, see Run 45)

`RawMergedZarrDataset` keeps the xarray init (CF-decodes the `days since 1901` time axis) but reads
every patch via the raw `zarr` API (`zarr_array.oindex[:, ts, lat, lon]`), eliminating the per-sample
xarray/dask graph. Inherits all processing from `MergedZarrDataset` → parity by construction.

### Jobs

- 42419686 — parity: raw vs xarray, 100 shared seeded samples, all 5 output tensors
- 42419687 — load timing, 6 arms (adds `merged_big_raw`), 100 samples single-thread, `[2001,2010]`
- 42419688 — multi-worker throughput, `merged`/`raw` × {12w, 20w}, merged-big, `[1960,2000]`

(All three ran in parallel with the running chain → absolute timings inflated by GPFS contention;
the within-job raw-vs-xarray ratio is the clean signal.)

### Interesting values

- **Parity: PASS** — `max_abs_diff = 0.000e+00` on x_continuous, x_kg, x_static, x_hilda, targets
  (100 samples). Raw read is bit-identical (dynamic/data is float32, `fill_value=NaN`, no scale/offset).
- Per-sample single-thread (median ms): single_small 1251, single_big 5511, merged_small 140,
  merged_big 244, **merged_big_raw 37.0**, multi 8052. Raw is **6.6×** merged_big and faster than even
  merged_small xarray (140).
- Multi-worker sps (single-rank): merged_w12 **38.2** (p95 2547 ms, 18 stalls) → raw_w12 **68.1**
  (p95 832, 5 stalls); merged_w20 **61.3** (p95 871, 8) → raw_w20 **114.2** (p95 604, 5).

### Quick conclusions

- **SOLID (contention-proof): the raw loader is correct** — bit-identical to xarray (parity 0.0).
- **PRELIMINARY (ran under contention — do NOT treat as fact):** raw looked ~6.6× single-thread but
  only ~1.8× multi-worker. That mismatch means either the single-thread number is inflated, the
  multi-worker scales poorly, or one reading is corrupted by the concurrent chain. **Numbers omitted
  from knowledge.md until re-measured clean.**
- **Clean re-measurement = Run 45 (queued serially after the current chain):** 100-sample timing ×2
  + throughput {12,16,20}w ×2 (non-contended), then `merged_raw` trainings {12,16,20}w × {1r,4r} ×2
  + evals to settle the real speedup and confirm the stall tax drops with matching metrics.
- **Merging still wins independently** of the xarray fix: raw `single` reads ~93 chunks/sample vs
  merged's ~9, so `merged_raw` (both wins stacked) stays well ahead — orthogonal levers.

---

## Run 45 — Clean raw-loader benchmark + full {12,16,20}w × {1r,4r} matrix: ~7× per-sample, ~2–2.5× multi-worker, GPU-bound training

Serial (non-contended) re-measurement of Run 44 plus the end-to-end training/eval matrix. Analysis
scripts: `analysis/` package (`uv run python -m analysis.report`). Run after the previous chain
drained, so timings are clean.

### Jobs

- 42441485, 42441486 — 100-sample single-thread load timing ×2 (clean)
- 42441487, 42441488 — multi-worker throughput, `merged`/`raw` × {12,16,20}w ×2 (clean)
- 42441489–42441512 — `merged_raw` trainings {12,16,20}w × {1r,4r} ×2 + intercalated evals
- Killed by group disk-quota (bsc32 scratch 511/488 TB): **42441509** (1r20w r2, ep7) and
  **42441511** (4r20w r2, ep1, truncated checkpoint → eval 42441512 crashed in `torch.load`)

### Interesting values

Sample timing — clean median ms/sample (2 reps), and raw stability across **all four** probes:

| arm            | 42441485 | 42441486 | clean mean | (Run 44 contended) |
| -------------- | -------: | -------: | ---------: | -----------------: |
| merged_big     |    270.5 |    277.2 |      273.9 |          244 / 290 |
| merged_big_raw |     38.6 |     37.2 |   **37.9** |          37.0 / 38.0 |

→ clean speedup **7.0× / 7.5×**. The raw number is rock-stable at 37–38 ms across early/contended/clean;
the 6.6–7.6× spread was entirely the **noisy xarray numerator**, not raw.

Throughput — clean mean over 2 reps (sps), raw/merged ratio grows with workers:

| workers | merged | raw   | ratio  |
| ------: | -----: | ----: | -----: |
| 12      |   42.0 |  80.0 | 1.90×  |
| 16      |   52.6 | 114.6 | 2.18×  |
| 20      |   59.0 | 145.3 | 2.46×  |

(Raw arms are noisier rep-to-rep — e.g. raw_w12 60→100 — because at those rates the main-process
collation/pin/IPC and GPFS-metadata jitter, not the per-sample read, set the ceiling.)

Training (raw `merged_raw`, healthy reps, per-batch from the analysis pass):

| combo  | p50 ms | mean ms | stalls | sps/gpu | wall h | best_val |
| ------ | -----: | ------: | -----: | ------: | -----: | -------: |
| 1r12w  |    153 | 158/221 |  0/2.9% | 51/36   | 1.27/1.80 | .594/.577 |
| 1r16w  |    154 | 157/165 |  ~0%   | 51/48   | 1.26/1.37 | .584/.568 |
| 1r20w  |    153 | 158/159 |  ~0%   | 51/50   | 1.28/0.92 | .591/.579 |
| 4r12w  |    153 | 165/226 |  0/3.2% | 49/35   | 1.32/1.84 | .582/.620 |
| 4r16w  |    154 | 166/177 |  ~0%   | 48/45   | 1.32/1.48 | .582/.568 |
| 4r20w  |    153 | 167/165 |  ~0%   | 48/48   | 1.39   | .572 |

- **Merged baselines for contrast** (Run 43): 1r12w 272 ms / 29.4 sps / 7.7% stalls / 2.34 h;
  1r20w 199 ms / 40 sps / 1.8% / 1.76 h.
- Quality cross-check (best_val, complete reps): merged vs raw ranges **overlap** per category (e.g.
  4r16w merged .593–.598 vs raw .568–.582; 1r12w merged .588–.596 vs raw .577–.594).
- Eval (raw): Year0 acc ~0.87–0.90 / Year13 ~0.78–0.82 — same band as merged (Run 43).

### Quick conclusions

- **Resolves Run 44's ambiguity — neither reading was "wrong".** Single-thread is **~7×** (clean,
  reproduced 4×); multi-worker is **~1.9–2.5×** and *grows with workers* (clean). The contended 1.8×
  was just the low end. The 7× → ~2.5× compression is the expected ceiling effect: with many workers
  the per-sample cost is overlapped, so the end-to-end win is smaller than the per-call win.
- **Raw makes training GPU-bound.** Raw runs sit at p50 ≈ 153 ms ≈ **51 sps/GPU (the compute floor)
  with ~0% stalls** — the dataloader stall tax is essentially gone. Wall drops 2.34 h → **1.27 h**
  (~1.8×) at 12w, and **raw 12w (51 sps, 0 stalls) beats merged 20w (40 sps)** — fewer workers needed.
- **Quality preserved.** Bit-identical data (Run 44 parity) → merged and raw produce statistically
  equivalent models and evals; best_val and per-year accuracy bands match across loaders.
- **Reproducibility.** Within-loader timing reps agree (cv ≤ ~3%) except the two **12w** reps that
  overlapped the disk-pressure window (16% cv, one slow rep each — GPFS, not the loader). Quality cv
  ≤ 3%; eval is looser (recurrent 14-yr rollout amplifies tiny model diffs — inherent, not a bug).
- **Infra incident (not a result):** the **bsc32 group scratch went over quota** (511/488 TB, 5.2-day
  grace) mid-chain, killing the two w20 rep-2 raw jobs. They have valid *timing* but invalid *quality*;
  rerun 42441509 / 42441511 once space frees to complete the 2nd quality rep for 1r20w and 4r20w.
