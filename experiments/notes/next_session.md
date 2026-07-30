# Next session — pick up the 5-rep validation sweep

**Session-close state (2026-06-15 mid-day):**

- **90-job mega-chain submitted** as the 5-rep validation of the 18-config
  sweep. Job IDs `41825117`–`41825214`, single `--dependency=afterany`
  chain. Tag file: `jobs-map/chain_1781521500_tags.tsv`.
- **Reason for the rerun:** Run 33 (`41799685`–`41799702`) found two
  anomalies — merged 8r×20 = 320 sps_wall (last-in-chain, likely
  cache-warmed) and single 4r×20 = +56% vs Run 31's clean reference.
  Both are flagged but not yet explained.
- **5 seeds (42–46)**, each independently shuffling the 9 `(ranks,
  num_workers)` configs from `{1,4,8} × {8,12,20}` per dataset. Every
  rep alternates m → s → m → s → … so no two consecutive jobs read the
  same store. Estimated wall ≈ 5 × Run 33's 8 h = **~40 h end-to-end**.
- **Profiling chain finished** (`41817507`–`41817512`): 3 merged + 3 single
  1r×12 with `nsys --capture-range=cudaProfilerApi` over the last 5 of 15
  batches. `nsys-rep` files in each `jobs-map/<id>/`. **Unanalysed.**
  Observation noted: single sqlite event store is 15× larger than merged
  (243–354 MB vs 14–18 MB) — consistent with the worker-stall pattern.
- **Earlier verification chain (Run 34)**: `41811414` completed (m 8r×20),
  `41811415` + `41811416` cancelled mid-flight. **Discard all three**;
  the 5-rep chain supersedes them.
- **New scripts available** in `scripts/` (use these, do NOT paste the
  inline Python script from prior next_session):
  - `scripts/build_run_chain.py` — chain generator (already used).
  - `scripts/build_run_csv.py` — reads tag TSV + per-job logs + queries
    sacct via ssh, emits CSV.
  - `scripts/compare_runs.py` — groups multiple CSVs by `(dataset,
    ranks, num_workers)`, reports mean ± std per metric.
- **New configs**: `inputs/profile-quick-{single,merged}.yaml` — copies
  of `debug-{single,merged}.yaml` with `epochs: 1`,
  `train/val/test_steps: 15/1/1`, and `training.profiling.{enable=true,
  active_batches=5}` baked in. Used with `scripts/launch_profiling.sh`.
- `knowledge.md` has NOT been propagated for Runs 29–34 (held pending
  this 5-rep validation, per explicit user decision).
- `todo.md` holds the long-term lever pipeline (Lever 1 next).
- **File-opens probe queued as Run 36** (`41930447`,
  `scripts/launch_profile_file_opens.sh`). Submitted with
  `--dependency=afterany:41825191..41825214` — the tail 23 jobs of the
  5-rep chain — so it cannot run concurrently with any bench job.
  Measures empirical chunk-file opens per `__getitem__` for both
  `SingleZarrDataset` and `MergedZarrDataset` (1000 random samples each).
  See dedicated section below.

## When the 5-rep chain finishes

### Step 1 — pull and verify

```bash
/hpc-pull                                     # syncs logs/ + jobs-map/
ssh mn5 "squeue -u bsc096444"                 # confirm queue empty
ls jobs-map/ | grep -E "^4182[0-9]{4}$" | wc -l  # should be ≥ 90
```

Each chain link uses `--dependency=afterany`, so a single failed job
still allows the rest to run. If a job FAILED in the chain (check
`sacct -j ... --format=State`), salvage what completed and ask the
user before resubmitting failures.

### Step 2 — build CSV and split by rep

```bash
uv run python scripts/build_run_csv.py jobs-map/chain_1781521500_tags.tsv > 5reps.csv

# Split into one CSV per seed for compare_runs.py
for seed in 42 43 44 45 46; do
  (head -1 5reps.csv && awk -F, -v s=$seed '$5==s' 5reps.csv) > rep_${seed}.csv
done
```

### Step 3 — compare across reps

```bash
uv run python scripts/compare_runs.py rep_42.csv rep_43.csv rep_44.csv rep_45.csv rep_46.csv
```

The output groups each `(dataset, ranks, num_workers)` and reports
`mean ± std` for: `wall_s`, `startup_s`, `sps_wall`, `b_med`, `b_mean`,
`b_p95`.

**Acceptance criteria:**

- `sps_wall` coefficient of variation (std / mean) < 10% across the 5
  reps for every config → measurements are reproducible.
- > 10% CV on any config → flag for the user. Especially watch m8r×20
  (the Run-33 anomaly) and s4r×20 (the cross-check anomaly).

### Step 4 — write Run 35 in `results.md`

- "Interesting values": the `compare_runs.py` table (raw measurements
  only) + the Run 33 column for direct comparison.
- "Quick conclusions": did the m8r×20 = 320 sps_wall reproduce? did
  s4r×20 settle? what's the empirical noise floor on `sps_wall` for
  this benchmark setup?

## What this chain answers

Two specific questions left open by Run 33:

1. **Is merged 8r×20 = 320 sps_wall reproducible**, or was it a
   last-in-chain cache-warming artifact? The 5 reps each have 8r×20
   placed at different positions in their respective 18-job
   sub-sequences — if 320 holds across reps, it's a real sweet spot;
   if it varies wildly with position, it was cache state.

2. **What's the noise floor on `sps_wall` measurements** in this
   benchmark? Run 33's single 4r×20 cross-check showed +56% vs
   Run 31. The 5-rep std/mean will tell us if that gap is within
   normal GPFS-state variance.

It also re-confirms (or contradicts) the Run 33 finding that **no 2-node
merged regression exists in real DDP training**. With 5 independent reps,
the conclusion becomes statistically defensible.

## After Run 35 — branching plan

**If 5-rep CV < 10% across all configs and m8r×20 ≈ 320 stable:**
propagate the durable facts to `knowledge.md`, then pick up the
long-term backlog in `todo.md` (Lever 1 next).

Specifically propagate to `knowledge.md`:

- Update the Lever 3 row in "Active levers" with the 5-rep
  production-scale validation.
- Add: optimal `num_workers` per rank count (NW=20 wins at 8r, NW=12
  wins at 4r, NW=12 ≈ NW=20 at 1r — per Run 33; confirm with Run 35).
- Add: real DDP scaling efficiency at 4r and 8r with 5-rep error bars.
- Add: empirical `sps_wall` noise floor on this benchmark.

**If 5-rep CV > 10% on m8r×20 or s4r×20:** the metric is noisier than
we thought. Two follow-ups before propagating anything:

1. Add per-rank sample-completion timestamps to the merged dataloader
   so we can see if specific ranks are systematically slow under load.
2. Consider whether to add a third repetition mechanism — e.g. each
   rep runs each config 3 times instead of 5 reps total, giving 15
   data points per config but only 6 h × 9 = ~54 h.

## File-opens probe — Run 36 (independent of the 5-rep chain)

Job `41930447` ("file_opens") is queued behind the tail of the 5-rep
chain. When it lands, pull and analyse:

```bash
/hpc-pull
ls jobs-map/41930447/file_opens.csv
```

CSV schema: `dataset,sample_idx,var,n_chunks` — one row per `(sample,
variable)` pair with ≥ 1 chunk read. Suggested first cuts (one-liners
against the CSV):

- **Per-sample total opens**, distribution by dataset:
  `awk -F, 'NR>1 {sum[$1","$2]+=$4} END {for (k in sum) print k","sum[k]}'`
- **Per-modality-call chunks**, mean by `(dataset, var)`:
  `awk -F, 'NR>1 {sum[$1","$3]+=$4; n[$1","$3]++} END {for (k in sum) print k","sum[k]/n[k]}'`

**What we're verifying:**

1. Theoretical 2.25 chunks-per-modality-call (uniform-straddling on
   256² patches over 512² chunks). Empirical mean per `(dataset, var)`
   should land near 2.25 if land-pixel sampling doesn't bias toward
   chunk-aligned positions.
2. Reconcile the `knowledge.md` numbers: merged total should be ~6.75;
   single total should be **~92** (= 41 isel calls × 2.25), NOT the 41
   currently quoted in the Lever 3 estimate. If the empirical single
   total is ~92, the Lever 3 speedup story in file-open terms is
   ~92 → 6.75 (≈14×), not 41 → 6.75 (≈6×). Update `knowledge.md`
   accordingly.
3. Distribution shape per modality call: theoretical breakdown is
   25% × 1 chunk, 50% × 2 chunks, 25% × 4 chunks under uniform `(y0,
   x0)`. If the 1-chunk fraction is substantially > 25%, land-pixel
   sampling is biased toward chunk-aligned positions.

When Run 36 results land, fill `results.md` § Run 36 with raw
measurements only, then propagate the durable findings to
`knowledge.md` (correct the Lever 3 file-open count claim).

## Profiling-job follow-up (independent of the 5-rep chain)

The 6 nsys traces from `41817507`–`41817512` are unanalysed. When the
user has bandwidth, propose either:

- Open one merged + one single trace in `nsys-ui` and diff the
  dataloader NVTX range (visual inspection of the worker-stall
  mechanism).
- Run `nsys stats --report=osrt` on a single + merged pair and tabulate
  futex/sem_wait/pread proportions. Confirms whether the 15× sqlite
  size delta is mostly futex traffic (worker IPC stall) or pread
  traffic (cold reads).

## Quick reference: archived/discarded job IDs (do not include in analysis)

| Job(s) | Why discarded |
| --- | --- |
| `41788795`–`41788797` | single 1r×{8,12,20}, contaminated by Run 31 concurrent merged |
| `41788801`–`41788803` | single 8r×{8,12,20}, contaminated by Run 32 concurrent merged |
| `41788804`–`41788812` | Run 31 merged stream — concurrent with single stream |
| `41798193`–`41798201` | Run 32 merged — concurrent or cancelled |
| `41811414`–`41811416` | Run 34 verification — only 1 ran, superseded by Run 35 |
| `41812997`–`41813002` | first profiling chain attempt — failed (config-key error, retried) |

Clean Run 31 single 4r references (use only if needed for direct
cross-check against Run 35): `41788798` (4r×8), `41788799` (4r×12),
`41788800` (4r×20). Run 33 (`41799685`–`41799702`) is fully clean and
should be the primary cross-check baseline.
