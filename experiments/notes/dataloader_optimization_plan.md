# Dataloader Optimization Plan

## Context

Profiling runs on MN5 (15-batch jobs, `num_workers=12`, `batch_size=8`, `accumulation_steps=4`) revealed:

- GPU consumption per batch: **~150ms** (steady state, observed in profile window 6–10)
- Singlezarr / preprocessed loaders: fast when prefetch queue is warm
- Multizarr: consistent **~80s** stall around batch 12 across every job
- The "batch 12 spike" coincides with both `num_workers=12` (worker cycle) and `accumulation_steps=4 × 3 = 12` (first optimizer step inside profiling window)
- After the spike, multizarr returns to fast batches — but 15 batches isn't enough to know if spikes recur

## Open question

Does the batch-12 stall **recur every ~12 batches** in steady state (real-training disaster), or is it a **one-time prefetch-drain transient** (acceptable)?

A 50-batch run without profiling will answer this.

## Knobs identified

| Knob                 | Current                               | Notes                                                                                                  |
| -------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `num_workers`        | 12                                    | CPUs per task = 20                                                                                     |
| `batch_size`         | 8                                     | Effective with accumulation = 32                                                                       |
| `accumulation_steps` | 4                                     |                                                                                                        |
| `prefetch_factor`    | 4                                     | Hardcoded in `base_trainer.py:167,176`                                                                 |
| `persistent_workers` | False                                 | Hardcoded in `base_trainer.py:166,175`                                                                 |
| `pin_memory`         | True                                  | Hardcoded, already optimal                                                                             |
| `synchronizer`       | `ProcessSynchronizer` on every zarr   | **`datasets.py:108–111` and `datasets.py:715–718`** — only needed for concurrent writes. We read only. |
| Filesystem location  | GPFS (`/gpfs/scratch/ehpc606/...`)    | Shared, high per-op latency                                                                            |
| Data layout          | multizarr / singlezarr / preprocessed | Singlezarr/preprocessed already a big win                                                              |

---

## Phase 1 — Baseline (50-batch run, no profiling)

**Goal:** Establish per-batch wall times across 50 batches without nsys overhead. Determine whether multizarr's batch-12 stall is transient or periodic.

**Actions**

1. Set `train_steps_per_epoch: 50` in all 3 profiling configs.
2. Keep `profiling.enable: false` (already disabled).
3. Keep `log_loss_every_batch: true` (already enabled).
4. Submit via `scripts/acc_training.sh` (no nsys wrapping).

**Decision rules after Phase 1**

- **Multizarr spikes recur every ~12 batches:** sustained I/O bottleneck → jump straight to Phase 2 Test 2.1 (remove synchronizer) and Phase 4 (NVMe staging).
- **Multizarr settles after one transient:** measure steady-state rate. If close to GPU consumption (~150ms), Phase 2 is enough. If still slow, Phase 4 likely needed.
- **Singlezarr/preprocessed are GPU-bound after Phase 1:** Phase 3 (batch size) becomes the next lever.

---

## Phase 2 — Free wins (no infrastructure changes)

Run tests in order of expected impact, against the loaders most affected. Change **one** knob at a time.

### Test 2.1 — Remove `synchronizer` from training-path zarr opens **[NEW, top priority]**

- Change `xr.open_zarr(..., synchronizer=synchronizer)` → `xr.open_zarr(...)` for training-mode opens in `SingleZarrDataset._init_dataset` and `MultiZarrDataset._init_stores`. Keep synchronizer for any path that writes (preprocessing scripts).
- **Hypothesis:** eliminates per-chunk file-lock acquire/release on GPFS, drastically reducing metadata-server pressure. Should disproportionately help multizarr (5× zarrs × locks).
- **Risk:** none for pure-read training. Verify no writer path shares the same zarr.
- **Effort:** trivial diff. Run against multizarr first.

### Test 2.2 — `persistent_workers=True`

- Change in `base_trainer.py:166,175`.
- **Hypothesis:** removes ~25s × 10 = 4 min of validation worker respawn in real training. May also retain zarr handles/page cache locality across epochs.
- **Risk:** minor — workers keep some memory between epochs.

### Test 2.3 — `num_workers` sweep

- Test 8, 16, 20 (CPUs per task = 20). Current = 12.
- **Hypothesis:** more workers → larger aggregate I/O throughput, but past ~half CPU count, contention sets in. Sweet spot likely around 16.
- **Note:** if Test 2.1 dramatically improved multizarr, re-baseline first.

### Test 2.4 — `prefetch_factor` 4 → 8

- Hardcoded in `base_trainer.py:167,176`. Move to config first.
- **Hypothesis:** bigger initial buffer delays drain; only meaningful if drain is recurrent (depends on Phase 1 result).

---

## Phase 3 — Batch size / accumulation

### Test 3.1 — `batch_size: 32, accumulation_steps: 1`

- Same effective batch size (32), but fewer dataloader calls per effective step.
- **Hypothesis:** better amortization of per-batch fixed overhead (forward/backward sync, NVTX, accelerate accumulate context).
- **Risk:** GPU memory pressure. Check with `nvidia-smi` after first batch.

### Test 3.2 — `batch_size: 16, accumulation_steps: 2`

- Same effective batch (32). Halfway point.
- Only run if 3.1 succeeds and gives a clear answer about direction.

---

## Phase 3.5 — Remove per-batch GPU→CPU syncs **[only after loader is fast]**

### Test 3.5.1 — Defer loss sync to log boundaries

- `train.py:336-337` currently forces a sync every batch:
  ```python
  if torch.isfinite(loss):        # 0-d GPU tensor → Python bool = sync
      running_loss += loss.detach().item()  # .item() = sync
  ```
- Replace with a GPU-side accumulator; sync only when logging or at epoch end:
  ```python
  running_loss = torch.zeros((), device=accelerator.device)
  ...
  finite_mask = torch.isfinite(loss)
  running_loss += torch.where(finite_mask, loss.detach(), torch.zeros_like(loss))
  ...
  if accelerator.is_main_process and should_log_batch:
      avg = (running_loss / steps).item()  # one sync per log, not per batch
  ```
- **Hypothesis:** removes a GPU→CPU barrier per batch, letting the next batch's CPU-side prep overlap with the in-flight forward/backward. With ~150ms compute and a steady loader, this is non-trivial overhead. With our current loader spikes (seconds), it's invisible — hence "only after Phase 2/4".
- **Risk:** none functional. Validate NaN handling is preserved (the masked accumulation still drops non-finite losses).
- **Pairs well with:** flipping `compile: true` (Phase 3 sibling) — both are GPU-starvation fixes that only pay off post-loader-fix.

---

## Phase 4 — Filesystem (biggest potential for multizarr)

### Test 4.1 — Pre-stage data to local NVMe

- Add sbatch prologue that rsyncs the zarrs to `$TMPDIR` (or `/scratch/local/` on MN5 ACC) once per node at job start.
- Point the config's `data.stores` at the staged path.
- **Hypothesis:** removes GPFS from the hot path. Could turn multizarr from I/O-bound to GPU-bound.
  - Reference: NVMe vs GPFS read perf ≈ 9× (HPC research — see plan sources).
- **Effort:** real engineering — sbatch script changes, path templating in config. Amortizes well for long training runs (one-time copy cost).
- **Caveats:**
  - Node-local disk size must fit zarrs.
  - Copy time at job start adds startup overhead — only worth it for training runs ≫ stage time.
  - Multi-node training needs staging on every node.

---

## Multi-GPU / multi-node considerations (apply to every phase)

- Each DDP rank has its own `DataLoader`. 4 ranks × current 12 workers = 48 worker processes hitting GPFS per node — likely amplifies stall.
- Optimizer step in DDP triggers all-reduce. NCCL over IB adds cost across nodes, especially on optimizer-step batches.
- Per-node page cache state differs — cold-cache transient happens once per node, not once per training run.
- If we pre-stage (Phase 4), staging happens per node.

---

## Sources (research input)

- [PyTorch DataLoader best practices 2026 (Modexa)](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)
- [PyTorch DataLoader Tutorial 2026 (Progressive Robot)](https://www.progressiverobot.com/2026/02/04/pytorch-dataloader-tutorial/)
- [Cloud native data loaders with Zarr (Earthmover)](https://www.earthmover.io/blog/cloud-native-dataloader/)
- [Zarr GPU pipeline for AI/ML (Xarray)](https://xarray.dev/blog/gpu-pipeline)
- [HPC file systems fail for DL at scale (NextPlatform)](https://www.nextplatform.com/2018/10/09/hpc-file-systems-fail-for-deep-learning-at-scale/)
- [HVAC: removing I/O bottleneck for large DL (OSTI)](https://www.osti.gov/servlets/purl/1902810)
- [Speed Up Model Training (PyTorch Lightning docs)](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)
