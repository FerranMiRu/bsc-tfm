# Thesis

## Abstract

## Acknowledgments

- Pasha, Marina, Amanda, Joan Vedri
- Oriol, Silvia
- My girlfriend, my family, friends, and everyone who supported me during this journey.

## Introduction

- Explain what is AI4Land (use already done text)
- Tools that are used (python, pytorch, slurm, uv, zarr, etc.)
- How currently the pipeline works
- Optimization is more art than science
- Why is it important to optimize the training of deep learning models?

## Objectives

- Achieve the fastest possible training time
- If possible enable seeing more world in the same amount of time

## Methodology

### Literature Review

- look for LitReview on the other ai4land papers
- explain that this is a very applied project
- since this is new

## Results

### uv implementation

- Conda problems on MN5 that motivated the migration:
  - Conda manages Python packages AND system libraries, which leads to environment
    pollution: you can no longer tell exactly which version of a given system library is
    being used.
  - The AI4Land team shared a single Conda environment across all branches; any dependency
    change applied to one branch silently affected everyone, and rolling back to reproduce
    a previous result was hard.
  - The dependency solver is slow: a fresh AI4Land environment took three to ten minutes
    to build, which becomes a real bottleneck when a development branch needs a different
    dependency set.
- How uv solves those issues:
  - Clean environment builds in seconds; lockfile pins every transitive dependency to an
    exact version with hash digests.
  - Each branch can have its own environment without duplicating package bytes in memory,
    thanks to the shared content-addressed cache.
  - Standard `pyproject.toml` interop; no Conda-channel lock-in.
  - Aside: since the arrival and stabilization of uv, a sizeable fraction of the Python
    scientific-computing community has been migrating off Conda for these same reasons.
- Movement to packaged code:
  - A standard Python package layout under a single source directory; the project becomes
    installable.
  - Entry-point scripts import from the package rather than from sibling files: now
    launchable from any working directory under SLURM, independent of where the checkout
    lives.
  - Enables clean side-by-side comparison between branches in concurrent SLURM allocations.
- Application of ruff:
  - Single Rust-based linter and formatter replaces a stack of legacy Python tools.
  - Enforced via pre-commit hooks so commits never carry stylistic noise; diffs only show
    substantive changes, which is critical when comparing single-knob A/B branches.
- Creation of a testing script:
  - Smoke test that runs a short training epoch; used as a regression gate before any
    perf-sensitive change is merged.
  - Parity test that asserts tensor-level equivalence between layouts before any
    perf comparison.
- Scientific value:
  - Faster iteration on experiments
  - Better reproducibility
  - Easier read of the code

### Profiling

- Motivation: we cannot optimize without knowing where time goes. First step is to
  instrument a representative training run.
- Tooling overview (already introduced in Introduction): Nsight Systems for the
  system-level timeline, NVTX ranges for application-level annotation, plus a per-batch
  wall-clock log written by the training loop.
- [Profiling image] Nsight Systems timeline of a training step on the multi-zarr
  baseline. The GPU sits idle most of the time; the CPU side is dominated by dataloader
  fetches.
- Per-batch wall-clock log shows a clear periodic pattern: roughly twelve sub-second
  batches, then one slow batch on the order of ten seconds, then another twelve, and so
  on. The twelve-batch cadence matches the dataloader's worker count of twelve rotating
  through one full prefetch cycle.
- Inside an Nsight Systems trace, each DataLoader worker thread spends the bulk of its
  recorded time blocked in a kernel-level wait operation. That signature can mean either
  "the worker is idle on the inter-process queue waiting for work" or "the worker is busy
  inside its per-sample fetch, on a library that itself uses kernel-level waits". The
  trace alone cannot tell those two states apart.
- These observations frame the optimization campaign: the dataloader is the bottleneck,
  and the twelve-batch periodicity tells us workers all stall at the same point in their
  prefetch cycle. The next subsections work through candidate causes.

### Synchronizer removal

- First hypothesis on the kernel-level wait pattern: the zarr library's process
  synchronizer is a file-based lock intended to make multi-process concurrent writes
  safe. We were enabling it on every store. The lock's wait operations show up in
  Nsight Systems traces with the same signature as the worker stalls we were trying
  to explain, so it was a natural suspect.
- A/B with the synchronizer enabled versus disabled:
  - Run 1 (in training loop, fifty batches each): identical batch-time patterns,
    identical spike magnitudes.
  - Run 2 (standalone dataloader profiler, single-thread): per-sample fetch mean
    1177 ms with the synchronizer on versus 1178 ms with it off, within run-to-run
    noise.
  - Run 7 (Nsight Systems summary): the kernel-level wait share of recorded time
    was 89.2 % with the synchronizer on and 82.9 % with it off, essentially the same.
- Source-level confirmation: in the version of zarr that we use, the lock is only
  acquired on the chunk-write path, not on the chunk-read path. Since the dataloader
  reads from the store and never writes, the synchronizer is a structural no-op on
  our workload.
- Conclusion: the synchronizer is not the source of the wait pattern; the wait we see
  must come from somewhere else (the inter-process queue, the parallel filesystem's
  client daemon, or the compression library's internal threading). The configuration
  knob was removed; the synchronizer instance itself is kept defensively in case
  concurrent writers are ever added.

### Multizarr vs singlezarr

- Original AI4Land layout was multizarr: one zarr store per modality (LUH2, HILDA+,
  static, Köppen-Geiger, population), five stores opened per worker on every sample.
  Each open pays the same metadata round-trip cost and forces five times the directory
  bookkeeping per fetch.
- The hypothesis to test: collapsing the five stores into one ARCO-compliant store
  should remove the per-store fixed overhead and produce more coherent access patterns
  on the parallel filesystem.
- Datasets used in the comparison (table):
  - multizarr: 606 G LUH2 + 4.5 G HILDA + 4.1 G static + 4.4 G KG + 61 G population
  - singlezarr 1960-2015: 213 G
  - singlezarr non-preprocessed 2000-2015: 66 G
  - singlezarr preprocessed 2000-2015: 64 G
- [Add results — multizarr vs singlezarr twelve-batch spike comparison]
  - Multizarr: ~80 s spikes every twelve batches.
  - Singlezarr: ~10 s spikes every twelve batches.
  - Same periodicity; about an 8x reduction in spike magnitude.
- This was an unambiguous win and bundling became the new baseline.
- New surprise: the 213 G singlezarr (covering 1960-2015) had spikes 1.9-3x larger than
  the 64 G singlezarr (covering 2000-2015), under identical chunk layout and patch size.
  - [Add results — store-size effect plot, preprocessed (64 G) vs preprocessed-big
    (213 G), BATCH 10 and 11]
- This raised a new question that the rest of the chapter addresses: why does the total
  on-disk size of a store change how long it takes to read one patch from it?

### Dataloader isolation

- In-loop profiling conflates dataloader cost with model forward and backward. To
  attribute numbers reliably, we built a standalone dataloader profiler that runs
  without the training loop:
  - Phase 1: fifty single-thread per-sample fetches; per-modality timing.
  - Phase 2: load versus preprocess split inside each per-modality fetch.
  - Phase 3: worker-count sweep over zero, one, four, twelve, each running for eighty
    batches in a fresh subprocess so the child cannot inherit configuration state.
- Per-sample cost (singlezarr 2000-2015, single-thread baseline):
  - Per-sample fetch mean ~1.07 s.
  - Sample output tensor 9.5 MB after stacking.
  - Effective per-reader throughput ~9 MB/s.
- [Add per-modality results figure]
  - LUH2 ~314 ms (60-68 % of the per-sample fetch).
  - Static ~62 ms, population ~52 ms, HILDA ~48 ms each, KG ~42 ms.
- Load versus preprocess split: load accounts for ~92 % of per-sample cost; preprocess
  accounts for ~8 %. Any prep-side optimization is capped at 8 %; effort must go into
  the read path.
- Worker scaling: ~0.8 samples/s per worker, linear from zero through twelve
  (Run 4: 0 → 0.8, 1 → 0.9, 4 → 3.3, 12 → 9.6 samples/s). The headline twelve-worker
  throughput of 9.6 samples/s is exactly worker count divided by per-sample time,
  confirming the dataloader is bottlenecked on per-sample work and not on queue
  mechanics.
- Pre-baked preprocessing (the preprocessed 2000-2015 store, with all fill-NaN and
  normalize steps baked in on disk) measured identical per-sample time and identical
  twelve-worker throughput. This rules out the on-the-fly fill-NaN and normalize cost
  as a real contributor; the residual ~80 ms of preprocess time is the in-memory
  stacking, transpose, and reshape, which the bake-in does not remove.
- The cost is localized to the LUH2 read path; the next subsection drills into a single
  chunk read on that path.

### Tests on synthetic data

#### The obvious fix that did not work (Lever A, brief)

- Following the dataloader-isolation finding (fourteen LUH2 variables loaded in a
  serial loop), the obvious move was to parallelise the fourteen loads with a thread
  pool.
- Run 8 sweeps thread counts of one, two, four. Per-LUH2-call load: 308 → 380 → 322 ms.
  Two threads _regressed_ by +23 %; four threads recovered to baseline. End-to-end
  twelve-worker throughput: 8.5 / 9.0 / 9.1 samples/s, flat within run-to-run noise.
- Pure I/O bottlenecks do not get worse when a thread is added. The two-thread
  regression is the fingerprint of CPU or lock contention: most likely a known
  limitation of the compression library when called from a thread pool, combined with
  Python's Global Interpreter Lock (the kernel-level mutex inside the Python
  interpreter that lets only one thread run Python code at a time, abbreviated GIL
  hereafter) serializing the xarray layer between threads.
- Conclusion: the obvious parallelism lever is closed. We need to understand what a
  single chunk read is actually doing.

#### Anatomy of a single chunk read

- A targeted probe (the per-chunk anatomy probe, jobs 41543368 and 41545383) wraps one
  chunk read in four nested timers, each measuring one layer of the access path:
  bytes-from-disk (GPFS metadata plus byte transfer), zarr index lookup,
  decompression, and the xarray slice-and-materialize step.
- Per-chunk breakdown (cold cache, real production store):
  - Bytes-from-disk: 19-20 ms.
  - Decompression: 0.3 ms (~3 GB/s effective; negligible).
  - Zarr index lookup (warm): ~0 ms.
  - Xarray and dask graph work during slice-and-materialize: 16 ms, cache-invariant
    (the same warm and cold), runs under the GIL.
- Decomposition of the 314 ms LUH2 baseline:
  - Cold GPFS metadata plus bytes: ~107 ms (fourteen chunks at ~7-8 ms effective; the
    filesystem's server-side prefetch absorbs much of the worst-case cold round-trip).
  - Xarray and dask graph: ~224 ms (fourteen chunks at 16 ms each).
  - In-memory residual (stacking, transpose, reshape): ~80 ms.
  - The components partially overlap because the C-level bytes-from-disk step releases
    the GIL while bytes are in flight, which is why the sum exceeds the observed
    314 ms wall time.
- Two surprises:
  - Decompression is not the bottleneck. The codec decodes at roughly three gigabytes
    per second and contributes about one per cent of the per-LUH2-call time.
  - The per-chunk xarray and dask cost is cache-invariant. It runs under the GIL and
    accounts for roughly two thirds of the per-LUH2-call cost by itself.

#### How much do we actually read, and what arrives at the model?

- Chunk byte distribution on the real store (Run 12, 100 chunks across each of 23
  variables):
  - Eighteen of twenty three-dimensional variables have median equal to minimum
    compression floor (~4 KB for most variables, ~8 KB for the soil-attribute
    variables). These are all-fill ocean chunks, which the codec compresses to nothing.
  - LUH2 variable means range from 42 KB (urban) to 163 KB (c4ann); 95th percentile
    around 700-800 KB.
  - Static soil-attribute variables are the largest: means 330-374 KB, 95th percentile
    around 1.5 MiB.
- Per-sample bytes (single-zarr layout):
  - 28 LUH2 chunks (14 variables times 2 timesteps) plus 2 population, 2 KG, 6 static,
    1 HILDA prior, 2 HILDA target equals **41 chunk-file opens per sample**.
  - Compressed bytes per call (means): ~2.5 MiB on the LUH2 slice, ~3 MiB across all
    forty-one chunks.
  - Worst-case 95th-percentile land chunks: ~19 MiB on the LUH2 slice.
- Output sample tensor (uncompressed, after stacking, transpose, and cast): ~9.5 MB.
  The disk-to-tensor compression ratio is about three to one; the remaining tensor
  bytes are filled in by the per-chunk Python overhead.

#### Theoretical read speed and the gap

- GPFS bandwidth measurements on a single MN5 ACC node (using a synthetic block-copy
  benchmark with one-megabyte reads):
  - Single-stream read, with the OS page cache enabled: 556 MB/s.
  - Twelve parallel streams over separate four-gigabyte files: aggregate 2510 MB/s,
    per-stream 209 MB/s.
- If the read path were bandwidth-bound, the per-sample 3 MiB at 556 MB/s would take
  ~5 ms. The observed ~1.07 s is roughly 200x slower than the bandwidth ceiling.
- The two-hundred-fold gap is overwhelmingly per-file-open cost. Bytes are not where
  the time goes.

#### Direct test: GPFS metadata cost vs file count

- Synthetic file-count sweep (Runs 13 and 15): write N files of size S, read each in
  sequence (Run 13) or in shuffled order (Run 15), time the warm path.
  - Warm per-open cost is ~0.11 ms, essentially flat across file counts from ten to
    ten thousand at fixed size in the ten-to-hundred-kilobyte range. Pure metadata;
    bytes are noise.
  - At constant 100 MB total: one 100 MB file takes 44 ms versus ten thousand 10 KB
    files at 1082 ms (24.6x).
- Cold-cache measurement (Run 14, two pools of single-directory files at 10 KB each,
  ten thousand random reads per pool, interleaved):
  - 95th-percentile per open ~23 ms, matching the per-chunk anatomy probe's 19-20 ms
    cold bytes-from-disk independently.
  - One hundred thousand versus one million file pools give identical distributions
    (within 8 %): directory size is _not_ the cause of the per-open inflation we are
    after.
- Synthetic byte distribution matched to the real store's distribution (Run 17 versus
  Run 18): median wall within 19 %, 95th percentile within 5 %. Synthetic benchmarks
  are a valid proxy for the real production store once bytes match.

#### Production-shape A/B simulation

- Goal: simulate the per-sample read pattern of the production singlezarr (40
  chunk-file opens) against a hypothetical merged layout (2 opens), under twelve-worker
  concurrency.
- Two synthetic stores at the production scale (~30 GB each, ~300 K
  operating-system file records for A versus ~13 K for B):
  - Store A: 23 subdirectories of ~13 000 files, sizes sampled from the real chunk
    distribution (Run 12). Forty random opens per sample.
  - Store B: 1 subdirectory of 13 000 files, each ~2.3 MB (one merged "chunk"). Two
    opens per sample.
- Read pattern: twelve workers via a process pool, one hundred samples per worker,
  alternating access patterns (A-then-B-then-A, and A-then-B-then-A-then-B) to control
  for cache state.
- Cold-cache results (Runs 19 and 21, fresh job, no creation-time cache residue):
  - First A pass ~668 ms, first B pass ~90 ms; second A pass ~530 ms, second B pass
    ~82 ms.
  - A drift across passes: 1.25-1.28x (small, stable cold).
  - B drift across passes: 1.08x (small, stable).
  - Cold-cache A versus B ratio robust at five to eight across all percentiles.
- Cache-fit regime flip (Run 20, smaller stores: 31 K versus 1.35 K
  operating-system file records, both fit in the Linux dentry cache, the in-kernel
  cache of recently-resolved file path components):
  - First A pass 39 ms (still cold-ish) → second A pass 8.6 ms (warm) inside a single
    job, a 4.54x speedup just from cache warming.
  - B 16 ms; once metadata fits in cache, A is _faster_ than B because per-open
    collapses to ~200 µs and B still has to transfer about two megabytes per open.

#### Why the store-size effect was real

- This subsubsection answers the question that opened the chapter: why does the
  per-sample read time scale with the total on-disk size of the store?
- Each chunk-file open requires the operating system to resolve the file's path. The
  result of that resolution is cached in the Linux dentry cache, an in-kernel cache
  of recently-resolved path components. When the working set of chunk files fits in
  the dentry cache, each open is satisfied from memory and costs about a hundred
  microseconds. When the working set exceeds the cache, each open requires a network
  round-trip to the GPFS metadata server, which costs about twenty milliseconds. The
  per-open cost in the second regime is therefore about two hundred times higher
  than in the first.
- The 64 G singlezarr store has on the order of forty thousand chunk files per
  variable. This is on the edge of the dentry cache and the working set can warm
  during a job, so most opens hit the cheap regime.
- The 213 G singlezarr store has on the order of one hundred and forty thousand
  chunk files per variable. This is well past the cache, so most opens stay in the
  expensive regime.
- The 1.9-3x spike inflation observed in BATCH 10 and 11 is exactly this warm-vs-cold
  regime transition expressed at a different scale. The mechanism is the same as the
  per-sample slowness: GPFS metadata round-trips dominate, and whether the working
  set fits in cache decides whether each open is cheap or expensive.

#### What the synthesis points to

- GPFS metadata cost per chunk-file open dominates the read path. Bytes,
  decompression, worker count, the zarr synchronizer, and the choice of codec are
  all ruled out by independent measurement.
- The right lever is to collapse the per-sample file-open count from forty-one to as
  few as possible, not to change chunk size, codec, or worker count. That is what the
  next subsection does.

### Singlezarr vs mergedzarr

#### Design of the merged store (Lever 3)

- Stack the fourteen LUH2, one population, and one KG dynamic variables into a single
  "dynamic" array (with variable as a new axis), and the six static variables into a
  single "static" array. Categoricals are stored as floating-point and cast back at
  read time.
- Patches stay at 256 × 256; chunks stay at 512 × 512 spatially. The change is purely
  layout: the same bytes live in a much smaller set of files.
- Per-sample chunk-file opens: 41 → ~6.75. Each 256 × 256 patch straddles, on average,
  2.25 of the 512 × 512 spatial chunks per modality call (with probability 0.25 of
  hitting one chunk, 0.5 of straddling two, and 0.25 of straddling four under uniform
  random patch placement), and there are three modality calls per sample (dynamic
  times two timesteps, plus static).

#### Parity check

- Before measuring performance, we verify that the merged store and the single-zarr
  store return bit-identical samples (Run 23, parity test): 32 samples × 5 tensors ×
  7 statistics → 0 mismatches at one part in ten billion. Categorical value sets and
  sentinel checks all match. This rules out any data-side concern from the layout
  change; only the performance effect is in question.

#### Per-sample performance

- Phase 1 (single-thread per-sample fetch, fifty samples per dataset):
  - Run 22: single ~1.55 s mean → merged ~0.36 s.
  - Run 24: single 1411 ms → merged 201 ms. Ratio 7x mean, 8.85x median.
  - Run 26: single 1385 ms → merged 194 ms. Ratio 7.2x median (consistent across
    runs, independent of cache state).
- Worker scaling for merged is essentially 100 % efficient from zero workers (3.4-5.8
  samples/s) to twelve workers (40-70 samples/s). The variance between Run 24's 40
  samples/s and Run 26's 70 samples/s at twelve workers is GPFS and OS page-cache
  state, not a structural cost.
- [Add Phase-1 distribution figure single vs merged]
- [Add throughput-vs-workers figure single vs merged]
- The twelve-batch worker-cycle spike pattern is structurally gone in merged: a
  prefetch refill now opens six or seven files instead of forty.

#### Per-method breakdown of the merged per-sample fetch (Run 25)

- One hundred samples, single-thread, every step wrapped with a high-resolution
  timer:
  - Dynamic-variable read and materialize: 90 ms median (51 %).
  - Static-variable read and processing: 60 ms median (34 %).
  - In-memory continuous-tensor assembly: 15 ms median (9 %).
  - Lazy index construction (the deferred-I/O step that defers actual I/O): 1.4 ms
    median (1 %), essentially free.
  - Other per-modality extracts: < 1 ms total.
  - Sum of parts is 96 % of the total fetch time; no hidden overhead.
- Two implications for what comes next (Outlook):
  - The cost is not in the lazy index construction (the deferred-I/O step); the cost
    is inside the slice-and-materialize step, where the xarray graph work, the GPFS
    bytes-from-disk, and the dask compute all materialize together.
  - Any follow-up xarray bypass must target both the dynamic-variable read and the
    static-variable read to recover the full budget; leaving the static side on
    xarray gives back a third of the potential gain.

#### End-to-end DDP training

- 4-rank Distributed Data Parallel on one MN5 ACC node, identical model and
  hyperparameters, batch size 8, accumulation 4, twelve workers per rank,
  prefetch factor 4.
- Short run (Run 27: 15 train steps × 1 val × 1 test, 3 repetitions each):
  - Single mean total: 118.9 s. Merged mean total: 21.2 s. **5.61x total speedup**
    (train 4.64x, val 7.69x; the val ratio is inflated by single's prefetch-queue-fill
    cost at a single validation step).
- Longer run (Run 28: 200 train × 30 val × 30 test, 3 repetitions each):
  - Single mean total: 354.1 s. Merged mean total: 76.5 s. **4.63x total speedup**
    (train 4.65x, val 4.51x; the val number is now properly amortized, this is the
    honest end-to-end ratio).
- Per-batch spike pattern (Run 28, rep 3):
  - Single: 17 spikes per 200 batches at 10-34 s each, exactly every twelve batches
    — the structural worker-cycle stall.
  - Merged: 1 single spike of 2.7 s. The structural stall pattern that dominated
    historical singlezarr training is _eliminated_.
- [Add per-batch timeline plot single vs merged]
- Cache warming across the three reps is monotonic and large for merged (2.3x in
  ~5 min of wall time) but non-monotonic and noisy for single (rep 2 faster than
  rep 3). Single's spikes warm inconsistently across the forty-one-files-per-sample
  pattern; merged's six-or-seven-files-per-sample pattern warms cleanly.

#### Scale-out behaviour

> TODO: the headline numbers in this subsubsection come from a single Run 33 sweep.
> Five additional repetitions of the same 18-job sweep are planned in the final week;
> swap in the multi-run means and standard deviations once those land.

- Clean serialized intercalated sweep (Run 33, 18 jobs, 5 epochs of 300 train steps
  each): every (rank count, worker count) combination measured exactly once, with
  single and merged alternating between jobs so cache state cannot persist within a
  dataset. Aggregate throughput is computed from epoch wall time (excluding the cold
  epoch 1), not from median batch delta, which undercounts the tail spikes that
  single still has.
- Headline numbers (samples/s aggregate):
  - Single best: 8 ranks × 20 workers = 52 samples/s.
  - Merged best: 8 ranks × 20 workers = 320 samples/s. **6.1x single's best.**
  - Per-rank gain amplifies with rank count: 4.1x at 1 rank, 4.8x at 4 ranks, 6.1x at
    8 ranks.
- DDP scaling efficiency on merged at each rank count's best worker count:
  - 1 → 4 ranks: 2.91x (73 % of ideal 4x) at twelve workers.
  - 1 → 8 ranks: 6.51x (81 % of ideal 8x) at twenty workers.
- Optimal worker count depends on rank count:
  - Merged at 1 rank: twelve workers ≈ twenty workers (~49 samples/s).
  - Merged at 4 ranks: twelve workers best (145 samples/s).
  - Merged at 8 ranks: twenty workers decisively best (320 samples/s).
- [Add scale-out heatmap ranks × workers, single and merged side by side]
- The cap on the merged layout at 8-rank DDP is per-rank cold-cache variance, not the
  per-sample dataloader work. With merged's six-to-seven opens per sample, a single
  rank getting unlucky with cold metadata can stall the whole step. This motivates the
  Outlook items below.

## Conclusions

- Bottleneck identification:
  - Profiling localized the dataloader as the dominant cost (more than 90 % of training
    wall time on the production singlezarr at this scale).
  - Layered diagnostic work attributed the per-sample cost to GPFS metadata round-trips
    on the many small files that a zarr store decomposes into, ruling out preprocessing,
    the zarr synchronizer, the codec, the bytes, and the worker count along the way.
- Main contribution (the durable knowledge):
  - The chunking layout of a zarr store is the single largest performance variable for
    any dataloader that reads patches from a chunked store on a parallel filesystem.
    Before this work, the AI4Land team treated chunking as a convenience choice; the
    measurements collected here show that careless chunking can suppress throughput by
    an order of magnitude, and that the same set of stores can transition from fast to
    slow as the working set outgrows the operating system's metadata cache.
  - This knowledge is transferable to any geospatial deep-learning pipeline that
    ingests patches from a chunked store. It is independent of the specific
    deliverables described below and is the central scientific contribution of the
    thesis.
- Optimizations delivered:
  - Migration to uv and packaged code: reliable, side-by-side A/B SLURM runs and
    branch-local environments without environment pollution.
  - Single-zarr layout (one ARCO-compliant store with all modalities): about 8x
    reduction in the dataloader spike magnitude over the historical multizarr layout.
  - Variable-merging "merged-zarr" layout (collapses 41 per-sample file opens to ~6.75):
    about 7x per-sample speedup in single-thread, about 4.6x end-to-end at 200 train
    steps under 4-rank DDP, about 6.1x best-to-best at 8-rank DDP.
  - The structural twelve-batch worker-cycle stall that dominated historical training
    is eliminated under the merged layout.
- Implementation tradeoffs:
  - The merged-zarr layout requires storing the dataset twice on disk (the source
    layout still feeds other consumers; the merged layout is read-only for training).
    Whether to materialize the merged store in production depends on the dataset's
    total size, its retention policy, and how often it is updated. The decision is
    out of scope here.
  - The underlying knowledge stands regardless. Alternative implementations (an
    in-place re-chunking, an online merge pass during training, or a smaller chunked
    dataset for repeat experiments) all benefit from the same principles: collapse
    the per-sample file-open count and keep the working set within the operating
    system's metadata cache.
- Methodology contribution:
  - Three-phase standalone dataloader profiler (single-thread, load-versus-prep,
    worker sweep) that decouples per-sample cost from queue mechanics and model
    compute.
  - A reusable synthetic GPFS probe protocol (file-count × file-size ×
    directory-size, with controlled cache state) that diagnoses metadata-versus-
    bandwidth-versus-codec causes on any parallel filesystem.
  - Synthetic benchmarks shown to extrapolate to production within ~20 % when bytes
    are matched to the real chunk distribution.
- Transferability:
  - The optimizations are layout-level, not model-level, so they propagate to any
    training pipeline that ingests patches from a chunked geospatial store. The
    ELLIOT foundation-model effort that intends to reuse this pipeline inherits the
    speedups directly.

## Outlook

- Bypass xarray inside the per-sample fetch (Lever 1):
  - Run 25 attributes 51 % of the merged per-sample budget to the dynamic-variable
    read and 34 % to the static-variable read; both rely on xarray's slice-and-
    materialize step, which costs about sixteen milliseconds per chunk under the
    Python GIL.
  - Replacing the call with the raw zarr access path releases the GIL during the
    decompression step (which makes that step threadable per timestep) and removes
    the per-chunk graph overhead. The predicted floor is about 35-45 ms per sample
    (versus about 175 ms today), about 4x more per-sample gain on top of the merged
    layout. The bypass must attack both the dynamic and static reads; leaving the
    static side on xarray gives back a third of the gain.
- Upgrade to the next major version of zarr with built-in asynchronous concurrency
  (Lever 4):
  - The existing stores remain readable without recreation. The new version
    parallelises per-chunk metadata round-trips, which directly attacks the
    remaining cold-cache cost without recreating the store.
  - Smaller predicted gain than Lever 1 on our pattern (~1.2-1.4x on the merged
    single-thread fetch) because it does not remove the cache-invariant xarray
    graph cost. Composes with Lever 1.
- Theoretical ceiling and the regime flip:
  - Compressed bytes per sample are about 3 MiB; GPFS single-stream bandwidth is
    556 MB/s. The bandwidth-bound floor per sample is therefore about 5 ms. Per-rank
    throughput at twelve workers in this regime would be about 2200 samples per
    second.
  - Current merged at 8-rank × 20-worker DDP delivers about 320 samples per second
    aggregate, about 40 per rank. The dataloader is still about fifty times above
    the bandwidth floor.
  - At the bandwidth floor, the dataloader stops being the bottleneck. Model
    forward and backward time per sample then dominates the loop, which is the
    proper end-state for an HPC deep-learning workload: the system transitions
    from being I/O-bound to being compute-bound.
