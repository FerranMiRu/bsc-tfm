# Agents Instructions

## Master's Thesis Project

This repository contains the Final Master's Thesis (TFM) for the author.

### Identity

- **Author:** Ferran Mirabent Rubinat (NIU 1528268)
- **Program:** Master's Degree in Modelling for Science and Engineering, Universitat Autònoma
  de Barcelona (UAB)
- **Tutors:**
  - Oriol Tintó Prims — Barcelona Supercomputing Center (BSC), external tutor / director
  - Sílvia Cuadrado Gavilán — Universitat Autònoma de Barcelona (UAB), internal tutor
- **Language:** English
- **Target grade:** Cum laude. Quality of analysis, depth of reasoning, and clarity of exposition
  must be at the level expected for the highest distinction.
- **Workload:** Equivalent to two full courses.

### Title

_Performance Optimization of the AI4Land Training Pipeline on MareNostrum 5_

### Purpose

The AI4Land project, developed at the Barcelona Supercomputing Center within the Horizon Europe
projects CONCERTO and TerraDT, uses a UNet-based deep learning model to downscale Land Use and
Land Cover (LULC) data from the Land-Use Harmonization dataset (LUH2) from ~30 km to 1 km
resolution, covering 1850–2100 including future projections based on Shared Socioeconomic
Pathways (SSPs). The pipeline is also intended to be adopted by the ELLIOT project for training
geospatial foundation models. Training on multi-channel geospatial datasets is computationally
demanding and requires efficient use of HPC resources.

### Objectives

Identify and resolve performance bottlenecks in the AI4Land training pipeline running on
MareNostrum 5 (NVIDIA H100 GPUs). Preliminary profiling identifies the data-loading stage as the
primary bottleneck, with epoch-duration variance traceable to the current use of multiple Zarr
stores and suboptimal ingestion patterns. The work will:

- Systematically evaluate optimizations to the data-loading pipeline.
- Explore computational improvements at model and hardware level.
- Assess each optimization with NVIDIA Nsight Systems profiling, quantifying impact on
  throughput and training time.
- Deliver a measurably faster, more efficient pipeline that preserves model accuracy and
  contributes practical guidelines for training deep learning models on large-scale HPC
  infrastructure.

### Report Constraints

- **Page range:** 30–50 pages. Prefer the shorter end. Do not pad to reach the upper limit;
  filler degrades the report.
- **Required sections:**
  1. Title page (UAB logo, title, author, tutors, automatic date)
  2. Summary
  3. Acknowledgments
  4. Content — must frame the question, establish objectives, explain and contextualize results,
     and present conclusions.
  5. References

### Repository

Not under git. The directory is cloud-synced. Do not suggest `git` workflows, commits, branches,
or `.gitignore` files unless the user explicitly initialises a repo.

### AI4Land Source Code

A working copy of the AI4Land codebase that this thesis optimizes lives at
`ai4land-tfm/` inside this directory. Use it as the source of truth for the U-Net architecture,
the training-loop configuration, the dataloader and dataset code, and the SLURM launch scripts.

- `ai4land-tfm/src/ai4land/training/simple_unet.py` — U-Net definition (encoder/decoder widths,
  DoubleConv block, embedding layers for HILDA+ and Köppen–Geiger, RecurrentUNet rollout wrapper).
- `ai4land-tfm/src/ai4land/training/base_trainer.py` — main training loop, DataLoader construction
  (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`), Accelerator init.
- `ai4land-tfm/src/ai4land/utils/datasets.py` — `SingleZarrDataset` and `MultiZarrDataset`, plus
  the `use_synchronizer` config flag that controls `ProcessSynchronizer` use.
- `ai4land-tfm/inputs/unified.yaml` — production hydra config (patch size, batch size, loss
  weights, masking schedule, teacher-forcing decay, learning rate, etc.).
- `ai4land-tfm/scripts/acc_training.sh` — production SLURM launch script (cpus/task, GPUs/node,
  Accelerate launcher invocation).
- `ai4land-tfm/scripts/launch_profiling.sh` — single-rank profiling launch wrapped in `nsys`.

For claims about U-Net size and training parameters, the code is canonical. For claims about
which input variables the model ingests, the published AI4Land papers are canonical because the
input set in the code has drifted since publication.

### LaTeX Setup

- **Compiler:** `pdflatex` (via mactex-no-gui from Homebrew). Bibliography via `bibtex`.
- **Build:** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`, run from
  `report/`. Invoked from Zed's run button. Use `latexmk -C` for a clean rebuild.
- **Document class:** `article`, 12pt, a4paper.
- **Package policy:** Minimal preamble. Add a package only when a feature actually requires it.
  Do not preload "just in case" packages.
- **Page-numbering convention** (intentional — do not "tidy"): title page uses
  `\setcounter{page}{0}` so its hyperref anchor is `page.0` (avoids collision with the body's
  `page.1`). `\pagenumbering{roman}` is set after the title page for the front matter
  (Summary, Acknowledgments, TOC at i, ii, iii). `\pagenumbering{arabic}` is set before the
  body. The title page is built manually with `\thispagestyle{empty}`; do not reinstate
  `\begin{titlepage}` — it forces `\setcounter{page}{1}` at start and end and reintroduces the
  duplicate-identifier warning.
- **Bibliography placeholder:** `\nocite{*}` is currently present so all `refs.bib` entries
  render in the bibliography without explicit `\cite{}` commands. Remove `\nocite{*}` once
  real citations exist in the text.

## Reporting Style

I prefer concise, direct reports that prioritize content over fluff.

### Tone and Language

- **Conciseness:** Be direct. Avoid filler words and decorative adjectives like "crucial", "key
  takeaway", or "vital".
- **Objective:** Focus on facts and analysis.
- **No em-dashes.** Never produce an em-dash in prose: not the Unicode character `—`, not the
  LaTeX `---` triple-hyphen that typesets as one, not the typographic `--` double-hyphen used in
  some Markdown flavors. Use commas, semicolons, parentheses, colons, or rewrite the sentence.
  This rule applies to LaTeX source, Markdown, and plain text. En-dashes for numeric ranges (LaTeX
  `--`, e.g. `1960--2015`) remain acceptable because they are the correct typographic mark for a
  range and have no substitute.

### Diction and Sentence Structure

- **Plain technical prose over compressed jargon.** Concision is good when it cuts filler; it
  becomes bad when it compresses concrete operations into dense noun-phrases. When introducing a
  system, library, or design, describe what it actually does in everyday language. "Uv keeps a
  shared cache folder, and when an environment installs a package that is already there the files
  are shared instead of copied" reads better than "uv uses byte-level deduplication via a
  content-addressed cache". The former tells the reader what happens; the latter only hands them
  a label.
- **Spell out the change being claimed, not the label for it.** A sentence like "branch-local
  experimentation became routine" does not say what actually changed. Rewrite as: "each developer
  can now keep a separate environment per branch cheaply, without disturbing colleagues' work".
  The reader should be able to picture the before-and-after; a label alone does not let them.
- **Short, one-idea sentences with plain connectors.** Prefer two clean sentences joined by
  "Also", "On top of that", "Therefore", or "Because of this" over one sentence stitched together
  with semicolons and embedded "because" clauses. Long sentences are allowed when they spell
  things out, not when they pack qualifications tightly. As a quick test: if a single sentence
  contains more than one "and" connecting major ideas, it should usually be split.
- **Name the human agent.** Use "you", "we", "the team", "colleagues", or "the developer" rather
  than abstract subjects such as "a branch that requires a different dependency set" or "a
  candidate optimization must be testable". Optimization is done by people, and the prose should
  sound like it.
- **Everyday verbs and direct constructions.** "Iterate fast" beats "must permit fast iteration".
  "Interfere with" beats "be attributable to". "Improve the code" beats "deliver workflow
  infrastructure". The simple form is the default; the formal form is only for the rare case
  where the simple form would be ambiguous.
- **Avoid bureaucratic vocabulary** when a plain word will do. Examples: "prerequisites" can be
  written as "the things you need before X"; "infrastructure" as "the workflow and tools";
  "campaign" as "experiments" or "measurements"; "deliverable" as "the first piece of work"; "in
  the narrow sense" as "strictly speaking"; "attributable to" as "caused by" or "due to";
  "configuration" (for code or environment) as "setup".
- **Avoid casual programmer jargon.** "No-op" should be "is not engaged" or "has no effect".
  "Edge case" should name the actual case. "Stub" should be "placeholder". The reader of the
  thesis is a non-specialist; the prose has to read for them.

### Audience Assumptions (Thesis)

When writing for the thesis itself, assume the reader has advanced mathematical training but no
technical or computer-science background. They can follow algebra, calculus, statistics,
optimization, and numerical methods without help, but they should not be expected to know what a
GPU, a filesystem, a semaphore, a Python package, a U-Net, or distributed data parallelism is.

- **Expand every acronym on first use.** No exceptions, including widely-used ones (ESM, GPU,
  HPC, LULC, DDP, NVTX, GPFS, SLURM, SSP, CMIP).
- **Introduce every tool, library, framework, or system with at least one full paragraph**
  describing what it is and what it does, before referring to it in passing. This applies to
  PyTorch, Accelerate, Zarr, xarray, SLURM, Nsight Systems, NVTX, uv, ruff, and any other tool
  the thesis touches.
- **Explain every domain-specific concept** the first time it appears: filesystem and metadata
  operations, parallel filesystems, GPU and CPU, semaphores and file locks, prefetching,
  profiling, semantic segmentation, U-Net architecture, autoregressive priors, etc.
- Prefer thorough exposition over compactness. Do not assume the reader can fill in technical
  context from training; do not paper over an acronym or a tool name with a footnote.

### Presenting Results

These rules apply to anything that lands in the report body: section headings, figures, tables,
and the numerical claims tied to them.

- **Declarative headings only.** Every section, subsection, subsubsection, or paragraph title
  must be a noun phrase. No question titles ("Why X?", "How much do we read?", "What does this
  mean?"). Replace with the declarative form that names the topic: "Cause of X", "Per-sample
  bytes", "Implications of the result". Same rule at every heading level.
- **No internal experiment identifiers.** Lab-notebook labels like "Run 25", "BATCH 11", or job
  IDs like "41635364" are meaningless to the report's reader. Refer to a measurement
  descriptively: "the per-sample breakdown", "the first reproduction", "two independent
  measurements". Internal IDs belong in the supporting notes (`results.md`, `knowledge.md`), not
  in the report.
- **No code symbol names.** Function names (`_get_luh2_data`), class names (`MultiZarrDataset`),
  or configuration keys (`use_synchronizer`) are meaningless to a reader who cannot open the
  source. Refer to the task the code does: "the LUH2 read", "the dataset class for the
  multi-zarr layout", "the synchronizer toggle".
- **Decimal SI units for storage and bandwidth.** Megabytes (MB), gigabytes (GB), megabytes per
  second (MB/s). Do not use Mebibytes or MiB. The binary distinction does not matter at the
  precision the report works at, and the decimal form is the one a non-specialist reader knows.
- **Prefer a table to a figure when the data is tabular.** A breakdown of times by modality
  belongs in a table, not in an image. Reserve figures for what only a picture can convey: a
  timeline, a per-batch waveform, the shape of a distribution.
- **No redundant rows in tables.** If two configurations measure the same thing (for example
  `num_workers=0` and `num_workers=1` in PyTorch's DataLoader, which behave identically because
  the loader degenerates to a synchronous fetch), show one row and note the equivalence in the
  caption.
- **Tables must fit the page width.** When a row overflows the text width, restructure the
  column count, abbreviate column labels, or shrink column padding
  (`\setlength{\tabcolsep}{4pt}`). A truncated table is worse than a smaller one.
- **Figure and table captions describe what is shown.** Axes, what is being compared, and the
  takeaway. Not just "X vs Y". A reader who skims only the figures should still understand the
  experiment.
- **Be conservative with synthesised numbers.** When you sum component costs to match an
  observed total, do so only if every component has been measured directly. If any part of the
  decomposition is a guess (for example a residual term you cannot separately time), omit the
  synthesis and say instead that the measured components account for most of the observed wall
  time. Do not invent overlap to make the arithmetic add up.
- **Do not make quantitative predictions you cannot defend.** A claim like "extrapolates within
  20 per cent" is the kind of side prediction that a follow-up measurement can disprove. Unless
  it is the headline of the experiment, say "is representative" and let the concrete numbers
  below carry the weight.
- **Do not report experiments whose methodology you do not trust.** If a test is inconclusive
  because the design conflated several effects, remove it from the report rather than reporting
  a result you would have to caveat. The space is better spent on a measurement whose
  interpretation is solid.

### Pre-requisites

Before writing any report, ensure the following information is gathered. If any of these are missing
from the prompt, **ask the user** for clarification before proceeding:

1. **Target Audience Expertise:** Who is reading this? (e.g., C-level execs, technical engineers,
   general public). This dictates the complexity of the language used.
2. **Format:** What is the output format?
   - _Default:_ LaTeX (ask if the user has a specific template).
   - _Alternatives:_ Markdown, PDF, etc.
3. **Structure:** Which sections are required?
   - Is an Abstract, Introduction, or Conclusion necessary?
   - What specific content chapters/sections are expected?
4. **Constraints:** What is the maximum page count or word count?
5. **Attribution:** Who should be listed as the author(s)?

### Formatting

- **LaTeX:** Unless explicitly instructed otherwise, assume the report should be generated in LaTeX.
- **Templates:** Always inquire if a specific template (`.cls` or `.sty`) should be used to ensure
  compliance with specific standards.
