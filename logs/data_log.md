# Data Pipeline Log

_Log created 2026-09-13T20:33:35.078393_

---

## Run configuration
_2026-09-13T20:33:36_

Hyperparameters / run configuration for this step:

| Metric | Value |
|---|---|
| seed | 42 |
| checkpoint_dir | ../checkpoints |
| DEBATE_DATA_PATH | data/raw/debate_benchmark |
| ARGS_ME_DATA_PATH | data/raw/args_me_corpus |

## Step 1: Load DEBATE benchmark
_2026-09-13T20:33:36_

**SYNTHETIC fallback used** — real DEBATE_DATA_PATH not found. Results downstream are for pipeline validation only.

| Metric | Value |
|---|---|
| rows | 192 |
| groups | 32 |
| topics | 8 |
| participants | 128 |
| synthetic_rows | 192 |

## Step 2: Transcripts + stance-change instances
_2026-09-13T20:33:36_

Built 32 transcripts and 64 stance-change instances.

| Metric | Value |
|---|---|
| transcripts | 32 |
| instances | 64 |
| aligned | 37 |
| public_only | 27 |

## Step 3: args.me corpus
_2026-09-13T20:33:36_

Loaded 400 arguments (synthetic).

| Metric | Value |
|---|---|
| rows | 400 |

## Step 4: Annotation sample
_2026-09-13T20:33:36_

Drew a stratified sample of 64 stance-change instances for the 3-annotator study, saved to `../data/processed/annotation_sample.csv`.

| Metric | Value |
|---|---|
| n_sample | 64 |
| n_aligned | 35 |
| n_public_only | 29 |

## Step 6: Crash-resilience check
_2026-09-13T20:33:37_

Re-ran the annotation-sample step; loaded from checkpoint in 2.3 ms with an identical result. Safe to resume after a kernel death.

