# Experimentation Log

_Log created 2026-09-13T20:33:43.228097_

---

## Run configuration
_2026-09-13T20:33:44_

Hyperparameters / run configuration for this step:

| Metric | Value |
|---|---|
| backend | mock |
| seed | 42 |

## Panel 1: Debate generation
_2026-09-13T20:33:44_

Generated a 6-turn debate on: 'Universal basic income should be adopted nationally'.

| Metric | Value |
|---|---|
| backend | mock |
| n_turns | 6 |

## Panel 2: Rhetorical annotation
_2026-09-13T20:33:44_

Annotated every utterance in the sample transcript.

| Metric | Value |
|---|---|
| n_utterances | 6 |

## Panel 3: Stance dynamics
_2026-09-13T20:33:44_

Computed per-agent stance trajectories via the Eq.(1)-(2) log-odds update.

| Metric | Value |
|---|---|
| alpha | 0.8 |
| beta | 0.15 |
| n_agents | 2 |

## Attribution engine
_2026-09-13T20:33:45_

Ran the full analysis pipeline (annotation + stance + attribution) on the sample transcript.

## Batch analysis
_2026-09-13T20:33:45_

Analyzed 32 transcripts end-to-end.

| Metric | Value |
|---|---|
| n_transcripts | 32 |
| backend | mock |

## Save consolidated results
_2026-09-13T20:33:45_

Saved 32 analyzed transcripts to `checkpoints/analysis_results.pkl`.

| Metric | Value |
|---|---|
| evidence_adoption | 20 |
| echo | 131 |

