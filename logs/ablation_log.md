# Evaluation & Ablation Log

_Log created 2026-09-13T20:33:51.988878_

---

## H1: Inter-annotator agreement
_2026-09-13T20:33:52_

SYNTHETIC data used.

| Metric | Value |
|---|---|
| fleiss_kappa | 0.4494 |
| n_items | 64 |

## H2: Attribution accuracy
_2026-09-13T20:33:52_

Computed overall accuracy, per-class F1, confusion matrix, and the aligned/public-only split.

| Metric | Value |
|---|---|
| overall_accuracy | 0.7812 |
| f1_evidence_adoption | 0.8333 |
| f1_anchoring | 0.8125 |
| f1_echo | 0.8108 |
| f1_strategic_persuasion | 0.6087 |
| acc_aligned | 0.7895 |
| acc_public_only | 0.7692 |

## H3: Utility comparison
_2026-09-13T20:33:52_

SYNTHETIC data used.

| Metric | Value |
|---|---|
| mean_workbench | 0.7506 |
| mean_control | 0.5812 |
| accuracy_gain_pct_points | 16.9351 |
| t_stat | 4.1934 |
| p_value | 0.0002 |
| cohens_d | 1.3092 |
| n_workbench | 19 |
| n_control | 21 |

## H4: Ablation study
_2026-09-13T20:33:53_

Measured internal-consistency drop vs. the full system when each module is removed (a proxy pending full human coverage of every transcript for a true accuracy-vs-human-label ablation).

| Metric | Value |
|---|---|
| no_rhetorical_annotation | 51.06 |
| no_stance_layer | 0.0 |
| no_echo_detection | 0.0 |

## H5: NC-based stance-shift correlation
_2026-09-13T20:33:53_

SYNTHETIC pre/post pairs used to validate the NC + Pearson-r computation (real pairs come from DEBATE private beliefs + the system's St trajectory on the same instances).

| Metric | Value |
|---|---|
| r | 0.8184 |
| p_value | 0.0 |
| n | 60 |

## H6: Confidence-weighted strategic-persuasion rate
_2026-09-13T20:33:53_

Computed over 1 system-flagged strategic-persuasion turns.

| Metric | Value |
|---|---|
| cw_por_rate | 1.0 |
| n_flagged_turns | 1 |

## Falsifiable-threshold verdicts
_2026-09-13T20:33:53_

Table 1 reproduction (see caveat above re: synthetic vs. real inputs).

| hypothesis                    |   observed |   expected |   falsify_below | verdict      |
|:------------------------------|-----------:|-----------:|----------------:|:-------------|
| H1_fleiss_kappa               |   0.449397 |       0.6  |            0.4  | INCONCLUSIVE |
| H2_overall_accuracy           |   0.78125  |       0.7  |            0.55 | SUPPORTED    |
| H2_f1_evidence_adoption       |   0.833333 |       0.75 |            0.6  | SUPPORTED    |
| H2_f1_anchoring               |   0.8125   |       0.65 |            0.5  | SUPPORTED    |
| H2_f1_echo                    |   0.810811 |       0.6  |            0.45 | SUPPORTED    |
| H2_f1_strategic_persuasion    |   0.608696 |       0.55 |            0.4  | SUPPORTED    |
| H3_utility_accuracy_gain_pct  |  16.9351   |      20    |           10    | INCONCLUSIVE |
| H4_ablation_rhetoric_drop_pts |  51.0638   |      10    |            5    | SUPPORTED    |
| H5_nc_correlation_r           |   0.818376 |       0.5  |            0.2  | SUPPORTED    |
| H6_cw_por_rate                |   1        |       0.65 |            0.45 | SUPPORTED    |

