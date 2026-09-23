"""
src/metrics.py
Implements the full preregistered evaluation plan from the paper (§4),
including the exact falsification thresholds from Table 1. Every function
here is a real, working statistical implementation -- no placeholders --
so the notebook can be run immediately against synthetic data to validate
correctness, and later re-pointed at the real, collected human-annotation
and user-study data with zero code changes.

Covers:
  H1  Fleiss' kappa (inter-annotator agreement)               fleiss_kappa
    H2  Attribution accuracy / per-class F1 / confusion matrix, joint-vs-divergent movement
  H3  Utility comparison (workbench vs raw-transcript)         utility_comparison
    H4  Four ablations and per-signal accuracy                    ablation_deltas
  H5  NC-based stance-shift correlation (Bozdag et al., 2025)  normalized_change, nc_correlation
  H6  Confidence-weighted strategic-persuasion rate            confidence_weighted_persuasion_rate
      (CW-POR-adapted, Agarwal & Khanna, 2025)
  --  Falsifiable-threshold verdicts (Table 1)                 FALSIFICATION_TABLE, check_falsification
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

from .utils import logger

ATTRIBUTION_LABELS = ["evidence_adoption", "anchoring", "echo", "strategic_persuasion"]

# Table 1 of the paper, verbatim expected/falsification thresholds.
FALSIFICATION_TABLE = {
    "H1_fleiss_kappa": {"expected": 0.60, "falsify_below": 0.40, "direction": "gte"},
    "H2_overall_accuracy": {"expected": 0.70, "falsify_below": 0.55, "direction": "gte"},
    "H2_f1_evidence_adoption": {"expected": 0.75, "falsify_below": 0.60, "direction": "gte"},
    "H2_f1_anchoring": {"expected": 0.65, "falsify_below": 0.50, "direction": "gte"},
    "H2_f1_echo": {"expected": 0.60, "falsify_below": 0.45, "direction": "gte"},
    "H2_f1_strategic_persuasion": {"expected": 0.55, "falsify_below": 0.40, "direction": "gte"},
    "H3_utility_accuracy_gain_pct": {"expected": 20.0, "falsify_below": 10.0, "direction": "gte"},
    "H4_ablation_rhetoric_drop_pts": {"expected": 10.0, "falsify_below": 5.0, "direction": "gte"},
    "H5_nc_correlation_r": {"expected": 0.50, "falsify_below": 0.20, "direction": "gte"},
    "H6_cw_por_rate": {"expected": 0.65, "falsify_below": 0.45, "direction": "gte"},
}


# ---------------------------------------------------------------------------
# H1: Fleiss' kappa
# ---------------------------------------------------------------------------
def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for inter-annotator agreement across >=3 raters on a
    fixed set of categorical labels.

    ratings: an (N_items, N_categories) integer matrix where
             ratings[i, j] = number of annotators who assigned item i to
             category j. Row sums must all equal n_annotators.
    """
    ratings = np.asarray(ratings, dtype=float)
    n_items, n_categories = ratings.shape
    n_annotators = ratings.sum(axis=1)
    if not np.allclose(n_annotators, n_annotators[0]):
        raise ValueError("Every item must have the same number of annotator votes.")
    n = n_annotators[0]

    p_j = ratings.sum(axis=0) / (n_items * n)  # category marginal proportions
    P_i = (np.sum(ratings * ratings, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e_bar = np.sum(p_j ** 2)
    if P_e_bar == 1.0:
        return 1.0  # degenerate: all raters always pick the same single category
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return float(kappa)


def ratings_from_long_df(df: pd.DataFrame, item_col: str, annotator_col: str, label_col: str,
                          labels: Optional[list] = None) -> np.ndarray:
    """Convert a long-format (item, annotator, label) dataframe into the
    (N_items, N_categories) count matrix fleiss_kappa expects."""
    labels = labels or sorted(df[label_col].unique())
    # Build the crosstab from positional numpy arrays: pd.crosstab keys on the
    # frame's index, so a concatenated frame with duplicate indices (e.g. one
    # block per annotator) would otherwise raise "cannot reindex on an axis
    # with duplicate labels".
    pivot = pd.crosstab(
        pd.Series(df[item_col].to_numpy(), name=item_col),
        pd.Series(df[label_col].to_numpy(), name=label_col))
    pivot = pivot.reindex(columns=labels, fill_value=0)
    return pivot.values


# ---------------------------------------------------------------------------
# H2: Attribution accuracy / F1 / confusion matrix / alignment split
# ---------------------------------------------------------------------------
def attribution_accuracy(y_true: list, y_pred: list, labels: Optional[list] = None) -> dict:
    labels = labels or ATTRIBUTION_LABELS
    y_true, y_pred = list(y_true), list(y_pred)
    overall_acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(labels, [float(x) for x in per_class_f1])),
        "confusion_matrix": cm,
        "labels": labels,
    }


def accuracy_by_movement(df: pd.DataFrame, true_col: str, pred_col: str, movement_col: str = "movement_class") -> dict:
    """H2 accuracy split by joint versus divergent proposition/side movement."""
    out = {}
    for group, g in df.groupby(movement_col):
        acc = float((g[true_col] == g[pred_col]).mean())
        out[group] = {"accuracy": acc, "n": len(g)}
    return out


accuracy_by_alignment = accuracy_by_movement


# ---------------------------------------------------------------------------
# H3: Utility comparison
# ---------------------------------------------------------------------------
def utility_comparison(
    workbench_scores: np.ndarray,
    control_scores: np.ndarray,
    workbench_time: Optional[np.ndarray] = None,
    control_time: Optional[np.ndarray] = None,
) -> dict:
    """Independent-samples comparison of diagnostic-question accuracy
    between the workbench condition and the raw-transcript control (§4.2).
    Reports mean accuracy per condition, absolute and percentage-point
    gain, Welch's t-test, and Cohen's d effect size."""
    workbench_scores = np.asarray(workbench_scores, dtype=float)
    control_scores = np.asarray(control_scores, dtype=float)
    mean_wb, mean_ctrl = workbench_scores.mean(), control_scores.mean()
    t_stat, p_val = stats.ttest_ind(workbench_scores, control_scores, equal_var=False)
    pooled_sd = np.sqrt((workbench_scores.var(ddof=1) + control_scores.var(ddof=1)) / 2)
    cohens_d = float((mean_wb - mean_ctrl) / pooled_sd) if pooled_sd > 0 else float("nan")
    result = {
        "mean_workbench": float(mean_wb),
        "mean_control": float(mean_ctrl),
        "accuracy_gain_pct_points": float((mean_wb - mean_ctrl) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d,
        "n_workbench": len(workbench_scores),
        "n_control": len(control_scores),
    }
    if workbench_time is not None and control_time is not None:
        result.update({
            "mean_workbench_time_to_answer": float(np.asarray(workbench_time, dtype=float).mean()),
            "mean_control_time_to_answer": float(np.asarray(control_time, dtype=float).mean()),
            "time_to_answer_difference": float(np.asarray(workbench_time, dtype=float).mean() - np.asarray(control_time, dtype=float).mean()),
        })
    return result


# ---------------------------------------------------------------------------
# H4: Ablation deltas
# ---------------------------------------------------------------------------
def ablation_deltas(full_accuracy: float, ablation_accuracies: dict) -> dict:
    """H4: percentage-point drop in overall attribution accuracy when each
    module is removed, relative to the full system. Larger drop => that
    module is more load-bearing."""
    return {
        name: {
            "accuracy": acc,
            "drop_pts": (full_accuracy - acc) * 100,
        }
        for name, acc in ablation_accuracies.items()
    }


def per_signal_accuracy(df: pd.DataFrame, true_col: str, signal_prediction_cols: dict) -> dict:
    """Accuracy for the no-priority-order ablation, one signal at a time."""
    return {
        signal: float((df[true_col] == df[pred_col]).mean())
        for signal, pred_col in signal_prediction_cols.items()
    }


# ---------------------------------------------------------------------------
# H5: NC-based stance-shift correlation (Bozdag et al., 2025)
# ---------------------------------------------------------------------------
def normalized_change(pre: float, post: float, scale_min: float = -1.0, scale_max: float = 1.0) -> float:
    """Normalized Change (NC), a signed generalization of normalized gain
    to a bounded scale: how much of the *available room to move* was used.
        post > pre: NC = (post - pre) / (scale_max - pre)
        post < pre: NC = (post - pre) / (pre - scale_min)
        post == pre: NC = 0
    Bounded in [-1, 1] by construction."""
    if post > pre:
        denom = scale_max - pre
    elif post < pre:
        denom = pre - scale_min
    else:
        return 0.0
    if denom == 0:
        return 0.0
    return float((post - pre) / denom)


def normalized_change_agreement(pre: float, post: float, minimum: float = 1.0, maximum: float = 5.0) -> float:
    """Bozdag-adapted NC for DebateGPT's 1--5 proposition agreement scores."""
    if not minimum <= pre <= maximum or not minimum <= post <= maximum:
        raise ValueError("DebateGPT agreement scores must be in [1, 5]")
    if post == pre:
        return 0.0
    denominator = maximum - pre if post > pre else pre - minimum
    return float((post - pre) / denominator) if denominator else 0.0


def nc_correlation(system_pre_post: list, debategpt_pre_post: list) -> dict:
    """H5 correlation between system S_t NC and DebateGPT agreement NC."""
    nc_system = [normalized_change(pre, post) for pre, post in system_pre_post]
    nc_debategpt = [normalized_change_agreement(pre, post) for pre, post in debategpt_pre_post]
    r, p = stats.pearsonr(nc_system, nc_debategpt)
    return {"r": float(r), "p_value": float(p), "n": len(nc_system),
            "nc_system": nc_system, "nc_debategpt": nc_debategpt}


# ---------------------------------------------------------------------------
# H6: Confidence-weighted strategic-persuasion rate (CW-POR-adapted)
# ---------------------------------------------------------------------------
def confidence_weighted_persuasion_rate(weights: list, is_side_only_without_proposition_change: list) -> float:
    """H6: adapts the Confidence-Weighted Persuasion Override Rate
    (Agarwal & Khanna, 2025). Here, each turn flagged as strategic
    persuasion is weighted by normalized rhetorical JSD magnitude; we report
    the weighted rate at which flagged turns correspond to side-agreement-only
    change without proposition-agreement movement in DebateGPT.

        rate = sum(js_weight_i * side_only_i) / sum(js_weight_i)
    """
    weights = np.asarray(weights, dtype=float)
    is_side_only_without_proposition_change = np.asarray(is_side_only_without_proposition_change, dtype=float)
    denom = weights.sum()
    if denom == 0:
        return float("nan")
    return float(np.sum(weights * is_side_only_without_proposition_change) / denom)


# ---------------------------------------------------------------------------
# Falsification verdicts (Table 1)
# ---------------------------------------------------------------------------
def check_falsification(observed: dict, table: Optional[dict] = None) -> pd.DataFrame:
    """Compares observed metric values against the preregistered
    expected/falsification thresholds (Table 1) and returns a verdict
    per hypothesis: SUPPORTED (>= expected), INCONCLUSIVE (between
    falsify_below and expected), or FALSIFIED (< falsify_below)."""
    table = table or FALSIFICATION_TABLE
    rows = []
    for key, spec in table.items():
        value = observed.get(key)
        if value is None:
            verdict = "NOT_COMPUTED"
        elif value >= spec["expected"]:
            verdict = "SUPPORTED"
        elif value < spec["falsify_below"]:
            verdict = "FALSIFIED"
        else:
            verdict = "INCONCLUSIVE"
        rows.append(
            {
                "hypothesis": key,
                "observed": value,
                "expected": spec["expected"],
                "falsify_below": spec["falsify_below"],
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting (300 DPI, colorblind-safe -- see src/utils.set_plot_theme)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str = "Attribution confusion matrix"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "count"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Human majority label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ablation_bars(full_accuracy: float, ablation_dict: dict, title: str = "Ablation study"):
    import matplotlib.pyplot as plt

    names = ["Full system"] + list(ablation_dict.keys())
    values = [full_accuracy] + [v["accuracy"] for v in ablation_dict.values()]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(names, values)
    bars[0].set_alpha(1.0)
    for b in bars[1:]:
        b.set_alpha(0.75)
    ax.set_ylabel("Overall attribution accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_nc_scatter(nc_system: list, nc_human: list, r: float, title: str = "NC correlation (H5)"):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.scatter(nc_human, nc_system, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(nc_human) > 1:
        z = np.polyfit(nc_human, nc_system, 1)
        xs = np.linspace(min(nc_human), max(nc_human), 50)
        ax.plot(xs, np.polyval(z, xs), linestyle="--", linewidth=1.5)
    ax.set_xlabel("NC — proposition agreement (DebateGPT)")
    ax.set_ylabel("NC — system stance $S_t$")
    ax.set_title(f"{title}  (r={r:.2f})")
    fig.tight_layout()
    return fig


def plot_kappa_and_thresholds(verdict_df: pd.DataFrame, title: str = "Preregistered falsification verdicts"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"SUPPORTED": "#009E73", "INCONCLUSIVE": "#E69F00", "FALSIFIED": "#D55E00", "NOT_COMPUTED": "#999999"}
    y_pos = np.arange(len(verdict_df))
    vals = verdict_df["observed"].fillna(0).astype(float).values
    bar_colors = [colors.get(v, "#999999") for v in verdict_df["verdict"]]
    ax.barh(y_pos, vals, color=bar_colors)
    for i, (_, row) in enumerate(verdict_df.iterrows()):
        ax.plot([row["expected"], row["expected"]], [i - 0.4, i + 0.4],
                color="black", linewidth=1.5, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(verdict_df["hypothesis"])
    ax.set_xlabel("Observed value")
    ax.set_title(title)
    fig.tight_layout()
    return fig


