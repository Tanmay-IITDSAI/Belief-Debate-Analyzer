"""Metrics implemented from the attached reference papers.

All functions consume observed records.  This module deliberately contains no
data generators, random sampling, model calls, or fallback values.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def normalized_change_agreement(pre: float, post: float, minimum: float = 1.0, maximum: float = 5.0) -> float:
    """PMIYC normalized change in agreement (NCA), on the paper's 1--5 scale."""
    if not minimum <= pre <= maximum or not minimum <= post <= maximum:
        raise ValueError(f"agreement scores must be in [{minimum}, {maximum}]")
    if post == pre:
        return 0.0
    denominator = maximum - pre if post > pre else pre - minimum
    return float((post - pre) / denominator) if denominator else 0.0


def nca_summary(
    records: Iterable[dict],
    pre_key: str = "agreementPreTreatment",
    post_key: str = "agreementPostTreatment",
) -> dict:
    values = [normalized_change_agreement(float(row[pre_key]), float(row[post_key])) for row in records]
    if not values:
        raise ValueError("at least one observed conversation is required")
    return {"mean_nca": float(np.mean(values)), "std_nca": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values), "values": values}


def persuasion_effectiveness(records: Iterable[dict], persuader_key: str = "persuader_model") -> dict:
    """Mean NCA grouped by persuader, matching PMIYC effectiveness reporting."""
    groups: dict[str, list[float]] = {}
    for row in records:
        groups.setdefault(str(row[persuader_key]), []).append(
            normalized_change_agreement(float(row["agreementPreTreatment"]), float(row["agreementPostTreatment"]))
        )
    return {model: {"mean_nca": float(np.mean(values)), "n": len(values)} for model, values in sorted(groups.items())}


def susceptibility_by_persuadee(records: Iterable[dict], persuadee_key: str = "persuadee_model") -> dict:
    """Mean NCA grouped by persuadee, matching PMIYC susceptibility reporting."""
    groups: dict[str, list[float]] = {}
    for row in records:
        groups.setdefault(str(row[persuadee_key]), []).append(
            normalized_change_agreement(float(row["agreementPreTreatment"]), float(row["agreementPostTreatment"]))
        )
    return {model: {"mean_nca": float(np.mean(values)), "n": len(values)} for model, values in sorted(groups.items())}


def cw_por(
    records: Iterable[dict],
    weight_key: str = "normalized_jsd_weight",
    side_only_key: str = "side_only_without_proposition_change",
) -> dict:
    """DebateGPT CW-POR adaptation using normalized rhetorical JSD weights."""
    rows = list(records)
    if not rows:
        raise ValueError("at least one observed item is required")
    weights = np.asarray([float(row[weight_key]) for row in rows], dtype=float)
    side_only = np.asarray([float(row[side_only_key]) for row in rows], dtype=float)
    if np.any((weights < 0) | (weights > 1)) or np.any(~np.isin(side_only, [0.0, 1.0])):
        raise ValueError("normalized JSD weights must be in [0, 1] and side-only flags must be binary")
    total = float(weights.sum())
    if total == 0:
        raise ValueError("normalized JSD weight sum must be positive")
    return {"cw_por": float(np.dot(weights, side_only) / total), "side_only_rate": float(side_only.mean()), "n": len(rows)}


def stance_convergence_decomposition(records: Iterable[dict]) -> dict:
    """Decompose answer flips into spontaneous, stance-only, and reasoning effects.

    Each row must contain binary answers for the four matched conditions from
    the reference paper: ``round0``, ``self_reflection``, ``stance_only``, and
    ``full_reasoning``.  Flip rates are measured against ``round0``.
    """
    rows = list(records)
    if not rows:
        raise ValueError("at least one matched question is required")
    conditions = ("round0", "self_reflection", "stance_only", "full_reasoning")
    for row in rows:
        for condition in conditions:
            if row[condition] not in (0, 1, False, True):
                raise ValueError(f"{condition} must be binary")
    baseline = np.asarray([int(row["round0"]) for row in rows])
    rates = {condition: float(np.mean([int(row[condition]) != base for row, base in zip(rows, baseline)]))
             for condition in conditions[1:]}
    return {"n": len(rows), "flip_rates": rates,
            "increment_self_reflection": rates["self_reflection"],
            "increment_stance": rates["stance_only"] - rates["self_reflection"],
            "increment_reasoning": rates["full_reasoning"] - rates["stance_only"]}


def normalized_change_stance(pre: float, post: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    """Normalized change for the system's bounded stance scale."""
    if not minimum <= pre <= maximum or not minimum <= post <= maximum:
        raise ValueError(f"stance scores must be in [{minimum}, {maximum}]")
    if post == pre:
        return 0.0
    denominator = maximum - pre if post > pre else pre - minimum
    return float((post - pre) / denominator) if denominator else 0.0


def pearson_nca_correlation(system_records: Sequence[dict], human_records: Sequence[dict]) -> dict:
    """Pearson correlation of paired system and human NCA observations."""
    if len(system_records) != len(human_records) or len(system_records) < 2:
        raise ValueError("paired system and human records require equal length >= 2")
    system = [
        normalized_change_stance(float(row["system_pre"]), float(row["system_post"]))
        for row in system_records
    ]
    human = [
        normalized_change_agreement(
            float(row["agreementPreTreatment"]), float(row["agreementPostTreatment"])
        )
        for row in human_records
    ]
    r, p = stats.pearsonr(system, human)
    return {"r": float(r), "p_value": float(p), "n": len(system)}