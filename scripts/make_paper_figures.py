"""Generate the paper's data figures from real, on-disk evaluation outputs (no raw strings needed here).

Sources (all real artifacts in this repository):
  - reports/real_evaluation.json          -> H1-H6 verdicts
  - data/blind_study/summary_ab_comparison.csv -> H7 blind explanation-quality study
  - data/user_study/results.csv           -> H3 counterbalanced utility study
  - data/blind_study/scored_returns.csv   -> per-item H7 returns

Outputs go to Reference/figures/ as PDF (vector) for direct \includegraphics use:
  fig_attribution_distribution.pdf   full-corpus label distribution (900 turns)
  fig_annotator_collapse.pdf         per-annotator label distributions (H1)
  fig_h3_h7_summary.pdf              H3 accuracy by condition + H7 A/B metrics
  fig_case_study_stance.pdf          stance trajectory of the worked example

No number is invented: every plotted value is read from the files above, and the
script fails loudly if a source is missing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "Reference" / "figures"

# Okabe-Ito colorblind-safe palette.
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_YELLOW = "#F0E442"
C_GREY = "#999999"

LABELS = ["no_inflection", "echo", "evidence_adoption", "anchoring", "strategic_persuasion"]
LABEL_TITLES = {
    "no_inflection": "no\ninflection",
    "echo": "echo",
    "evidence_adoption": "evidence\nadoption",
    "anchoring": "anchoring",
    "strategic_persuasion": "strategic\npersuasion",
}
LABEL_COLORS = [C_GREY, C_BLUE, C_GREEN, C_YELLOW, C_VERM]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required real input missing: {path}")
    return path


def fig_attribution_distribution(eval_data: dict) -> None:
    counts = eval_data["falsification_verdicts"]  # placeholder, replaced below
    del counts
    h4 = eval_data["report"]["H4"]["attribution_label_counts_by_condition"]["full"]
    values = [h4[k] for k in LABELS]
    total = sum(values)
    assert total == 900, f"expected 900 turns, got {total}"

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bars = ax.bar([LABEL_TITLES[k] for k in LABELS], values, color=LABEL_COLORS,
                  edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8, str(v),
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("turns (of 900)", fontsize=8)
    ax.set_ylim(0, 460)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_attribution_distribution.pdf")
    plt.close(fig)


def fig_annotator_collapse(eval_data: dict) -> None:
    per_ann = eval_data["report"]["H1"]["per_annotator_label_counts"]
    annotators = ["TS", "GS", "RK"]
    cats = ["ambiguous", "anchoring", "evidence_adoption", "echo", "strategic_persuasion"]
    cat_titles = ["ambig.", "anch.", "ev. ad.", "echo", "strat."]
    cat_colors = [C_GREY, C_YELLOW, C_GREEN, C_BLUE, C_VERM]

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    left = [0.0] * len(annotators)
    for cat, title, color in zip(cats, cat_titles, cat_colors):
        vals = [per_ann[a].get(cat, 0) for a in annotators]
        ax.barh(annotators, vals, left=left, color=color, label=title,
                edgecolor="black", linewidth=0.3, height=0.55)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlabel("instances labeled (of 145)", fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, ncol=3, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_annotator_collapse.pdf")
    plt.close(fig)


def fig_h3_h7_summary() -> None:
    # --- H3: real returned judgments -------------------------------------
    h3_path = require(ROOT / "data" / "user_study" / "results.csv")
    acc = {"workbench": [], "raw_transcript": []}
    with h3_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            acc[row["condition"]].append(float(row["accuracy"]))
    wb = sum(acc["workbench"]) / len(acc["workbench"]) * 100
    raw = sum(acc["raw_transcript"]) / len(acc["raw_transcript"]) * 100

    # --- H7: real blind-study summary ------------------------------------
    h7_path = require(ROOT / "data" / "blind_study" / "summary_ab_comparison.csv")
    with h7_path.open(encoding="utf-8") as fh:
        rows = {r["condition"]: r for r in csv.DictReader(fh)}
    a, b = rows["A"], rows["B"]
    overlap_a, overlap_b = float(a["mean_mech_overlap"]), float(b["mean_mech_overlap"])
    ground_a, ground_b = float(a["mean_text_grounded"]) * 100, float(b["mean_text_grounded"]) * 100
    conf_a, conf_b = float(a["mean_confidence"]), float(b["mean_confidence"])
    useful_b = float(b["mean_usefulness"])

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))

    ax = axes[0]
    ax.bar([0, 1], [raw, wb], color=[C_GREY, C_BLUE], edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw\ntranscript", "workbench"], fontsize=7)
    ax.axhline(20, color=C_VERM, linestyle="--", linewidth=0.8)
    ax.text(0.5, 21.5, "preregistered +20 bar", fontsize=5.5, color=C_VERM, ha="center")
    ax.set_title(f"H3 accuracy: {raw:.1f}% vs {wb:.1f}%\n(+{wb - raw:.1f} pts, inconclusive)",
                 fontsize=7)
    ax.set_ylabel("accuracy (%)", fontsize=7)

    ax = axes[1]
    ax.bar([0, 1], [overlap_a, overlap_b], color=[C_GREY, C_BLUE], edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: transcript\nonly", "B: +workbench"], fontsize=7)
    ax.set_ylim(0, 0.4)
    ax.set_title("H7 mechanism-list\noverlap (Jaccard)", fontsize=7)
    ax.text(0, overlap_a + 0.01, f"{overlap_a:.2f}", ha="center", fontsize=6.5)
    ax.text(1, overlap_b + 0.01, f"{overlap_b:.2f}", ha="center", fontsize=6.5)

    ax = axes[2]
    ax.bar([0, 1], [ground_a, ground_b], color=[C_GREY, C_BLUE], edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: transcript\nonly", "B: +workbench"], fontsize=7)
    ax.set_ylim(0, 60)
    ax.set_title("H7 text-grounded\nevidence answers (%)", fontsize=7)
    ax.text(0, ground_a + 1, f"{ground_a:.1f}", ha="center", fontsize=6.5)
    ax.text(1, ground_b + 1, f"{ground_b:.1f}", ha="center", fontsize=6.5)

    ax = axes[3]
    ax.bar([0, 1, 2], [conf_a, conf_b, useful_b],
           color=[C_GREY, C_BLUE, C_GREEN], edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["conf.\n(A)", "conf.\n(B)", "usefulness\n(B only)"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title("H7 confidence and\nusefulness (1–5 Likert)", fontsize=7)
    ax.text(0, conf_a + 0.08, f"{conf_a:.2f}", ha="center", fontsize=6.5)
    ax.text(1, conf_b + 0.08, f"{conf_b:.2f}", ha="center", fontsize=6.5)
    ax.text(2, useful_b + 0.08, f"{useful_b:.2f}", ha="center", fontsize=6.5)

    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_h3_h7_summary.pdf")
    plt.close(fig)


def fig_case_study_stance() -> None:
    """Stance trajectory of the worked example (death-penalty debate, con side).

    Values 0.291 -> 0.521 -> 0.685 are the con speaker's own stance scores at
    turns 0, 2, 4 of transcript 176.0, as exported to
    reports/case_study/strategic_persuasion_contexts.json (read here, not hardcoded).
    """
    ctx_path = require(ROOT / "reports" / "case_study" / "strategic_persuasion_contexts.json")
    contexts = json.loads(ctx_path.read_text(encoding="utf-8"))
    case = None
    for c in contexts:
        if "death penalty" in c["topic"].lower():
            case = c
            break
    if case is None:
        raise ValueError("death-penalty worked-example case not found in contexts export")

    # The flagged turn carries the speaker and stance; the con speaker's own
    # turns give the trajectory (0.29 -> 0.52 -> 0.69 in the paper).
    speaker = case["turns"][case["flagged_turn"]]["speaker"]
    turns = case["turns"]
    xs, ys = [], []
    for i, t in enumerate(turns):
        if t["speaker"] == speaker:
            xs.append(i)
            ys.append(t["stance"]["S"] if isinstance(t.get("stance"), dict) else t["stance_S"])

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(xs, ys, marker="o", color=C_BLUE, linewidth=1.2, markersize=4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xlabel("turn", fontsize=8)
    ax.set_ylabel("stance $S_t$", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_case_study_stance.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = require(ROOT / "reports" / "real_evaluation.json")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

    fig_attribution_distribution(eval_data)
    fig_annotator_collapse(eval_data)
    fig_h3_h7_summary()
    fig_case_study_stance()
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
