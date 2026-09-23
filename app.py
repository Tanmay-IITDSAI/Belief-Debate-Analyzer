"""
app.py -- Belief-Tracking Debate Analyzer, interactive Streamlit workbench.

Made by Tanmay Kumar Shrivastava -- first author of the accompanying paper.

The installable-package / live-demo counterpart to the static reviewer site
in demo_site/. Panels, matching the paper's Figure 1/2:

  Panel 1  Debate generation & loading   (sidebar)
  Panel 2  Rhetorical tag-strip transcript
  Panel 3  Stance trajectories + belief shift + "Why?" attribution card

Two data paths, both real, never mocked:
  - "Browse a real DebateGPT example": reads this repo's own precomputed
    output over the human-human corpus. No GPU, no HF_TOKEN, instant --
    the reliable path for an on-stage demo.
  - "Generate a new debate live": drives the actual backend="hf" pipeline
    (prompted Llama-3.1-8B-Instruct + sentence-transformer) end-to-end.
    Requires a GPU and HF_TOKEN; agent-generated debates are secondary,
    qualitative material only, per this repo's non-negotiable rules --
    they are never used for H1-H6 claims, only shown here for illustration.

Visual theme mirrors demo_site/style.css so the two surfaces read as one system.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader as dl  # noqa: E402
from src import engine  # noqa: E402

# ---------------------------------------------------------------- theme ----
# Single source of truth for the workbench look (mirrors demo_site/style.css).
RHETORIC_COLORS = {
    "causal": "#4f8dfd",
    "empirical": "#33b679",
    "emotional": "#e8734a",
    "moral": "#b48ae8",
}
LABEL_COLORS = {
    "evidence_adoption": "#33b679",
    "strategic_persuasion": "#e8734a",
    "echo": "#b48ae8",
    "anchoring": "#f2c94c",
    "no_inflection": "#5a6472",
}
LABEL_TITLES = {k: k.replace("_", " ").title() for k in LABEL_COLORS}
PRO_COLOR = "#33b679"
CON_COLOR = "#e8734a"
ACCENT = "#5b8cff"
APP_AUTHOR = "Tanmay Kumar Shrivastava"
BG = "#0f1115"
PANEL = "#171a21"
PANEL_ALT = "#1d212a"
BORDER = "#2a2f3a"
TEXT = "#e8eaed"
TEXT_DIM = "#9aa3b2"

st.set_page_config(
    page_title="Belief-Tracking Debate Analyzer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One injected stylesheet drives the whole dark look.
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="css"], .stApp {{
          background: {BG} !important;
          color: {TEXT} !important;
          font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
      }}
      [data-testid="stSidebar"] {{
          background: {PANEL} !important;
          border-right: 1px solid {BORDER} !important;
      }}
      [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
      [data-testid="stSidebar"] .stRadio label {{
          background: {PANEL_ALT};
          border: 1px solid {BORDER};
          border-radius: 8px;
          padding: 6px 10px;
          margin-bottom: 4px;
      }}
      [data-testid="stSidebar"] .stRadio label:hover {{ border-color: {ACCENT}; }}
      [data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
          border-color: {ACCENT};
          background: rgba(91,140,255,0.12);
      }}
      .block-container {{ padding-top: 1.6rem !important; max-width: 1500px; }}
      h1, h2, h3, h4 {{ color: {TEXT} !important; }}
      a {{ color: {ACCENT} !important; }}
      hr {{ border-color: {BORDER} !important; }}

      .dt-hero {{
          background: linear-gradient(180deg, #14161c, {BG});
          border: 1px solid {BORDER};
          border-radius: 14px;
          padding: 18px 22px 14px;
          margin-bottom: 14px;
      }}
      .dt-title {{
          font-size: 22px; font-weight: 700; margin: 0 0 10px;
          letter-spacing: 0.2px; color: {TEXT};
      }}
      .dt-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
      .dt-badge {{
          font-size: 11.5px; color: {TEXT_DIM};
          border: 1px solid {BORDER}; background: {PANEL_ALT};
          padding: 3px 10px; border-radius: 999px;
      }}
      .dt-tagline {{ color: {TEXT_DIM}; font-size: 13px; margin-top: 6px; }}
      .dt-survey {{ margin-top: 8px; font-size: 11.5px; color: {TEXT_DIM}; }}
      .dt-hero hr {{ margin: 12px 0 10px; }}

      .dt-kpis {{ display: flex; gap: 12px; flex-wrap: wrap; }}
      .dt-kpi {{
          flex: 1 1 150px; min-width: 150px;
          background: {PANEL}; border: 1px solid {BORDER};
          border-radius: 12px; padding: 10px 14px;
      }}
      .dt-kpi .v {{ font-size: 20px; font-weight: 700; color: {TEXT}; }}
      .dt-kpi .l {{ font-size: 11px; color: {TEXT_DIM}; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-top: 2px; }}

      .dt-sub {{
          font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px;
          color: {TEXT_DIM}; margin: 6px 0 10px;
      }}

      .dt-turn {{
          background: {PANEL};
          border: 1px solid {BORDER};
          border-left: 4px solid {BORDER};
          border-radius: 10px;
          padding: 10px 14px 8px;
          margin-bottom: 10px;
      }}
      .dt-turn:hover {{ border-color: {ACCENT}; }}
      .dt-turn-head {{
          display: flex; align-items: center; gap: 10px;
          font-size: 12px; color: {TEXT_DIM}; margin-bottom: 6px;
      }}
      .dt-speaker {{ font-weight: 700; font-size: 13px; }}
      .spk-pro {{ color: {PRO_COLOR}; }}
      .spk-con {{ color: {CON_COLOR}; }}
      .dt-chip {{
          margin-left: auto;
          font-size: 10.5px; padding: 2px 9px; border-radius: 999px;
          border: 1px solid; white-space: nowrap;
      }}
      .dt-turn-text {{
          font-size: 13.5px; line-height: 1.5; color: {TEXT};
          white-space: pre-wrap;
      }}
      .dt-tagbar {{
          display: flex; height: 6px; border-radius: 4px; overflow: hidden;
          margin-top: 9px; background: {PANEL_ALT};
      }}
      .dt-turn-foot {{
          display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px;
          font-size: 11px; color: {TEXT_DIM};
      }}
      .dt-metric {{
          flex: 1 1 0; min-width: 96px; text-align: center;
          background: {PANEL_ALT}; border: 1px solid {BORDER};
          border-radius: 6px; padding: 3px 6px; white-space: nowrap;
      }}

      .dt-why {{
          background: {PANEL}; border: 1px solid {BORDER};
          border-radius: 12px; padding: 14px 16px;
      }}
      .dt-signal {{
          display: flex; align-items: center; gap: 10px;
          border: 1px solid {BORDER}; background: {PANEL_ALT};
          border-radius: 8px; padding: 7px 10px; margin-bottom: 6px;
          font-size: 12.5px;
      }}
      .dt-signal.on {{ border-color: {ACCENT}; }}
      .dt-dot {{ width: 9px; height: 9px; border-radius: 999px; background: {BORDER}; }}
      .dt-signal.on .dt-dot {{ background: {ACCENT}; }}
      .dt-mech {{
          margin-top: 10px; padding: 10px 12px; font-size: 12.5px;
          background: rgba(91,140,255,0.12); border: 1px solid {ACCENT};
          border-radius: 8px;
      }}
      .stButton > button[kind="secondary"] {{
          background: {PANEL_ALT} !important; color: {TEXT} !important;
          border: 1px solid {BORDER} !important; border-radius: 8px !important;
      }}
      .stButton > button[kind="secondary"]:hover {{ border-color: {ACCENT} !important; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
      .stTabs [data-baseweb="tab"] {{
          background: {PANEL}; border: 1px solid {BORDER};
          border-radius: 8px 8px 0 0; color: {TEXT_DIM} !important;
      }}
      .stTabs [aria-selected="true"] {{
          border-color: {ACCENT}; color: {TEXT} !important;
      }}
      div[data-testid="stExpander"] {{
          background: {PANEL}; border: 1px solid {BORDER};
          border-radius: 10px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_precomputed() -> list[dict]:
    data_path = ROOT / "demo_site" / "data.json"
    if not data_path.is_file():
        return []
    return json.loads(data_path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def get_pipeline() -> "engine.DebateAnalysisPipeline":
    return engine.DebateAnalysisPipeline(backend="hf", checkpoint_dir=str(ROOT / "checkpoints"))


@st.cache_resource(show_spinner=False)
def get_generator() -> "engine.DebateGenerator":
    return engine.DebateGenerator(backend="hf")


def to_workbench_transcript(result: dict, source_label: str | None = None) -> dict:
    """Normalize either a precomputed record or a pipeline.analyze_transcript
    output into the single shape the render functions below expect."""
    turns = []
    for i, speaker in enumerate(result["speakers"]):
        turns.append({
            "speaker": speaker,
            "text": result["texts"][i],
            "rhetoric": result["rhetoric"][i],
            "quality": result["quality"][i],
            "stance": result["stance"][i],
            "attribution": result["attribution"][i],
        })
    return {
        "transcript_id": result["transcript_id"],
        "topic": result.get("topic") or "(untitled)",
        "turns": turns,
        "source": source_label or "real DebateGPT human-human",
    }


def prev_stance_for_speaker(turns: list[dict], index: int) -> float:
    speaker = turns[index]["speaker"]
    for j in range(index - 1, -1, -1):
        if turns[j]["speaker"] == speaker:
            return turns[j]["stance"]["S"]
    return 0.0


# ------------------------------------------------------------- rendering ----

def render_hero(chosen: dict) -> None:
    turns = chosen["turns"]
    n_turns = len(turns)
    speakers = sorted({t["speaker"] for t in turns})
    labels: dict[str, int] = {}
    for t in turns:
        labels[t["attribution"]["label"]] = labels.get(t["attribution"]["label"], 0) + 1
    n_inflections = sum(v for k, v in labels.items() if k != "no_inflection")

    # NOTE: every HTML chunk below is assembled as a Python variable first and
    # emitted on a guaranteed non-blank line. A blank line inside the markdown
    # block would split the raw-HTML context and render the rest as literal
    # text (the bug seen for live debates, where there are no belief KPIs).
    agreement = chosen.get("agreement") or {}
    kpi_items = [
        f'<div class="dt-kpi"><div class="v">{p}→{q}</div>'
        f'<div class="l">{html.escape(str(premises_side(s)))} belief (1–5, stated)</div></div>'
        for s, (p, q) in _belief_kpis(chosen).items()
    ]
    kpi_items.append(
        f'<div class="dt-kpi"><div class="v">{_net_shift(turns):+.2f}</div>'
        f'<div class="l">net system stance drift</div></div>'
    )
    kpi_items.append(
        f'<div class="dt-kpi"><div class="v">{html.escape(_top_mechanism(labels))}</div>'
        f'<div class="l">dominant mechanism</div></div>'
    )
    kpi_html = "".join(kpi_items)

    if agreement:
        bits = []
        for spk in speakers:
            a = agreement.get(spk)
            if a:
                pre, post = a["agreementPreTreatment"], a["agreementPostTreatment"]
                arrow = "→" if post >= pre else "↓"
                color = PRO_COLOR if post > pre else (CON_COLOR if post < pre else TEXT_DIM)
                bits.append(
                    f'<span style="color:{color}">{html.escape(str(premises_side(spk)))} '
                    f'{pre}{arrow}{post}</span>'
                )
        survey_line = (
            f'Stated belief change (DebateGPT survey, 1–5): {" · ".join(bits)}'
            if bits
            else "Stated belief change: n/a (no survey data attached to this debate)."
        )
    else:
        survey_line = "Stated belief change: n/a for this debate (no DebateGPT survey data)."

    badges = "".join([
        f'<span class="dt-badge">transcript {html.escape(str(chosen["transcript_id"]))}</span>',
        f'<span class="dt-badge">{n_turns} turns</span>',
        f'<span class="dt-badge">{n_inflections} attributed inflection{"s" if n_inflections != 1 else ""}</span>',
        f'<span class="dt-badge">{html.escape(str(chosen.get("source") or "real DebateGPT human-human"))}</span>',
    ])

    st.markdown(
        '<div class="dt-hero">'
        f'<h1 class="dt-title">{html.escape(str(chosen["topic"]))}</h1>'
        f'<div class="dt-badges">{badges}</div>'
        '<div class="dt-tagline">Interactive workbench for diagnosing persuasion dynamics in multi-agent debates.</div>'
        '<hr>'
        f'<div class="dt-kpis">{kpi_html}</div>'
        f'<div class="dt-survey">{survey_line}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def premises_side(speaker: str) -> str:
    return speaker.split("_")[-1].upper() if "_" in speaker else speaker


AGENT_PALETTE = ["#5b8cff", "#f2c94c", "#b48ae8", "#33b679", "#e8734a"]


def speaker_color(speaker: str, turns: list[dict] | None = None) -> str:
    """Stable per-speaker color: pro/con for corpus debates; one distinct
    palette entry per agent for live-generated debates (Agent_1, Agent_2...)."""
    if speaker.endswith("_pro"):
        return PRO_COLOR
    if speaker.endswith("_con"):
        return CON_COLOR
    if turns:
        ordered = sorted({t["speaker"] for t in turns})
        if speaker in ordered:
            return AGENT_PALETTE[ordered.index(speaker) % len(AGENT_PALETTE)]
    return TEXT


def _belief_kpis(chosen: dict) -> dict:
    agreement = chosen.get("agreement") or {}
    out = {}
    for spk in sorted(agreement.keys()):
        a = agreement.get(spk) or {}
        if {"agreementPreTreatment", "agreementPostTreatment"} <= set(a):
            out[premises_side(spk)] = (a["agreementPreTreatment"], a["agreementPostTreatment"])
    return out


def _net_shift(turns: list[dict]) -> float:
    by_last = {}
    for t in turns:
        by_last[t["speaker"]] = t["stance"]["S"]
    return sum(abs(v) for v in by_last.values())


def _top_mechanism(labels: dict) -> str:
    real = {k: v for k, v in labels.items() if k != "no_inflection" and v > 0}
    if not real:
        return "—"
    best = max(real, key=real.get)
    return LABEL_TITLES.get(best, best).title()


def render_transcript_panel(turns: list[dict]) -> None:
    st.markdown("Panel 2 — Transcript with rhetorical tag strips", help=None)
    legend = " ".join(
        f'<span style="color:{c};font-size:16px">■</span> '
        f'<span style="color:{TEXT_DIM};font-size:12px">{k}</span>'
        for k, c in RHETORIC_COLORS.items()
    )
    st.markdown(f"<div style='margin-bottom:10px'>{legend}</div>", unsafe_allow_html=True)

    for i, turn in enumerate(turns):
        label = turn["attribution"]["label"]
        color = LABEL_COLORS.get(label, "#5a6472")
        side = turn["speaker"].split("_")[-1] if "_" in turn["speaker"] else "?"
        spk_color = speaker_color(turn["speaker"], turns)
        chip_color = color if label != "no_inflection" else TEXT_DIM
        chip_border = chip_color if label != "no_inflection" else BORDER

        scores = turn["rhetoric"]["scores"]
        total = sum(scores.values()) or 1.0
        bar = "".join(
            f"<span style='display:inline-block;height:100%;width:{scores[k] / total * 100:.1f}%;"
            f"background:{RHETORIC_COLORS[k]};opacity:{0.35 + 0.65 * scores[k]:.2f}'></span>"
            for k in RHETORIC_COLORS
        )
        q = turn.get("quality")
        try:
            q_txt = f"{float(q):.2f}"
        except (TypeError, ValueError):
            q_txt = "—"
        foot_html = "".join(
            f'<span class="dt-metric">{k} {scores[k]:.2f}</span>' for k in RHETORIC_COLORS
        ) + f'<span class="dt-metric">quality {q_txt}</span>'
        selected = "outline:1px solid " + ACCENT + ";" if st.session_state.get("selected_turn") == i else ""

        st.markdown(
            f"""
            <div class="dt-turn" style="border-left-color:{color};{selected}">
              <div class="dt-turn-head">
                <span class="dt-speaker" style="color:{spk_color}">{html.escape(turn['speaker'])}</span>
                <span>turn {i + 1}</span>
                <span class="dt-chip" style="color:{chip_color};border-color:{chip_border}">
                    {LABEL_TITLES.get(label, label)}
                </span>
              </div>
              <div class="dt-turn-text">{html.escape(str(turn['text']))}</div>
              <div class="dt-tagbar">{bar}</div>
              <div class="dt-turn-foot">{foot_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect", key=f"select_{i}", help="Show this turn's attribution signals in the Why? card"):
            st.session_state.selected_turn = i


def render_stance_panel(turns: list[dict]) -> None:
    st.markdown("**Panel 3 — Stance trajectories**")
    by_speaker: dict[str, list[tuple[int, float]]] = {}
    for i, turn in enumerate(turns):
        by_speaker.setdefault(turn["speaker"], []).append((i + 1, turn["stance"]["S"]))

    fig = go.Figure()
    for speaker, points in by_speaker.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = speaker_color(speaker, turns)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=speaker,
            line={"color": color, "width": 3},
            marker={"size": 8, "line": {"width": 1, "color": BG}},
            hovertemplate="%{x}: S=%{y:.3f}<extra>" + html.escape(speaker) + "</extra>",
        ))

    fig.update_yaxes(range=[-1, 1], title="S (stance)", gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_xaxes(title="turn", dtick=1, gridcolor=BORDER)
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": TEXT_DIM, "family": "Inter, Segoe UI, sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="dt-sub">Per-turn stance movement</div>', unsafe_allow_html=True)
    deltas = []
    for i, turn in enumerate(turns):
        delta = turn["stance"]["S"] - prev_stance_for_speaker(turns, i)
        if abs(delta) >= 1e-9:
            deltas.append((i, delta))
    max_abs = max((abs(d) for _, d in deltas), default=0.0) or 1.0
    bars = []
    for i, delta in deltas:
        side = turns[i]["speaker"].split("_")[-1] if "_" in turns[i]["speaker"] else ""
        color = speaker_color(turns[i]["speaker"], turns)
        width = max(abs(delta) / max_abs * 50, 1.5)
        bars.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'<span style="width:46px;font-size:11px;color:{TEXT_DIM}">turn {i + 1}</span>'
            f'<div style="flex:1;display:flex;justify-content:{"flex-start" if delta >= 0 else "flex-end"}">'
            f'<div style="height:8px;border-radius:4px;width:{max(width, 1.5)}%;'
            f'background:{color};opacity:0.85"></div></div>'
            f'<span style="width:56px;font-size:11px;color:{TEXT_DIM}">{delta:+.3f}</span></div>'
        )
    st.markdown("".join(bars) or f'<span style="color:{TEXT_DIM};font-size:12px">No per-turn movement.</span>',
                unsafe_allow_html=True)


def render_why_card(turns: list[dict]) -> None:
    st.markdown('**Panel 3 — "Why?" attribution card**')
    index = st.session_state.get("selected_turn")
    if index is None or index >= len(turns):
        st.info("Click **Inspect** on any turn in Panel 2 to see its attribution signals.")
        return
    turn = turns[index]
    prev_s = prev_stance_for_speaker(turns, index)
    now_s = turn["stance"]["S"]
    delta = now_s - prev_s
    label = turn["attribution"]["label"]
    color = LABEL_COLORS.get(label, "#5a6472")

    signals_html = ""
    for name, on in turn["attribution"]["signals"].items():
        dot = ACCENT if on else BORDER
        weight = "700" if on else "400"
        value = "yes" if on else "no"
        signals_html += (
            f'<div class="dt-signal{" on" if on else ""}">'
            f'<span class="dt-dot" style="background:{dot}"></span>'
            f'<span>{name.replace("_", " ")}</span>'
            f'<span style="margin-left:auto;font-weight:{weight}">{value}</span></div>'
        )

    st.markdown(
        f"""
        <div class="dt-why">
          <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px">
            <span class="dt-speaker">{html.escape(turn['speaker'])}</span>
            <span style="font-size:12px;color:{TEXT_DIM}">turn {index + 1} of {len(turns)}</span>
          </div>
          <div style="font-size:13px;margin-bottom:10px">
            Stance <code style="color:{TEXT_DIM}">{prev_s:.2f} → {now_s:.2f}</code>
            (<b style="color:{PRO_COLOR if delta >= 0 else CON_COLOR}">Δ {delta:+.2f}</b>)
          </div>
          {signals_html}
          <div class="dt-mech">
            ⇒ Likely mechanism: <b style="color:{color}">{LABEL_TITLES.get(label, label).upper()}</b>
            (confidence {turn['attribution']['confidence']:.2f}, quality {turn['quality']:.2f})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Turn deep-dive"):
        st.caption("Everything below is this system's own real per-turn output for the selected turn.")
        c1, c2, c3 = st.columns(3)
        c1.metric("stance S", f"{now_s:.3f}", f"{delta:+.3f}")
        c1.metric("latent z", f"{turn['stance']['z']:.3f}")
        c2.metric("quality q", f"{float(turn['quality']):.2f}")
        c2.metric("evidence direction", str(turn.get("evidence_direction", "—")))
        c3.metric("confidence", f"{turn['attribution']['confidence']:.2f}")
        scores = turn["rhetoric"]["scores"]
        c3.metric("dominant rhetoric", f"{max(scores, key=scores.get)} ({max(scores.values()):.2f})")
        st.markdown(f'<div class="dt-sub">Rhetorical mix (causal · empirical · emotional · moral)</div>', unsafe_allow_html=True)
        probs = turn["rhetoric"].get("probs") or scores
        st.bar_chart(pd_normalize(probs))


def pd_normalize(probs: dict) -> dict:
    total = sum(v for v in probs.values() if isinstance(v, (int, float))) or 1.0
    return {k: (v / total if isinstance(v, (int, float)) else 0.0) for k, v in probs.items()}


def render_workbench(chosen: dict) -> None:
    render_hero(chosen)
    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        render_transcript_panel(chosen["turns"])
    with right:
        render_stance_panel(chosen["turns"])
        render_why_card(chosen["turns"])


def _sidebar_mode() -> str:
    st.sidebar.header("Panel 1 — Debate generation & loading")
    return st.sidebar.radio(
        "Data source",
        ["Browse a real DebateGPT example", "Generate a new debate live", "Upload transcript JSON"],
        label_visibility="collapsed",
    )


def render_credit() -> None:
    """Visible authorship credit, shown on every mode of the workbench."""
    st.markdown(
        f'<div style="margin-top:20px;padding:10px 2px 2px;border-top:1px solid {BORDER};'
        f'color:{TEXT_DIM};font-size:12.5px">'
        f'Made by <b style="color:{TEXT}">{APP_AUTHOR}</b>'
        f'<span style="color:{TEXT_DIM}"> · Belief-Tracking Debate Analyzer · '
        f'EACL 2027 System Demonstrations Track</span></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    mode = _sidebar_mode()

    if "selected_turn" not in st.session_state:
        st.session_state.selected_turn = None

    if mode == "Browse a real DebateGPT example":
        precomputed = load_precomputed()
        if not precomputed:
            st.sidebar.error(
                "demo_site/data.json not found. Run "
                "`python scripts/build_demo_site_data.py` first."
            )
            render_credit()
            return
        topics = [f"{t['topic']}  ({t['transcript_id']})" for t in precomputed]
        query = st.sidebar.text_input("Search topic", "").strip().lower()
        if query:
            topics = [t for t in topics if query in t.lower()] or topics
            st.sidebar.caption("No match — showing all." if len(topics) == len(precomputed) else "")
        choice = st.sidebar.selectbox("Choose a real human-human debate", topics)
        chosen = next(t for t in precomputed if f"{t['topic']}  ({t['transcript_id']})" == choice)
        if st.session_state.get("_last_topic") != choice:
            st.session_state.selected_turn = None
            st.session_state["_last_topic"] = choice
        render_workbench(chosen)
        st.sidebar.caption(
            "This uses only this system's own precomputed real output over the "
            "public DebateGPT human-human corpus. No model call happens in this mode."
        )

    elif mode == "Generate a new debate live":
        st.sidebar.warning("Requires a GPU and HF_TOKEN. Agent-generated debates are secondary/qualitative only.")
        if not os.environ.get("HF_TOKEN"):
            token = st.sidebar.text_input("HF_TOKEN", type="password", help="Paste your read-access token once; it stays for the session and is never stored.")
            if token:
                os.environ["HF_TOKEN"] = token
                st.sidebar.success("HF_TOKEN set for this session — ready to generate.")
                st.rerun()
        proposition = st.sidebar.text_area("Proposition", "Remote work should be the default")
        n_agents = st.sidebar.slider("Number of agents", 2, 3, 2)
        n_turns = st.sidebar.slider("Number of turns", 4, 8, 6)
        if st.sidebar.button("Generate & analyze", type="primary"):
            if not os.environ.get("HF_TOKEN"):
                st.sidebar.error("HF_TOKEN is required for live generation.")
            else:
                try:
                    with st.status("Generating & analyzing live debate…", expanded=True) as status:
                        st.write("Loading debate-generation model…")
                        generator = get_generator()
                        st.write(f"Generating a {n_agents}-agent, {n_turns}-turn debate…")

                        def _turn_progress(done: int, total: int, persona: str) -> None:
                            st.write(f"Turn {done + 1}/{total} — {persona} is thinking…")

                        transcript = generator.generate(
                            proposition, n_agents=n_agents, n_turns=n_turns,
                            on_progress=_turn_progress,
                        )
                        st.write("Loading analysis pipeline (Llama-3.1-8B + rhetorical/quality models)…")
                        pipeline = get_pipeline()
                        st.write("Running stance, rhetoric, quality and attribution analysis…")
                        result = pipeline.analyze_transcript(transcript, force=False)
                        status.update(label="Debate generated & analyzed", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Generation failed", state="error", expanded=True)
                    st.error(
                        f"Live generation failed: `{type(e).__name__}`. Common causes: an invalid "
                        "HF_TOKEN, missing access to meta-llama/Llama-3.1-8B-Instruct (accept the "
                        "license on the model page), or a GPU-memory issue. Full message: "
                        f"{e}"
                    )
                    st.session_state.pop("_generated_result", None)
                else:
                    st.session_state["_generated_result"] = result
                    st.session_state.selected_turn = None
        if "_generated_result" in st.session_state:
            render_workbench(to_workbench_transcript(
                st.session_state["_generated_result"],
                source_label="agent-generated · qualitative only",
            ))
        else:
            st.info("Configure a proposition on the left and click **Generate & analyze**.")

    else:  # Upload transcript JSON
        st.sidebar.warning("Analysis still requires a GPU and HF_TOKEN (backend='hf' is the only supported backend).")
        if not os.environ.get("HF_TOKEN"):
            token = st.sidebar.text_input("HF_TOKEN", type="password", key="hf_token_upload", help="Paste your read-access token once; it stays for the session and is never stored.")
            if token:
                os.environ["HF_TOKEN"] = token
                st.sidebar.success("HF_TOKEN set for this session — ready to analyze.")
                st.rerun()
        uploaded = st.sidebar.file_uploader("Transcript JSON (canonical schema)", type="json")
        if uploaded is not None and st.sidebar.button("Analyze", type="primary"):
            if not os.environ.get("HF_TOKEN"):
                st.sidebar.error("HF_TOKEN is required.")
            else:
                try:
                    with st.status("Analyzing uploaded transcript…", expanded=True) as status:
                        transcript = json.loads(uploaded.read().decode("utf-8"))
                        st.write("Loading analysis pipeline (Llama-3.1-8B + rhetorical/quality models)…")
                        pipeline = get_pipeline()
                        st.write("Running stance, rhetoric, quality and attribution analysis…")
                        result = pipeline.analyze_transcript(transcript, force=False)
                        status.update(label="Analysis complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Analysis failed", state="error", expanded=True)
                    st.error(
                        f"Analysis failed: `{type(e).__name__}`. Common causes: an invalid HF_TOKEN, "
                        f"missing Llama-3.1-8B-Instruct access, or malformed JSON. Full message: {e}"
                    )
                    st.session_state.pop("_uploaded_result", None)
                else:
                    st.session_state["_uploaded_result"] = result
                    st.session_state.selected_turn = None
        if "_uploaded_result" in st.session_state:
            render_workbench(to_workbench_transcript(
                st.session_state["_uploaded_result"],
                source_label="uploaded transcript · analyzed with backend='hf'",
            ))
        else:
            st.info("Upload a transcript JSON on the left, then click **Analyze**.")

    render_credit()


if __name__ == "__main__":
    main()
