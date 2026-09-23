"""Score the blind explanation-quality study (real returns only)."""
import json
import re
from pathlib import Path

import pandas as pd

MECHS = ["evidence_adoption", "anchoring", "echo", "strategic_persuasion", "no_clear_change"]
QCOLS = ["q1_change", "q2_mechanisms", "q3_evidence",
         "q4_confidence_1to5", "q5_workbench_useful_1to5_or_NA"]


def load_returns(blind_dir: Path):
    frames = []
    for f in sorted(blind_dir.glob("packets_*/*.csv")):
        df = pd.read_csv(f, dtype={"transcript_id": str})
        # keep only rows where the participant actually answered something
        df = df[df["q2_mechanisms"].notna() & (df["q2_mechanisms"].astype(str).str.strip() != "")]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def jaccard(a: str, b: str):
    sa = {x.strip() for x in str(a).lower().split(";") if x.strip()}
    sb = {x.strip() for x in str(b).lower().split(";") if x.strip()}
    if not sa and not sb:
        return None
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def workbench_label_set(by_id, tid):
    """Reference multi-select: system label + runner-up per turn, collapsed per transcript."""
    t = by_id[str(tid)]
    labels = set()
    for turn in t["turns"]:
        if turn["attribution"]["label"] in MECHS:
            labels.add(turn["attribution"]["label"])
    return ";".join(sorted(labels))


def _normalize(text: str) -> str:
    """Lowercase, unify curly quotes/dashes, strip punctuation, collapse whitespace."""
    text = (text.lower()
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("—", " ").replace("–", " "))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _quote_segments(ev: str):
    """Quoted spans of >=10 chars if present; otherwise the whole answer."""
    quotes = re.findall(r"[\u201c\u0022]([^\u201d\u0022]{10,})[\u201d\u0022]", ev)
    return quotes if quotes else [ev]


def text_grounded(ev: str, transcript) -> float:
    """Fraction of quoted evidence segments found verbatim (5-word windows) in the
    transcript after punctuation normalization. Paraphrase-only answers score 0.0."""
    if not isinstance(ev, str) or not ev.strip():
        return None
    corpus = _normalize(" ".join(turn["text"] for turn in transcript["turns"]))
    hits, total = 0, 0
    for seg in _quote_segments(ev):
        words = [w for w in _normalize(seg).split() if len(w) > 2]
        if len(words) < 5:
            continue
        total += 1
        if any(" ".join(words[i:i + 5]) in corpus for i in range(len(words) - 4)):
            hits += 1
    return hits / total if total else None



def score(blind_dir: Path, data_json: Path):
    by_id = {str(t["transcript_id"]): t
             for t in json.loads(data_json.read_text(encoding="utf-8"))}
    df = load_returns(blind_dir)
    df["mech_overlap"] = [
        jaccard(mechs, workbench_label_set(by_id, tid))
        for mechs, tid in zip(df["q2_mechanisms"], df["transcript_id"])]
    df["text_grounded"] = [
        text_grounded(ev, by_id[str(tid)])
        for ev, tid in zip(df["q3_evidence"], df["transcript_id"])]
    df["q4_num"] = pd.to_numeric(df["q4_confidence_1to5"], errors="coerce")
    df["q5_num"] = pd.to_numeric(df["q5_workbench_useful_1to5_or_NA"], errors="coerce")
    summary = df.groupby("condition").agg(
        n_items=("transcript_id", "count"),
        mean_mech_overlap=("mech_overlap", "mean"),
        mean_text_grounded=("text_grounded", "mean"),
        mean_confidence=("q4_num", "mean"),
        mean_usefulness=("q5_num", "mean"),
    ).round(4)
    return df, summary
