"""Fail-closed real-data loaders for DebateGPT and args.me."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import CheckpointManager, logger, stable_hash

RHETORIC_LABELS = ["causal", "empirical", "emotional", "moral"]
DEBATEGPT_FIELDS = {
    "agreementPreTreatment", "agreementPostTreatment",
    "sideAgreementPreTreatment", "sideAgreementPostTreatment",
}
DEBATEGPT_BASE_FIELDS = {"debateID", "treatmentType", "topic", "side", "argument", "rebuttal", "conclusion"}
HUMAN_HUMAN_VALUES = {"human-human", "human_human", "human human"}


def _read_tabular_files(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    candidates = [
        f for f in list(path.glob("*.json")) + list(path.glob("*.jsonl")) + list(path.glob("*.csv"))
        if not f.name.startswith(".") and f.name != ".download_manifest.json"
    ]
    if not path.exists() or not candidates:
        raise FileNotFoundError(f"No dataset files found at {data_path}")
    frames = []
    for file_path in candidates:
        if file_path.suffix == ".csv":
            frames.append(pd.read_csv(file_path))
        elif file_path.suffix == ".jsonl":
            frames.append(pd.read_json(file_path, lines=True))
        else:
            frames.append(pd.read_json(file_path))
    return pd.concat(frames, ignore_index=True)


def load_debategpt(
    data_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    force: bool = False,
    human_human_only: bool = True,
) -> pd.DataFrame:
    """Load real DebateGPT rows and validate the current PDF schema.

    Primary evaluation defaults to the human-human condition. Proposition
    agreement is the belief signal; side agreement is a weaker behavioral
    proxy and is never treated as a public/private stance channel.
    """
    data_path = data_path or os.environ.get("DEBATEGPT_DATA_PATH", "data/raw/debategpt")
    ckpt = CheckpointManager(checkpoint_dir)
    cache_key = f"debategpt_v2__{stable_hash([str(data_path), human_human_only])}"

    def _load():
        df = _read_tabular_files(data_path)
        missing = (DEBATEGPT_BASE_FIELDS | DEBATEGPT_FIELDS) - set(df.columns)
        if missing:
            raise ValueError(f"DebateGPT export is missing required columns: {sorted(missing)}")
        if human_human_only:
            condition_column = next(
                (c for c in ("condition", "treatment", "treatment_condition", "treatmentType") if c in df.columns),
                None,
            )
            if condition_column is None:
                raise ValueError("DebateGPT export must include condition/treatment for human-human filtering")
            normalized = df[condition_column].astype(str).str.strip().str.lower()
            mask = normalized == "human-human"
            if not mask.any():
                raise ValueError(f"No human-human DebateGPT rows found in {condition_column!r}")
            df = df.loc[mask].copy()
            df["condition"] = "human-human"
        for field in DEBATEGPT_FIELDS:
            values = pd.to_numeric(df[field], errors="coerce")
            if values.isna().any() or not values.between(1, 5).all():
                raise ValueError(f"DebateGPT field {field} must contain only numeric values in [1, 5]")
            df[field] = values.astype(int)
        df["synthetic"] = False
        df["group_id"] = df["debateID"].astype(str)
        df["participant_id"] = df["debateID"].astype(str) + "_" + df["side"].astype(str).str.lower()
        df["turn"] = 0
        logger.info("Loaded real DebateGPT rows from %s (%d rows)", data_path, len(df))
        return df.reset_index(drop=True)

    return ckpt.load_or_compute(cache_key, _load, fmt="pkl", force=force)


DEBATEGPT_ROUNDS = ("argument", "rebuttal", "conclusion")


def build_transcripts(df: pd.DataFrame) -> list:
    """Convert DebateGPT rows into the canonical transcript schema.

    Each DebateGPT row is one participant's full record for a debate: their
    own `argument`/`rebuttal`/`conclusion` texts (the *Opponent columns are
    the other side's identical texts, mirrored for convenience, and are not
    a separate observation). The real exchange is therefore three rounds,
    each with both sides speaking once -- six turns per debate, not one row
    == one turn. Side order within each round is fixed by sorting on `side`
    so the reconstruction is deterministic and reproducible.
    """
    required = DEBATEGPT_BASE_FIELDS | DEBATEGPT_FIELDS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot build transcripts; missing DebateGPT columns: {sorted(missing)}")
    transcripts = []
    for group_id, group in df.groupby("group_id"):
        group = group.sort_values("side")
        turns = []
        for round_field in DEBATEGPT_ROUNDS:
            for row in group.itertuples():
                turns.append({
                    "turn": len(turns),
                    "speaker": row.participant_id,
                    "text": getattr(row, round_field),
                    "agreement_pre": int(row.agreementPreTreatment),
                    "agreement_post": int(row.agreementPostTreatment),
                    "side_agreement_pre": int(row.sideAgreementPreTreatment),
                    "side_agreement_post": int(row.sideAgreementPostTreatment),
                })
        transcripts.append({
            "transcript_id": str(group_id),
            "topic": group["topic"].iloc[0],
            "source": "debategpt_human_human",
            "condition": "human-human",
            "turns": turns,
        })
    return transcripts


def build_stance_change_instances(df: pd.DataFrame) -> pd.DataFrame:
    """Derive joint/divergent proposition-versus-side movement instances."""
    required = DEBATEGPT_FIELDS | {"participant_id", "turn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot derive movement; missing DebateGPT columns: {sorted(missing)}")
    result = df.sort_values(["participant_id", "turn"]).copy()
    result["d_agreement"] = result["agreementPostTreatment"] - result["agreementPreTreatment"]
    result["d_side_agreement"] = result["sideAgreementPostTreatment"] - result["sideAgreementPreTreatment"]
    result = result[result["d_agreement"].abs() > 0].copy()
    result["movement_class"] = np.where(
        np.sign(result["d_agreement"]) == np.sign(result["d_side_agreement"]),
        "joint_movement",
        "divergent_movement",
    )
    return result.reset_index(drop=True)


def sample_for_annotation(
    instances: pd.DataFrame, n: int = 150, seed: int = 42, out_path: Optional[str] = None
) -> pd.DataFrame:
    """Sample joint/divergent DebateGPT movement instances reproducibly."""
    if "movement_class" not in instances.columns:
        raise ValueError("instances must contain movement_class")
    n = min(n, len(instances))
    per_class = n // 2
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in instances.groupby("movement_class")
    ]
    sample = pd.concat(parts, ignore_index=True) if parts else instances.head(0).copy()
    if len(sample) < n:
        remaining = instances.drop(sample.index, errors="ignore")
        sample = pd.concat([sample, remaining.sample(n=min(n - len(sample), len(remaining)), random_state=seed)])
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["instance_id"] = [f"inst_{i:05d}" for i in range(len(sample))]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out_path, index=False)
    return sample


def load_args_me_corpus(
    data_path: Optional[str] = None,
    n_sample: int = 400,
    checkpoint_dir: str = "checkpoints",
    force: bool = False,
) -> pd.DataFrame:
    """Load a real local args.me sample without network or fallback data."""
    data_path = data_path or os.environ.get("ARGS_ME_DATA_PATH", "data/raw/args_me_corpus")
    ckpt = CheckpointManager(checkpoint_dir)
    cache_key = f"args_me__{stable_hash([str(data_path), n_sample])}"

    def _load():
        df = _read_tabular_files(data_path)
        df["synthetic"] = False
        logger.info("Loaded real args.me export from %s (%d rows)", data_path, len(df))
        return df.sample(n=min(n_sample, len(df)), random_state=42).reset_index(drop=True)

    return ckpt.load_or_compute(cache_key, _load, fmt="pkl", force=force)


def prepare_args_me_for_rhetorical_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Join observed args.me premise text without adding labels."""
    required = {"id", "premises", "context", "conclusion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"args.me export is missing columns: {sorted(missing)}")
    rows = []
    for record in df.itertuples(index=False):
        texts = [str(item.get("text", "")) for item in record.premises if isinstance(item, dict)]
        context = record.context
        rows.append({
            "arg_id": str(record.id),
            "text": " ".join(text for text in texts if text).strip(),
            "conclusion": str(record.conclusion),
            "portal": str(context.get("sourceTitle", "")) if isinstance(context, dict) else "",
            "synthetic": False,
        })
    prepared = pd.DataFrame(rows)
    if prepared.empty or prepared["text"].eq("").all():
        raise ValueError("args.me preparation produced no non-empty argument text")
    return prepared


def save_transcript_json(transcript: dict, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return out


def load_transcript_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
