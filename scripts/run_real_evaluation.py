#!/usr/bin/env python3
"""Compute DebateGPT evaluation metrics from observed CSV files only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper_metrics import cw_por, pearson_nca_correlation

DEBATEGPT_FIELDS = {
    "condition", "participant_id", "group_id", "topic", "turn", "public_message",
    "agreementPreTreatment", "agreementPostTreatment",
    "sideAgreementPreTreatment", "sideAgreementPostTreatment",
}
SCHEMAS = {
    "debategpt": DEBATEGPT_FIELDS,
    "cw_por": {"normalized_jsd_weight", "side_only_without_proposition_change"},
    "convergence": {"system_pre", "system_post", "agreementPreTreatment", "agreementPostTreatment"},
    "annotations": {"instance_id", "annotator", "label"},
    "user_study": {"participant_id", "condition", "accuracy", "time_to_answer"},
}


def read_csv(path: Path, schema_name: str) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required observed file is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = SCHEMAS[schema_name] - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no observed rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debategpt", type=Path, required=True)
    parser.add_argument("--cw-por", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    debategpt = read_csv(args.debategpt, "debategpt")
    human_human = [row for row in debategpt if row["condition"].strip().lower().replace("_", "-") == "human-human"]
    if not human_human:
        raise ValueError("DebateGPT evaluation input contains no human-human rows")
    cw_rows = read_csv(args.cw_por, "cw_por")
    convergence = read_csv(args.convergence, "convergence")
    convergence_records = [
        {
            "system_pre": float(row["system_pre"]),
            "system_post": float(row["system_post"]),
            "agreementPreTreatment": float(row["agreementPreTreatment"]),
            "agreementPostTreatment": float(row["agreementPostTreatment"]),
        }
        for row in convergence
    ]
    report = {
        "provenance": {
            "synthetic": False,
            "primary_dataset": "DebateGPT",
            "condition": "human-human",
            "inputs": [str(args.debategpt), str(args.cw_por), str(args.convergence)],
        },
        "debategpt": {"rows_read": len(debategpt), "human_human_rows": len(human_human)},
        "h5_nca_correlation": pearson_nca_correlation(
            [{"system_pre": row["system_pre"], "system_post": row["system_post"]} for row in convergence_records],
            [{"agreementPreTreatment": row["agreementPreTreatment"], "agreementPostTreatment": row["agreementPostTreatment"]} for row in convergence_records],
        ),
        "h6_cw_por": cw_por(cw_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
