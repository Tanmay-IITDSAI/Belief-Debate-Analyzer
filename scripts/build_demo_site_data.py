#!/usr/bin/env python3
"""Shape the real per-transcript system output into demo_site/data.json.

Reads only already-computed, real artifacts (no model calls, no generation):
  data/processed/debategpt_human_human_system_output.json
  data/processed/debategpt_instances.csv
and writes a single static JSON file the reviewer-facing demo site reads
client-side. Every value in the output traces back to an observed DebateGPT
field or a value this repository's own models actually produced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_agreement_lookup(instances_path: Path) -> dict:
    lookup: dict[str, dict] = {}
    with instances_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            lookup[row["participant_id"]] = {
                "agreementPreTreatment": int(row["agreementPreTreatment"]),
                "agreementPostTreatment": int(row["agreementPostTreatment"]),
                "sideAgreementPreTreatment": int(row["sideAgreementPreTreatment"]),
                "sideAgreementPostTreatment": int(row["sideAgreementPostTreatment"]),
            }
    return lookup


def build(system_output_path: Path, instances_path: Path) -> list[dict]:
    results = json.loads(system_output_path.read_text(encoding="utf-8"))
    agreement = _load_agreement_lookup(instances_path)

    transcripts = []
    for result in results:
        turns = []
        for i, speaker in enumerate(result["speakers"]):
            rhetoric = result["rhetoric"][i]
            attribution = result["attribution"][i]
            turns.append({
                "speaker": speaker,
                "text": result["texts"][i],
                "rhetoric": rhetoric,
                "quality": result["quality"][i],
                "evidence_direction": result["evidence_direction"][i],
                "stance": result["stance"][i],
                "attribution": attribution,
            })
        participants = sorted(set(result["speakers"]))
        transcripts.append({
            "transcript_id": result["transcript_id"],
            "topic": result["topic"],
            "turns": turns,
            "agreement": {p: agreement.get(p) for p in participants},
        })

    transcripts.sort(key=lambda t: t["topic"])
    return transcripts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-output", type=Path,
                         default=ROOT / "data" / "processed" / "debategpt_human_human_system_output.json")
    parser.add_argument("--instances", type=Path,
                         default=ROOT / "data" / "processed" / "debategpt_instances.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "demo_site" / "data.json")
    args = parser.parse_args()

    if not args.system_output.is_file():
        raise FileNotFoundError(f"missing real system output: {args.system_output}")
    if not args.instances.is_file():
        raise FileNotFoundError(f"missing real instances file: {args.instances}")

    transcripts = build(args.system_output, args.instances)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(transcripts, indent=None, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "transcripts": len(transcripts),
        "out": str(args.out),
        "bytes": args.out.stat().st_size,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
