#!/usr/bin/env python3
"""Fail-closed audit for the DebateGPT-based evaluation repository."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

REQUIRED_DIRS = ("data/raw", "data/processed", "data/annotations", "data/user_study", "reports")
REQUIRED_FILES = (
    "README.md", "requirements.txt", "src/data_loader.py", "src/metrics.py",
    "src/engine.py", "src/paper_metrics.py", "scripts/run_real_evaluation.py",
)
DEBATEGPT_FIELDS = {
    "agreementPreTreatment", "agreementPostTreatment",
    "sideAgreementPreTreatment", "sideAgreementPostTreatment",
}
MARKERS = ("_synthesize_", "backend=\"mock\"", "backend='mock'", "fallback to synthetic", "pseudo_truth", "system_label")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: list[str] = field(default_factory=list)


def check(name: str, status: str, detail: str, evidence: Iterable[str] = ()) -> Check:
    return Check(name, status, detail, list(evidence))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def headers(path: Path) -> set[str] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return set(next(csv.reader(handle)))
    except (OSError, StopIteration, UnicodeError, csv.Error):
        return None


def data_files(path: Path) -> list[Path]:
    return sorted(
        file for file in path.glob("*")
        if file.is_file() and file.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}
        and not file.name.startswith(".") and file.name != ".download_manifest.json"
    )


def first_json_keys(path: Path) -> set[str] | None:
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                return set(json.loads(next(handle)).keys())
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                return set(value[0].keys()) if value else set()
            return set(value.keys())
    except (OSError, StopIteration, UnicodeError, json.JSONDecodeError, AttributeError):
        return None
    return None


def audit(root: Path) -> dict:
    root = root.resolve()
    checks: list[Check] = []
    missing_dirs = [name for name in REQUIRED_DIRS if not (root / name).is_dir()]
    checks.append(check("required_directories", "PASS" if not missing_dirs else "FAIL", "Required directories exist." if not missing_dirs else "Required directories are missing.", missing_dirs))
    missing_files = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    checks.append(check("required_files", "PASS" if not missing_files else "FAIL", "Core files exist." if not missing_files else "Core files are missing.", missing_files))

    debategpt = data_files(root / "data/raw/debategpt") if (root / "data/raw/debategpt").is_dir() else []
    schema_errors = []
    for path in debategpt:
        observed = headers(path) if path.suffix == ".csv" else first_json_keys(path)
        if observed is None:
            schema_errors.append(f"{path.relative_to(root)}: unreadable")
        else:
            missing = DEBATEGPT_FIELDS - observed
            if missing:
                schema_errors.append(f"{path.relative_to(root)}: missing {sorted(missing)}")
    checks.append(check("primary_debategpt", "PASS" if debategpt and not schema_errors else "FAIL", "DebateGPT files and four agreement fields are present." if debategpt and not schema_errors else "DebateGPT data or required fields are missing.", [str(path.relative_to(root)) for path in debategpt] + schema_errors))

    annotations = sorted((root / "data/annotations").glob("*.csv")) if (root / "data/annotations").is_dir() else []
    studies = sorted((root / "data/user_study").glob("*.csv")) if (root / "data/user_study").is_dir() else []
    schema_errors = []
    for path in annotations:
        observed = headers(path)
        if observed is None or not {"instance_id", "annotator", "label"}.issubset(observed):
            schema_errors.append(f"{path.relative_to(root)}: expected instance_id, annotator, label")
    for path in studies:
        observed = headers(path)
        if observed is None or not {"participant_id", "condition", "accuracy", "time_to_answer"}.issubset(observed):
            schema_errors.append(f"{path.relative_to(root)}: expected participant_id, condition, accuracy, time_to_answer")
    checks.append(check("human_study_inputs", "PASS" if annotations and studies and not schema_errors else "FAIL", "Human annotation and user-study inputs are present and valid." if annotations and studies and not schema_errors else "Human annotation/user-study inputs are missing or invalid.", schema_errors))

    processed = [root / "data/processed/debategpt_instances.csv", root / "data/processed/cw_por.csv", root / "data/processed/convergence.csv"]
    missing_processed = [str(path.relative_to(root)) for path in processed if not path.is_file() or path.stat().st_size == 0]
    checks.append(check("processed_evaluation_inputs", "PASS" if not missing_processed else "FAIL", "Processed H1-H6 inputs exist." if not missing_processed else "Processed DebateGPT evaluation inputs are missing.", missing_processed))

    source_files = sorted((root / "src").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    marker_hits = []
    for path in source_files + sorted((root / "notebooks").glob("*.ipynb")):
        if path.name == "run_reproducibility_audit.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in MARKERS:
            if marker.lower() in text:
                marker_hits.append(f"{path.relative_to(root)} contains {marker!r}")
    checks.append(check("proxy_scan", "WARN" if marker_hits else "PASS", "Proxy markers found." if marker_hits else "No known proxy-generation markers found.", marker_hits))

    syntax_errors = []
    for path in source_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            syntax_errors.append(f"{path.relative_to(root)}: {exc}")
    checks.append(check("python_syntax", "PASS" if not syntax_errors else "FAIL", "Python source parses successfully." if not syntax_errors else "Python syntax errors found.", syntax_errors))

    notebook_hits = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
            for needle in ("_synthesize_human_labels", "plausible effect", "system_label", "pseudo_truth", "public_stance", "private_stance"):
                if needle in text:
                    notebook_hits.append(f"{path.relative_to(root)} contains obsolete/proxy term {needle!r}")
        except (OSError, json.JSONDecodeError) as exc:
            notebook_hits.append(f"{path.relative_to(root)}: invalid JSON ({exc})")
    checks.append(check("notebook_evaluation_integrity", "FAIL" if notebook_hits else "PASS", "Notebook evaluation is clean." if not notebook_hits else "Notebook contains obsolete/proxy evaluation code.", notebook_hits))

    real_ready = all(item.status == "PASS" for item in checks)
    checks.append(check("real_results_gate", "PASS" if real_ready else "BLOCKED", "Authentic metrics are eligible." if real_ready else "Authentic metrics remain blocked until all observed artifacts pass."))
    return {"root": str(root), "real_results_eligible": real_ready, "checks": [asdict(item) for item in checks]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit(args.root)
    report = [f"# Reproducibility Audit\n\nRoot: `{result['root']}`\n", f"\n**Authentic-results eligible:** `{result['real_results_eligible']}`\n"]
    for item in result["checks"]:
        report.append(f"\n## {item['name']} — {item['status']}\n{item['detail']}\n")
        report.extend(f"- {e}\n" for e in item["evidence"])
    text = "".join(report)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if result["real_results_eligible"] or not args.strict else 2


if __name__ == "__main__":
    sys.exit(main())
