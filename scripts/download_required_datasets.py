#!/usr/bin/env python3
"""Download verified public DebateGPT and args.me dataset revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEBATEGPT_REPO = "frasalvi/debategpt"
DEBATEGPT_REVISION = "ff19fc4cd38424023b8db616f7358a9b94e4c9d5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(directory: Path, dataset: str, repo_id: str, revision: str, license_name: str) -> None:
    files = [
        {"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.iterdir()) if path.is_file() and path.name != ".download_manifest.json"
    ]
    (directory / ".download_manifest.json").write_text(json.dumps({
        "dataset": dataset,
        "repo_id": repo_id,
        "revision": revision,
        "license": license_name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--args-me", action="store_true")
    parser.add_argument("--revision", default=DEBATEGPT_REVISION)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    debategpt_dir = args.root / "data" / "raw" / "debategpt"
    snapshot_download(
        repo_id=DEBATEGPT_REPO,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(debategpt_dir),
        allow_patterns=["debategpt.csv", "README.md", ".gitattributes"],
    )
    write_manifest(debategpt_dir, "debategpt", DEBATEGPT_REPO, args.revision, "CC-BY-SA-4.0")
    print({"dataset": "DebateGPT", "repo": DEBATEGPT_REPO, "revision": args.revision, "path": str(debategpt_dir)})

    if args.args_me:
        args_dir = args.root / "data" / "raw" / "args_me_corpus"
        snapshot_download(repo_id="webis/args_me", repo_type="dataset", local_dir=str(args_dir))
        write_manifest(args_dir, "args_me", "webis/args_me", "main", "source repository license")
        print({"dataset": "args.me", "repo": "webis/args_me", "path": str(args_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
