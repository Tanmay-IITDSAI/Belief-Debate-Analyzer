"""Build notebook 01 for public DebateGPT and args.me preparation."""
import json


def cell(kind, source, cell_id):
    return {
        "cell_type": kind,
        "metadata": {"id": cell_id, "language": "python" if kind == "code" else "markdown"},
        "source": source.splitlines(True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


notebook = {
    "cells": [
        cell("markdown", """# Data pipeline: DebateGPT primary, args.me secondary

This notebook downloads or loads public DebateGPT data, filters the human-human condition, validates the four agreement fields, and prepares args.me. It never creates replacement rows.""", "nb01-md"),
        cell("code", """from pathlib import Path
import json
import importlib
import os
import sys

root = Path.cwd()
while root != root.parent and not (root / 'src').is_dir():
    root = root.parent
sys.path.insert(0, str(root))
from src import data_loader as dl
importlib.reload(dl)
""", "nb01-imports"),
        cell("code", """debategpt_path = root / 'data' / 'raw' / 'debategpt'
debategpt_files = [p for p in debategpt_path.glob('*') if p.is_file() and p.suffix.lower() in {'.json', '.jsonl', '.csv'}]
if not debategpt_files:
    import subprocess
    subprocess.run([sys.executable, str(root / 'scripts' / 'download_required_datasets.py')], check=True)
    print({'status': 'downloaded', 'path': str(debategpt_path)})
else:
    print({'status': 'using_local_export', 'path': str(debategpt_path)})
""", "nb01-download"),
        cell("code", """debategpt = dl.load_debategpt(data_path=str(debategpt_path), checkpoint_dir=str(root / 'checkpoints'), force=True)
args_path = root / 'data' / 'raw' / 'args_me_corpus'
args_me = dl.load_args_me_corpus(data_path=str(args_path), checkpoint_dir=str(root / 'checkpoints'), force=False)
instances = dl.build_stance_change_instances(debategpt)
report = {
    'primary_dataset': 'DebateGPT',
    'condition': 'human-human',
    'debategpt_path': str(debategpt_path),
    'debategpt_rows': int(len(debategpt)),
    'debategpt_columns': list(debategpt.columns),
    'movement_rows': int(len(instances)),
    'movement_classes': instances['movement_class'].value_counts().to_dict(),
    'args_me_rows': int(len(args_me)),
    'sample': debategpt.head(5).to_dict(orient='records'),
}
report_path = root / 'reports' / 'data_verification.json'
report_path.write_text(json.dumps(report, indent=2, default=str) + '\\n', encoding='utf-8')
print({'status': 'verified', 'report': str(report_path), 'debategpt_rows': len(debategpt), 'args_me_rows': len(args_me)})
""", "nb01-verify"),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/01_data_pipeline.ipynb", "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)
print("wrote notebooks/01_data_pipeline.ipynb")
