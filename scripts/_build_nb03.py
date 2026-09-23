"""Build notebook 03 for the observed DebateGPT evaluation plan."""
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
        cell("markdown", """# DebateGPT H1-H6 evaluation and ablations

This notebook is fail-closed. It uses human-human DebateGPT records, observed annotations, observed user-study responses, and observed model outputs only. It never fabricates rows or metrics.

Required processed inputs:
- `data/processed/debategpt_instances.csv`
- `data/processed/cw_por.csv`
- `data/processed/convergence.csv`
- `data/annotations/human_labels.csv`
- `data/user_study/results.csv` with `time_to_answer`

H4 includes no rhetorical annotation, no stance layer, no echo detection, and no priority order; the final condition reports per-signal accuracy.""", "nb03-md"),
        cell("code", """from pathlib import Path
import sys

root = Path.cwd()
while root != root.parent and not (root / 'src').is_dir():
    root = root.parent
sys.path.insert(0, str(root))
""", "nb03-imports"),
        cell("code", """required = {
    'debategpt_instances': root / 'data' / 'processed' / 'debategpt_instances.csv',
    'cw_por': root / 'data' / 'processed' / 'cw_por.csv',
    'convergence': root / 'data' / 'processed' / 'convergence.csv',
    'annotations': root / 'data' / 'annotations' / 'human_labels.csv',
    'user_study': root / 'data' / 'user_study' / 'results.csv',
}
missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
if missing:
    print({'status': 'blocked', 'reason': 'observed DebateGPT evaluation inputs are missing', 'missing': missing, 'required_paths': {name: str(path) for name, path in required.items()}})
else:
    import subprocess
    subprocess.run([sys.executable, str(root / 'scripts' / 'run_real_evaluation.py'), '--debategpt', str(required['debategpt_instances']), '--cw-por', str(required['cw_por']), '--convergence', str(required['convergence']), '--output', str(root / 'reports' / 'real_evaluation.json')], check=True)
    print({'status': 'completed', 'report': str(root / 'reports' / 'real_evaluation.json')})
""", "nb03-gate"),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/03_evaluation_ablation.ipynb", "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)
print("wrote notebooks/03_evaluation_ablation.ipynb")
