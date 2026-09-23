"""Build notebook 02 for human-human DebateGPT analysis."""
import json


def cell(kind, source, cell_id):
    value = {
        "cell_type": kind,
        "metadata": {"id": cell_id, "language": "python" if kind == "code" else "markdown"},
        "source": source.splitlines(True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


notebook = {
    "cells": [
        cell("markdown", """# Primary experimentation on DebateGPT human-human debates

This notebook analyzes only the human-human DebateGPT condition for primary results. Agent-generated debates are secondary qualitative material and are never used for H1-H6 ground truth.""", "nb02-md"),
        cell("code", """from pathlib import Path
import importlib
import os
import sys
from getpass import getpass

root = Path.cwd()
while root != root.parent and not (root / 'src').is_dir():
    root = root.parent
sys.path.insert(0, str(root))
from src import data_loader as dl
from src import engine
importlib.reload(dl)
importlib.reload(engine)
DebateAnalysisPipeline = engine.DebateAnalysisPipeline

if not os.environ.get('HF_TOKEN'):
    os.environ['HF_TOKEN'] = getpass('Enter HF_TOKEN for this notebook session (input hidden): ')
if not os.environ['HF_TOKEN']:
    raise RuntimeError('HF_TOKEN is required for the prompted Llama analysis.')
""", "nb02-imports"),
        cell("code", """debategpt_path = root / 'data' / 'raw' / 'debategpt'
rhetorical_repo = __import__('os').environ.get('RHETORICAL_MODEL_REPO')
debategpt_files = [p for p in debategpt_path.glob('*') if p.is_file() and p.suffix.lower() in {'.json', '.jsonl', '.csv'}]
if not debategpt_files:
    results = None
    print({'status': 'blocked', 'reason': 'DebateGPT export is missing', 'required_path': str(debategpt_path)})
else:
    try:
        debategpt = dl.load_debategpt(data_path=str(debategpt_path), checkpoint_dir=str(root / 'checkpoints'), force=False, human_human_only=True)
        transcripts = dl.build_transcripts(debategpt)
        pipeline = DebateAnalysisPipeline(backend='hf', checkpoint_dir=str(root / 'checkpoints'))
        results = pipeline.analyze_batch(transcripts)
        print({'status': 'completed', 'condition': 'human-human', 'transcripts': len(results), 'backend': 'hf', 'classifier_override': bool(rhetorical_repo)})
    except (OSError, RuntimeError, ValueError) as exc:
        results = None
        print({'status': 'blocked', 'reason': 'HF model initialization or inference failed', 'error_type': type(exc).__name__, 'error': str(exc), 'next_step': 'Verify HF_TOKEN access, CUDA/bitsandbytes compatibility, and available GPU memory, then rerun.'})
""", "nb02-primary"),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/02_experimentation.ipynb", "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)
print("wrote notebooks/02_experimentation.ipynb")
