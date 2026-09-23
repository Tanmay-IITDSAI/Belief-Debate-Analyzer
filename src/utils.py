"""
src/utils.py
Central utilities shared across all three notebooks of the Belief-Tracking
Debate Analyzer pipeline.

Responsibilities:
  1. Environment / credential handling (HF_TOKEN, .env loading).
  2. Crash-resilient checkpointing (CheckpointManager) so that any unit of
     work -- a data batch, a training/generation step, an ablation variant --
     can be resumed after a kernel death without recomputation.
  3. Structured markdown logging (MarkdownLogger) that appends timestamped,
     tabular entries to logs/data_log.md, logs/exp_log.md, logs/ablation_log.md.
  4. Consistent, publication-ready plotting theme (300 DPI, colorblind-safe
     Okabe-Ito palette, whitegrid style) for direct inclusion in the paper.
  5. Global seeding for reproducibility.

No placeholders: every function here is fully implemented and is exercised
by the accompanying notebooks.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("belief_debate_analyzer")


# ---------------------------------------------------------------------------
# Environment / credentials
# ---------------------------------------------------------------------------
def load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv).
    Only sets a variable if it is not already present in the environment,
    so real environment/session variables always take precedence."""
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_hf_token(required: bool = False) -> Optional[str]:
    """Read the Hugging Face Hub token from the environment.
    NEVER hardcode credentials -- this is the only sanctioned way tokens
    enter the pipeline. Returns None if unset; callers that need a gated
    model (e.g. Llama-3.1-8B-Instruct) should set required=True to fail
    loudly and early rather than mid-batch."""
    token = os.environ.get("HF_TOKEN")
    if token is None:
        msg = (
            "HF_TOKEN not found in environment. Gated/private HF models "
            "(e.g. meta-llama/Llama-3.1-8B-Instruct) will fail to download. "
            "Set it with `export HF_TOKEN=...` or place it in a local .env "
            "file (see .env.example) before starting Jupyter."
        )
        if required:
            raise RuntimeError(msg)
        logger.warning(msg)
    return token


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG we touch so pipeline runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # optional heavy dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash used to derive checkpoint/cache keys from
    arbitrary content (e.g. a transcript's text) so re-running a cell with
    unchanged input reuses the exact same cached artifact."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Crash-resilient checkpointing
# ---------------------------------------------------------------------------
class CheckpointManager:
    """
    Generic, crash-resilient checkpointing used by every notebook.

    Every unit of work (a preprocessed dataframe, a generated debate, a
    batch of annotations, an ablation run) gets a stable string `key`.
    `save()` persists arbitrary state; `load()` / `exists()` let a re-run
    of a cell resume instead of recompute. Writes are atomic (write to a
    `.tmp` file, then `os.replace`) so a kernel death mid-write can never
    leave a corrupt checkpoint behind.
    """

    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, fmt: str) -> Path:
        safe_key = key.replace("/", "__").replace(" ", "_")
        return self.dir / f"{safe_key}.{fmt}"

    def exists(self, key: str, fmt: str = "pkl") -> bool:
        return self._path(key, fmt).exists()

    def save(self, key: str, obj: Any, fmt: str = "pkl") -> Path:
        path = self._path(key, fmt)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if fmt == "pkl":
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "json":
            with open(tmp_path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        elif fmt == "pt":
            import torch

            torch.save(obj, tmp_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {fmt}")
        os.replace(tmp_path, path)  # atomic on POSIX filesystems
        return path

    def load(self, key: str, fmt: str = "pkl") -> Any:
        path = self._path(key, fmt)
        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif fmt == "json":
            with open(path) as f:
                return json.load(f)
        elif fmt == "pt":
            import torch

            return torch.load(path, map_location="cpu")
        raise ValueError(f"Unsupported checkpoint format: {fmt}")

    def load_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        fmt: str = "pkl",
        force: bool = False,
    ) -> Any:
        """The core resume primitive used throughout the pipeline.
        If a checkpoint for `key` already exists, load and return it
        (no recomputation). Otherwise call `compute_fn()`, persist the
        result, and return it. This is what makes 're-run the cell' safe
        after a kernel death: the expensive work already done is skipped."""
        if not force and self.exists(key, fmt):
            logger.info("Checkpoint hit: %s.%s (skipping recompute)", key, fmt)
            return self.load(key, fmt)
        logger.info("Checkpoint miss: %s.%s (computing)", key, fmt)
        result = compute_fn()
        self.save(key, result, fmt)
        return result

    def list_keys(self) -> list:
        return sorted(p.stem for p in self.dir.glob("*") if not p.name.endswith(".tmp"))


# ---------------------------------------------------------------------------
# Markdown progress logging
# ---------------------------------------------------------------------------
class MarkdownLogger:
    """Appends structured, timestamped entries to a markdown log file.
    Three logical logs are used across the pipeline:
      logs/data_log.md      <- 01_data_pipeline.ipynb
      logs/exp_log.md       <- 02_experimentation.ipynb
      logs/ablation_log.md  <- 03_evaluation_ablation.ipynb
    """

    def __init__(self, log_path: str, title: str):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write(f"# {title}\n\n")
                f.write(f"_Log created {datetime.datetime.now().isoformat()}_\n\n")
                f.write("---\n\n")

    def log(self, section: str, body: str = "", metrics: Optional[dict] = None) -> None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a") as f:
            f.write(f"## {section}\n")
            f.write(f"_{ts}_\n\n")
            if body:
                f.write(body.strip() + "\n\n")
            if metrics:
                f.write("| Metric | Value |\n|---|---|\n")
                for k, v in metrics.items():
                    f.write(f"| {k} | {v} |\n")
                f.write("\n")

    def log_hparams(self, hparams: dict, section: str = "Run configuration") -> None:
        self.log(section, "Hyperparameters / run configuration for this step:", metrics=hparams)

    def log_error(self, section: str, exc: Exception) -> None:
        self.log(section, body=f"**ERROR**: `{type(exc).__name__}: {exc}`")


# ---------------------------------------------------------------------------
# Publication-ready plotting theme
# ---------------------------------------------------------------------------
# Okabe-Ito palette: colorblind-safe, high contrast.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def set_plot_theme() -> list:
    """Apply a clean, high-contrast, 300-DPI plotting theme suitable for
    direct inclusion in a 6-page EACL paper. Call once per notebook/session
    before plotting; returns the palette in case callers want to index
    into it directly."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    sns.set_palette(sns.color_palette(OKABE_ITO))
    plt.rcParams.update(
        {
            "figure.dpi": 150,       # on-screen
            "savefig.dpi": 300,      # export quality
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.figsize": (6.0, 4.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    return OKABE_ITO


def savefig(fig, name: str, figures_dir: str = "figures") -> Path:
    """Save a matplotlib figure at 300 DPI into the figures/ directory,
    creating it if necessary. Returns the path written."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure: %s", path)
    return path
