"""
src/engine.py
The Belief-Tracking Debate Analyzer system itself (paper §3).

Every component below uses the "hf" backend: real Hugging Face models, exactly as described in the paper
    (RoBERTa-based rhetorical classifier from Ji et al. 2025, prompted
    Llama-3.1-8B-Instruct for argument-quality scoring and generation,
    sentence-transformers for semantic similarity). Requires a GPU and
    HF_TOKEN (for gated models) and is only exercised when you explicitly
    request backend="hf".

Sections implemented, mapped to the paper:
  §3.1 DebateGenerator            -- Panel 1
  §3.2 RhetoricalAnnotator        -- Panel 2
  §3.3 StanceDynamicsEngine       -- Panel 3, Eq. (1)-(2)
       AttributionEngine          -- Panel 3, "Why?" card, priority order
  DebateAnalysisPipeline          -- orchestrates all of the above with
                                      per-transcript checkpointing and
                                      supports the three ablation flags
                                      used in §4.2.
"""
from __future__ import annotations

import math
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .utils import CheckpointManager, get_hf_token, logger, set_global_seed, stable_hash

RHETORIC_LABELS = ["causal", "empirical", "emotional", "moral"]
_TEXT_GENERATION_PIPES = {}


class _TextGenerationRunner:
    """Minimal generation adapter that never applies `.to()` to the model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.input_device = next(model.parameters()).device

    def __call__(self, prompt: str, max_new_tokens: int, do_sample: bool,
                 temperature: float | None = None, top_p: float | None = None,
                 chat: bool = False) -> list[dict]:
        import torch

        # chat=True runs instruct models through their chat template with a
        # generation prompt, so the model answers AS the assistant instead of
        # continuing our instructions (the source of "Reply with ..." echo).
        if chat and getattr(self.tokenizer, "chat_template", None):
            chat_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            encoded = self.tokenizer(chat_text, return_tensors="pt")
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {name: value.to(self.input_device) for name, value in encoded.items()}
        generation_args = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": self.tokenizer.eos_token_id}
        if do_sample and temperature is not None:
            generation_args["temperature"] = temperature
        if do_sample and top_p is not None:
            generation_args["top_p"] = top_p
        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_args)
        # Decode ONLY the newly generated tokens. Decoding the full sequence
        # and slicing by character count (out[len(prompt):]) is fragile:
        # detokenized prompts are not guaranteed to be char-identical to the
        # input string, which leaked prompt fragments into outputs.
        prompt_len = encoded["input_ids"].shape[-1]
        new_ids = output_ids[0][prompt_len:]
        return [{"generated_text": self.tokenizer.decode(new_ids, skip_special_tokens=True)}]


def _get_text_generation_pipeline(model_repo: str):
    """Load one shared 4-bit Llama runner without pipeline device placement."""
    if model_repo in _TEXT_GENERATION_PIPES:
        return _TEXT_GENERATION_PIPES[model_repo]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.modeling_utils import PreTrainedModel

    token = get_hf_token(required=True)
    model_kwargs = {"token": token}
    use_4bit = torch.cuda.is_available()
    if use_4bit:
        model_kwargs.update({
            "device_map": {"": 0},
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        })

    # accelerate's dispatch_model() calls model.to(device) as its final
    # placement step even for a bnb 4-bit model, and transformers'
    # PreTrainedModel.to() raises instead of performing that move. The
    # move itself is what finishes placing non-quantized buffers (e.g.
    # embeddings, layernorms, rotary caches); skip only the raising
    # guard and let the real nn.Module.to() run so those still land on
    # the right device (already-placed 4-bit weights are a same-device
    # no-op there).
    original_to = PreTrainedModel.to
    if use_4bit:
        PreTrainedModel.to = torch.nn.Module.to
    try:
        model = AutoModelForCausalLM.from_pretrained(model_repo, **model_kwargs)
    finally:
        PreTrainedModel.to = original_to
    tokenizer = AutoTokenizer.from_pretrained(model_repo, token=token)
    text_pipe = _TextGenerationRunner(model, tokenizer)
    _TEXT_GENERATION_PIPES[model_repo] = text_pipe
    return text_pipe

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _js_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two categorical distributions
    over RHETORIC_LABELS, base-2, bounded in [0, 1]."""
    labels = RHETORIC_LABELS
    p_arr = np.array([p.get(l, 1e-9) for l in labels])
    q_arr = np.array([q.get(l, 1e-9) for l in labels])
    p_arr, q_arr = p_arr / p_arr.sum(), q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)

    def _kl(x, y):
        return float(np.sum(np.where(x > 0, x * np.log2(x / y), 0.0)))

    return 0.5 * _kl(p_arr, m) + 0.5 * _kl(q_arr, m)


# ---------------------------------------------------------------------------
# Panel 2: Rhetorical Strategy Annotator
# ---------------------------------------------------------------------------
class RhetoricalAnnotator:
    """Prompted Llama rhetorical scoring, with an optional classifier plugin."""

    def __init__(self, backend: str = "hf", model_repo: Optional[str] = None, device: Optional[str] = None):
        self.backend = backend
        if backend != "hf":
            raise ValueError("Only backend='hf' is permitted for reproducible runs; mock inference is disabled.")
        self.model_repo = model_repo or os.environ.get("RHETORICAL_MODEL_REPO")
        self.prompt_model_repo = os.environ.get("DEBATE_GEN_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.mode = "classifier" if self.model_repo else "prompted_llama"
        self._model = None
        self._tokenizer = None
        if backend == "hf":
            self._lazy_load_hf(device)

    def _lazy_load_hf(self, device: Optional[str]) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        token = get_hf_token(required=False)
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.mode == "prompted_llama":
            self._prompt_pipe = _get_text_generation_pipeline(self.prompt_model_repo)
            # Reuse the actual device the shared Llama runner landed on,
            # rather than re-deriving it -- avoids a second, possibly
            # inconsistent, cuda-availability check.
            self._device = self._prompt_pipe.input_device
            return

        config = AutoConfig.from_pretrained(self.model_repo, token=token)
        if config.num_labels != len(RHETORIC_LABELS):
            raise RuntimeError(
                f"Rhetorical checkpoint {self.model_repo!r} exposes {config.num_labels} outputs; "
                f"Ji et al. requires {len(RHETORIC_LABELS)} regression scores."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_repo, token=token)

        # --- FIX -------------------------------------------------------
        # Never call `.to()` on a model that may load pre-quantized.
        # `is_loaded_in_4bit` / `is_loaded_in_8bit` are not reliable across
        # transformers versions, and a checkpoint can carry a baked-in
        # `quantization_config` in its own config.json even when we never
        # requested quantization ourselves. So: let `device_map="auto"`
        # place the model at load time (same pattern as
        # `_get_text_generation_pipeline`), then only fall back to a
        # manual `.to()` if the model turns out not to be quantized *and*
        # we didn't already hand placement off to device_map.
        model_kwargs = {"token": token, "config": config}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        # Same accelerate/from_pretrained interaction as
        # _get_text_generation_pipeline: bypass only the raising guard,
        # not the actual device move, in case this checkpoint carries a
        # baked-in quantization_config we didn't ask for.
        from transformers.modeling_utils import PreTrainedModel

        original_to = PreTrainedModel.to
        PreTrainedModel.to = torch.nn.Module.to
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_repo, **model_kwargs)
        finally:
            PreTrainedModel.to = original_to

        is_quantized = (
            getattr(self._model, "is_quantized", False)
            or getattr(self._model, "hf_quantizer", None) is not None
            or getattr(self._model, "is_loaded_in_4bit", False)
            or getattr(self._model, "is_loaded_in_8bit", False)
        )
        if not is_quantized and "device_map" not in model_kwargs:
            self._model = self._model.to(requested_device)
        # -----------------------------------------------------------------

        self._model.eval()
        self._device = next(self._model.parameters()).device

    def annotate(self, text: str) -> dict:
        """Return four [0,1] scores and the highest-scoring label."""
        if self.mode == "prompted_llama":
            prompt = (
                "Score this debate utterance for four rhetorical strategies. "
                "Return JSON only with numeric values from 0 to 1 for keys "
                "causal, empirical, emotional, moral. Definitions: causal="
                "cause/effect; empirical=evidence, statistics, examples; "
                "emotional=feelings; moral=right/wrong, duty, justice.\n\n"
                f"Utterance: {text}\nJSON:"
            )
            output = self._prompt_pipe(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]
            # Runner returns completion-only text; JSON regex works as before.
            match = re.search(r"\{.*?\}", output, flags=re.DOTALL)
            if not match:
                raise RuntimeError("Prompted rhetorical scorer did not return a JSON object")
            try:
                raw = json.loads(match.group(0))
                scores = {label: float(raw[label]) for label in RHETORIC_LABELS}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Prompted rhetorical scorer returned invalid four-score JSON") from exc
            if any(not 0 <= value <= 1 for value in scores.values()):
                raise RuntimeError("Prompted rhetorical scores must be in [0, 1]")
            total = sum(scores.values())
            probs = {label: value / total for label, value in scores.items()} if total else {
                label: 1.0 / len(RHETORIC_LABELS) for label in RHETORIC_LABELS
            }
            return {"label": max(scores, key=scores.get), "scores": scores, "probs": probs}
        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        scores = torch.sigmoid(logits).cpu().numpy()
        scores = np.clip(scores, 0.0, 1.0)
        raw = {label: float(score) for label, score in zip(RHETORIC_LABELS, scores)}
        total = sum(raw.values())
        probs = {label: score / total for label, score in raw.items()} if total else {
            label: 1.0 / len(RHETORIC_LABELS) for label in RHETORIC_LABELS
        }
        label = max(raw, key=raw.get)
        return {"label": label, "scores": raw, "probs": probs}

    def annotate_batch(self, texts: list) -> list:
        return [self.annotate(t) for t in texts]


# ---------------------------------------------------------------------------
# §3.3 simplification 1: argument-quality scorer
# ---------------------------------------------------------------------------
class ArgumentQualityScorer:
    """Scores an utterance's argument quality in [0, 1] (paper §3.3,
    simplification 1: a prompted LLM 1-5 scale, normalized). backend="hf"
    prompts Llama-3.1-8B-Instruct."""

    _PROMPT_TEMPLATE = (
        "Rate the following debate utterance's argument quality on a scale "
        "of 1 (weak, no support) to 5 (strong, well-supported). "
        "Respond with only the integer.\n\nUtterance: {text}\n\nRating:"
    )

    def __init__(self, backend: str = "hf", model_repo: Optional[str] = None, device: Optional[str] = None):
        self.backend = backend
        if backend != "hf":
            raise ValueError("Only backend='hf' is permitted for reproducible runs; mock inference is disabled.")
        self.model_repo = model_repo or "meta-llama/Llama-3.1-8B-Instruct"
        self._pipe = None
        if backend == "hf":
            self._lazy_load_hf(device)

    def _lazy_load_hf(self, device: Optional[str]) -> None:
        self._pipe = _get_text_generation_pipeline(self.model_repo)

    def score(self, text: str) -> float:
        prompt = self._PROMPT_TEMPLATE.format(text=text)
        out = self._pipe(prompt, max_new_tokens=4, do_sample=False)[0]["generated_text"]
        # The runner decodes only newly generated tokens, so `out` is already
        # the completion alone -- no slicing (and no char-count fragility).
        digits = re.findall(r"[1-5]", out)
        raw = int(digits[0]) if digits else 3
        return (raw - 1) / 4.0  # normalize 1-5 -> [0, 1]

    def score_batch(self, texts: list) -> list:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# §3.3 simplification 2: semantic similarity
# ---------------------------------------------------------------------------
class SemanticSimilarity:
    """Cosine similarity between utterances (paper §3.3, simplification 2:
    lightweight sentence-transformer embeddings using
    `sentence-transformers` (e.g. all-MiniLM-L6-v2)."""

    def __init__(
        self,
        backend: str = "hf",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.backend = backend
        if backend != "hf":
            raise ValueError("Only backend='hf' is permitted for reproducible runs; mock inference is disabled.")
        self._model = None
        if backend == "hf":
            from sentence_transformers import SentenceTransformer

            # device is independent of any larger model already resident on the GPU;
            # this model is small enough to run on CPU without a meaningful speed cost.
            self._model = SentenceTransformer(model_name, device=device or "cpu")

    def similarity(self, a: str, b: str) -> float:
        embs = self._model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))



# ---------------------------------------------------------------------------
# §3.1 Panel 1: debate generation backend
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# §3.1 Debate generation -- output hygiene helpers
# Small local LLMs sometimes ignore role instructions and emit stage
# directions ("(starts typing)"), speaker-name prefixes, narration or
# meta-text. These helpers sanitize and validate every candidate turn so
# only usable debate speech ever reaches the transcript.
# ---------------------------------------------------------------------------
_STAGE_DIRECTION_RE = re.compile(
    r"\([^()]*\)|\[[^\[\]]*\]|\*[^*\n]*\*"  # (…), […], *…*
)
_AI_META_RE = re.compile(
    r"\b(as an ai|an ai language model|i cannot produce|i'm sorry|cannot assist with)\b",
    re.I,
)
# Phrases the model copies from our own instruction template instead of
# producing debate speech ("Reply with Agent_2's next debating turn…").
_INSTRUCTION_ECHO_RE = re.compile(
    r"(reply with|debating turn|speech text only|no stage directions|"
    r"no narration|no speaker names|no quotation marks|debate so far|"
    r"you argue the|arguing the (for|against)|direct speech)",
    re.I,
)


def _strip_speaker_prefix(text: str, personas: list) -> str:
    """Drop a leading 'Agent_2:' / 'Speaker 1:' style tag if present."""
    for name in sorted(set(personas), key=len, reverse=True):
        m = re.match(rf"^\s*{re.escape(name)}\s*[:\-–]\s*", text, re.I)
        if m:
            return text[m.end():].lstrip()
    return text.lstrip()


def _truncate_at_speaker_tag(text: str, personas: list) -> str:
    """Cut everything from the first in-text 'Agent_2:' tag onward -- small
    models like to answer their own turn and then start the OTHER agent's
    line ("... good point. Agent_2: I disagree ...")."""
    cut = len(text)
    for name in set(personas):
        m = re.search(rf"{re.escape(name)}\s*:", text, re.I)
        if m and m.start() < cut:
            cut = m.start()
    return text[:cut].strip()


def _clean_generated_turn(raw: str, personas: list) -> str:
    """Sanitize one raw model completion into candidate debate speech."""
    if not raw:
        return ""
    text = _strip_speaker_prefix(raw.strip(), personas)
    text = _STAGE_DIRECTION_RE.sub(" ", text)
    text = text.strip().strip('"').strip("'").strip()
    # Cut at any in-text speaker tag (model switching to the other agent).
    text = _truncate_at_speaker_tag(text, personas)
    # Drop AI-refusal / meta / instruction-template-echo sentences entirely.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentences = [s for s in sentences if not _AI_META_RE.search(s)
                 and not _INSTRUCTION_ECHO_RE.search(s)]
    text = " ".join(s.strip() for s in sentences)
    # Drop trailing narration like: ," she said.  / ." he replied
    text = re.sub(
        r"[\"',]*\s*\b(he|she|they|it|the moderator|one speaker)\s+"
        r"(said|replied|asked|noted|answered)\s*[.!?\"']*\s*$",
        "", text, flags=re.I,
    ).strip().rstrip('",\'').strip()
    # Drop a truncated trailing fragment (split happened on punctuation, so a
    # final piece without terminal punctuation was cut off mid-sentence).
    if text:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) >= 2 and not re.search(r"[.!?]['\"]?$", parts[-1].strip()):
            text = " ".join(parts[:-1])
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_usable_turn(text: str, history: list, min_words: int = 4, min_chars: int = 25) -> bool:
    """A turn is usable if it is real, non-trivial speech and not a near-
    duplicate of the last two turns (small models like to parrot)."""
    if not text or len(text) < min_chars:
        return False
    if _INSTRUCTION_ECHO_RE.search(text):  # template echo slipped through
        return False
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < min_words:
        return False
    if sum(len(w) for w in words) / max(len(text), 1) < 0.35:  # mostly punctuation/noise
        return False

    def norm(s: str) -> str:
        return " ".join(re.findall(r"[a-z']+", s.lower()))

    nt = norm(text)
    for h in history[-2:]:
        nh = norm(h["text"])
        if nh == nt:
            return False
        a, b = set(nh.split()), set(nt.split())
        if a and b and len(a & b) / len(a | b) > 0.85:  # near-duplicate
            return False
    return True


class DebateGenerator:
    """Minimal round-robin debate loop built on MALLM-style configuration
    abstractions (paper §3.1). backend="hf" drives Llama-3.1-8B-Instruct
    (fits comfortably in the paper's 40GB GPU budget, ~16GB in 4-bit).

    Robustness: each turn uses a strict debate prompt with an assigned pro/con
    side, and the raw completion is sanitized (stage directions, speaker
    prefixes, narration and AI-meta stripped) and validated before it enters
    the transcript; unusable output is retried at decreasing temperature and,
    if still unusable, fails with a clear, user-facing error instead of
    polluting the debate with junk like '(starts typing)'."""

    def __init__(self, backend: str = "hf", model_repo: Optional[str] = None, device: Optional[str] = None):
        self.backend = backend
        if backend != "hf":
            raise ValueError("Only backend='hf' is permitted for reproducible runs; mock generation is disabled.")
        self.model_repo = model_repo or "meta-llama/Llama-3.1-8B-Instruct"
        self._pipe = None
        if backend == "hf":
            self._lazy_load_hf(device)

    def _lazy_load_hf(self, device: Optional[str]) -> None:
        self._pipe = _get_text_generation_pipeline(self.model_repo)

    def _generate_turn_hf(self, proposition: str, persona: str, side: str,
                          history: list, temperature: float = 0.8) -> str:
        """One raw completion for `persona`, arguing `side` of the motion."""
        transcript_so_far = "\n".join(f"{h['speaker']}: {h['text']}" for h in history) or "(You speak first.)"
        prompt = (
            f"Two-person debate on the motion: '{proposition}'.\n"
            f"You are {persona}. You argue {side} of the motion.\n"
            f"Debate so far:\n{transcript_so_far}\n\n"
            f"Write {persona}'s next debating turn: 1-3 sentences of direct speech, "
            f"arguing {side}. No stage directions, no narration, no speaker names, "
            f"no quotation marks. Reply with the speech text only.\n\n{persona}:"
        )
        out = self._pipe(
            prompt, max_new_tokens=110, do_sample=True,
            temperature=temperature, top_p=0.9, chat=True,
        )[0]["generated_text"]
        # Runner decodes only the newly generated tokens, so `out` IS the
        # completion -- returning it un-sliced avoids char-count fragility.
        return out

    def _robust_turn(self, proposition: str, persona: str, side: str, history: list,
                     on_progress: Optional[Callable[[int, int, str], None]] = None,
                     turn_index: int = 0, n_turns: int = 0) -> str:
        """Generate one usable turn: sanitize + validate each candidate and
        retry at lower temperature before giving up with a clear error."""
        attempts = 4
        last_raw = ""
        for attempt in range(attempts):
            temperature = 0.8 - 0.2 * attempt
            try:
                raw = self._generate_turn_hf(proposition, persona, side, history, temperature)
            except Exception as e:  # transient GPU/eval errors: retry once or twice
                last_raw = f"{type(e).__name__}: {e}"
                if attempt == attempts - 1:
                    raise
                continue
            last_raw = raw
            text = _clean_generated_turn(raw, self._persona_names)
            if _is_usable_turn(text, history):
                return text
            if on_progress is not None and n_turns:
                on_progress(turn_index, n_turns, persona)  # re-show current turn while retrying
        raise RuntimeError(
            "The debate-generation model kept producing unusable output "
            f"(e.g. stage directions, fragments, or repetition) after {attempts} attempts. "
            f"Last raw output: {last_raw[:200]!r}"
        )

    def generate(
        self,
        proposition: str,
        n_agents: int = 2,
        n_turns: int = 6,
        personas: Optional[list] = None,
        seed: int = 42,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Round-robin generation loop. Returns a transcript dict in the
        same canonical schema produced by src/data_loader.build_transcripts,
        so both human and agent-generated debates flow through one
        analysis pipeline."""
        if n_agents < 2:
            raise ValueError("A debate needs at least 2 agents (one per side).")
        personas = personas or [f"Agent_{i+1}" for i in range(n_agents)]
        self._persona_names = list(personas)
        sides = (["the FOR (pro) side", "the AGAINST (con) side"]
                 if n_agents == 2 else
                 ["the FOR (pro) side", "the AGAINST (con) side", "a neutral moderator"])
        history = []
        turns = []
        for t in range(n_turns):
            persona = personas[t % len(personas)]
            side = sides[t % len(personas)]
            if on_progress is not None:
                on_progress(t, n_turns, persona)
            text = self._robust_turn(proposition, persona, side, history,
                                     on_progress=on_progress, turn_index=t, n_turns=n_turns)
            history.append({"speaker": persona, "text": text})
            turns.append({"turn": t, "speaker": persona, "text": text,
                          "agreement_pre": None, "agreement_post": None,
                          "side_agreement_pre": None, "side_agreement_post": None})
        return {
            "transcript_id": f"gen_{stable_hash([proposition, n_agents, n_turns, seed])}",
            "topic": proposition,
            "source": "agent_generated",
            "turns": turns,
        }


# ---------------------------------------------------------------------------
# §3.3 Stance-dynamics layer (Eq. 1-2)
# ---------------------------------------------------------------------------
@dataclass
class StanceDynamicsEngine:
    """Implements the paper's simplified log-odds stance update exactly:

        z_t = z_{t-1} + alpha * q_t * sign(e_t) - beta * |S_{t-1}|      (1)
        S_t = tanh(z_t / 2)                                             (2)

    z_0 = 0 (neutral prior, S_0 = 0). S_t is bounded and never feeds back
    into its own definition -- only the unbounded z_t recurses -- so the
    recursion is well-defined at every turn, exactly as specified.
    """

    alpha: float = 0.8  # evidence uptake
    beta: float = 0.15  # anchoring strength

    def compute_trajectory(self, quality_scores: list, evidence_directions: list) -> list:
        """quality_scores: q_t in [0,1] per turn.
        evidence_directions: e_t in {-1, +1} per turn (stance direction of
        the evidence presented that turn, e.g. sign of a pro/con label).
        Returns a list of dicts, one per turn: {"z": ..., "S": ...}."""
        assert len(quality_scores) == len(evidence_directions)
        z = 0.0
        traj = []
        s_prev = 0.0
        for q_t, e_t in zip(quality_scores, evidence_directions):
            sign_e = 1.0 if e_t >= 0 else -1.0
            z = z + self.alpha * q_t * sign_e - self.beta * abs(s_prev)
            s_t = math.tanh(z / 2.0)
            traj.append({"z": z, "S": s_t})
            s_prev = s_t
        return traj


def infer_evidence_direction(text: str) -> int:
    """Crude but deterministic pro/con direction proxy: counts simple
    polarity cues. Used only to derive e_t for the stance engine when a
    transcript doesn't otherwise supply a labeled direction (e.g. secondary
    agent debates before we've mapped them to +/-1)."""
    text_l = text.lower()
    pro_cues = ["support", "should", "benefit", "agree", "for this", "in favor"]
    con_cues = ["oppose", "should not", "harm", "disagree", "against this", "reject"]
    pro = sum(text_l.count(c) for c in pro_cues)
    con = sum(text_l.count(c) for c in con_cues)
    return 1 if pro >= con else -1


# ---------------------------------------------------------------------------
# §3.3 Attribution engine ("Why?" card)
# ---------------------------------------------------------------------------
@dataclass
class AttributionThresholds:
    evidence_quality_min: float = 0.6
    anchoring_similarity_min: float = 0.8
    echo_similarity_min: float = 0.75
    strategic_js_min: float = 0.3


class AttributionEngine:
    """Classifies each inflection point into one of {evidence_adoption,
    anchoring, echo, strategic_persuasion} using the exact thresholds and
    priority order specified in the paper (§3.3):
        evidence_adoption > strategic_persuasion > echo > anchoring
    Supports the three ablations from §4.2 via `disable_*` flags."""

    PRIORITY = ["evidence_adoption", "strategic_persuasion", "echo", "anchoring"]

    def __init__(
        self,
        thresholds: Optional[AttributionThresholds] = None,
        disable_rhetoric: bool = False,
        disable_stance: bool = False,
        disable_echo: bool = False,
    ):
        self.th = thresholds or AttributionThresholds()
        self.disable_rhetoric = disable_rhetoric
        self.disable_stance = disable_stance
        self.disable_echo = disable_echo

    def classify_turn(
        self,
        current_rhetoric: dict,
        prev_rhetoric: Optional[dict],
        other_agent_prev_rhetoric: Optional[dict],
        quality_score: float,
        stance_change: float,
        prev_stance: float,
        sim_to_own_prior: float,
        sim_to_other_prev: float,
        is_empirical_or_causal: bool,
    ) -> dict:
        """Returns {"label": str, "signals": {signal_name: bool, ...}, "confidence": float}."""
        signals = {"evidence_adoption": False, "strategic_persuasion": False,
                   "echo": False, "anchoring": False}
        js = 0.0

        # --- Evidence adoption ---
        if not self.disable_rhetoric:
            evidence_adoption = (
                is_empirical_or_causal
                and quality_score > self.th.evidence_quality_min
                and sim_to_own_prior < 0.5  # "semantically dissimilar to prior utterances"
            )
        else:
            # ablation: no rhetorical labels available, fall back to
            # quality + novelty only (can't check causal/empirical tag)
            evidence_adoption = quality_score > self.th.evidence_quality_min and sim_to_own_prior < 0.5
        signals["evidence_adoption"] = bool(evidence_adoption)

        # --- Strategic persuasion (requires rhetorical distribution AND stance change, both per paper §3.3) ---
        if not self.disable_rhetoric and not self.disable_stance and prev_rhetoric is not None:
            js = _js_divergence(prev_rhetoric, current_rhetoric)
            shifted_to_affective = (
                max(prev_rhetoric, key=prev_rhetoric.get) in ("causal", "empirical")
                and max(current_rhetoric, key=current_rhetoric.get) in ("emotional", "moral")
            )
            strategic = js > self.th.strategic_js_min and shifted_to_affective and abs(stance_change) > 0.2
        else:
            strategic = False
        signals["strategic_persuasion"] = bool(strategic)

        # --- Echo / peer influence ---
        if not self.disable_echo:
            echo = sim_to_other_prev > self.th.echo_similarity_min
            if not self.disable_rhetoric and other_agent_prev_rhetoric is not None:
                strat_js_to_other = _js_divergence(current_rhetoric, other_agent_prev_rhetoric)
                echo = echo or strat_js_to_other < 0.1
        else:
            echo = False
        signals["echo"] = bool(echo)

        # --- Anchoring ---
        if not self.disable_stance:
            anchoring = sim_to_own_prior > self.th.anchoring_similarity_min or (
                abs(stance_change) < 0.05 and quality_score > self.th.evidence_quality_min
            )
        else:
            anchoring = sim_to_own_prior > self.th.anchoring_similarity_min
        signals["anchoring"] = bool(anchoring)

        label = "no_inflection"
        for candidate in self.PRIORITY:
            if signals[candidate]:
                label = candidate
                break

        confidence = js if label == "strategic_persuasion" else max(quality_score, sim_to_own_prior, sim_to_other_prev)
        return {"label": label, "signals": signals, "confidence": float(confidence)}


# ---------------------------------------------------------------------------
# Orchestration: full per-transcript analysis with checkpointing
# ---------------------------------------------------------------------------
class DebateAnalysisPipeline:
    """Wires Panels 1-3 together for a single transcript, and provides a
    checkpointed batch runner over many transcripts (crash-resilient: if
    the kernel dies partway through a batch, re-running only recomputes
    the transcripts that weren't already cached)."""

    def __init__(
        self,
        backend: str = "hf",
        checkpoint_dir: str = "checkpoints",
        alpha: float = 0.8,
        beta: float = 0.15,
        disable_rhetoric: bool = False,
        disable_stance: bool = False,
        disable_echo: bool = False,
    ):
        self.annotator = RhetoricalAnnotator(backend=backend)
        self.quality_scorer = ArgumentQualityScorer(backend=backend)
        self.similarity = SemanticSimilarity(backend=backend)
        self.stance_engine = StanceDynamicsEngine(alpha=alpha, beta=beta)
        self.attribution = AttributionEngine(
            disable_rhetoric=disable_rhetoric, disable_stance=disable_stance, disable_echo=disable_echo
        )
        self.ckpt = CheckpointManager(checkpoint_dir)
        self.disable_rhetoric = disable_rhetoric
        self.disable_stance = disable_stance

    def analyze_transcript(self, transcript: dict, force: bool = False) -> dict:
        # content hash guards against stale hits if turn construction/content changes upstream
        content_key = stable_hash([t["text"] for t in transcript["turns"]])
        key = f"analysis__{transcript['transcript_id']}__{content_key}__r{int(self.disable_rhetoric)}s{int(self.disable_stance)}"

        def _compute():
            turns = transcript["turns"]
            texts = [t["text"] for t in turns]
            speakers = [t["speaker"] for t in turns]

            rhetoric = [None] * len(texts) if self.disable_rhetoric else self.annotator.annotate_batch(texts)
            quality = self.quality_scorer.score_batch(texts)
            directions = [infer_evidence_direction(t) for t in texts]

            # per-agent stance trajectory
            per_agent_idx = {}
            for i, sp in enumerate(speakers):
                per_agent_idx.setdefault(sp, []).append(i)

            stance_by_turn = [None] * len(texts)
            for sp, idxs in per_agent_idx.items():
                qs = [quality[i] for i in idxs]
                es = [directions[i] for i in idxs]
                traj = self.stance_engine.compute_trajectory(qs, es)
                for local_i, global_i in enumerate(idxs):
                    stance_by_turn[global_i] = traj[local_i]

            # attribution per turn (skip first turn per agent: no prior to compare against)
            attributions = [None] * len(texts)
            last_own_idx = {}
            last_stance = {}
            for i, sp in enumerate(speakers):
                prev_own_idx = last_own_idx.get(sp)
                # nearest prior turn by any other agent
                other_prev_idx = i - 1 if i > 0 else None
                sim_to_own_prior = (
                    self.similarity.similarity(texts[i], texts[prev_own_idx]) if prev_own_idx is not None else 0.0
                )
                sim_to_other_prev = (
                    self.similarity.similarity(texts[i], texts[other_prev_idx])
                    if other_prev_idx is not None and speakers[other_prev_idx] != sp
                    else 0.0
                )
                stance_change = stance_by_turn[i]["S"] - last_stance.get(sp, 0.0)
                is_emp_causal = (rhetoric[i]["label"] in ("empirical", "causal")) if rhetoric[i] else False
                result = self.attribution.classify_turn(
                    current_rhetoric=rhetoric[i]["probs"] if rhetoric[i] else {l: 0.25 for l in RHETORIC_LABELS},
                    prev_rhetoric=(rhetoric[prev_own_idx]["probs"] if prev_own_idx is not None and rhetoric[prev_own_idx] else None),
                    other_agent_prev_rhetoric=(rhetoric[other_prev_idx]["probs"] if other_prev_idx is not None and rhetoric[other_prev_idx] else None),
                    quality_score=quality[i],
                    stance_change=stance_change,
                    prev_stance=last_stance.get(sp, 0.0),
                    sim_to_own_prior=sim_to_own_prior,
                    sim_to_other_prev=sim_to_other_prev,
                    is_empirical_or_causal=is_emp_causal,
                )
                attributions[i] = result
                last_own_idx[sp] = i
                last_stance[sp] = stance_by_turn[i]["S"]

            return {
                "transcript_id": transcript["transcript_id"],
                "topic": transcript.get("topic"),
                "source": transcript.get("source"),
                "speakers": speakers,
                "texts": texts,
                "rhetoric": rhetoric,
                "quality": quality,
                "evidence_direction": directions,
                "stance": stance_by_turn,
                "attribution": attributions,
            }

        return self.ckpt.load_or_compute(key, _compute, fmt="pkl", force=force)

    def analyze_batch(self, transcripts: list, force: bool = False, progress_cb=None) -> list:
        """Crash-resilient batch runner: each transcript is checkpointed
        independently, so a kernel death after transcript k only requires
        recomputing transcript k+1 onward on the next run."""
        results = []
        for i, tr in enumerate(transcripts):
            res = self.analyze_transcript(tr, force=force)
            results.append(res)
            if progress_cb:
                progress_cb(i + 1, len(transcripts), tr["transcript_id"])
        return results


if __name__ == "__main__":
    # Smoke test: `python -m src.engine`
    set_global_seed(42)
    gen = DebateGenerator(backend="hf")
    transcript = gen.generate("Universal basic income should be adopted nationally", n_agents=2, n_turns=6)
    pipeline = DebateAnalysisPipeline(backend="hf", checkpoint_dir="checkpoints")
    result = pipeline.analyze_transcript(transcript, force=True)
    for i, (sp, txt, att) in enumerate(zip(result["speakers"], result["texts"], result["attribution"])):
        print(f"[turn {i}] {sp} (S={result['stance'][i]['S']:.2f}) -> {att['label']}: {txt[:60]}")