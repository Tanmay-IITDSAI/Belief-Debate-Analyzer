# Worked case study: the three `strategic_persuasion` attributions

Source: `strategic_persuasion_contexts.json` (dumped verbatim from the released
hash-suffixed `analysis__*` checkpoints by `notebooks/experiments/07_attribution_case_study.ipynb`).
Counts and confidences below are read directly from the checkpoints; the per-case readings are one
analyst's provisional judgments made against the four-question scaffold — they are arguments from
the transcript text, not measurements, and are meant to be checked (and if necessary revised) by the
authors before being cited.

Cross-case fact worth noting: the system's (uncalibrated) confidence ordering is
303.0 (0.610) > 125.0 (0.459) > 176.0 (0.311). An informal human readability ordering agrees that
303.0 is the most defensible case, but ranks 176.0 above 125.0 — i.e., the confidence score does not
fully track an informal human ordering. This is consistent with the paper's treatment of the
confidence display as an uncalibrated heuristic (§5, H6) and is exactly why the card surfaces
signals + alternatives rather than a verdict.

**Scaffold status:** `verification_scaffold.json` is filled with these same provisional readings
(each record marked `"provisional": true`), kept in sync with the prose below. The authors should
still verify each case against the full transcript before citing.

---

## Case 1 — transcript 303.0, death penalty, turn 2 (con), confidence 0.610 — most defensible

**Card:** `strategic_persuasion` (only this signal fired; evidence adoption, echo, anchoring all
false). Rhetorical shift: previous turn predominantly empirical (0.36), flagged turn pure moral (1.0).

**1. Rhetorical shift (quoted).**
- Prev (pro, empirical framing): *"I believe that capital punishment is a fair assessment/ruling
  when it fits the criminal's crime. For example, if someone murders 10+ people, I don't think it's
  acceptable to give that criminal a life that they get to live…"*
- Flagged (con, pure moral): *"Is it fair in all circumstances though? What if, which is not
  uncommon, later evidence suggests the criminal was indeed innocent but no longer alive, can that
  be true justice? A life in prison demands those which are in fact guilty to dwell on their crimes…"*

**2. Stance movement.** The flagged speaker's own trajectory strengthens monotonically after the
moral reframe: 0.291 → 0.521 → 0.685 (turns 0, 2, 4).

**3. Alternative explanations considered.** Evidence adoption: no new data or mechanism is
introduced — the turn poses moral hypotheticals. Echo: wording diverges from the pro turn rather
than mirroring it. Anchoring: the con is not restating an earlier position; the moral-risk-of-error
frame is new in this debate.

**4. Provisional verdict: defensible (lean endorse).** The turn is a textbook rhetorical reframe —
the opponent's empirical case is met with a values argument, not evidence — and the opponent's next
turn visibly accommodates it: *"While I agree with the sentiment of someone having to live with
their actions…"*. A human reader can verify each of these steps from the dumped context.

**Full-text note (complete context reviewed).** The arc is even stronger than the snippet
suggested: the pro's *final* turn (5) concedes the con's uncertainty standard outright — *"I don't
think the death penalty should be used in circumstances such as this scenario, I think it needs to
be proven with the upmost certainty that the criminal is guilty with strong evidence before we
take their life."* — while the system labels that same turn `echo`. A human reads concession; the
signal fires on mirrored structure. That gap is itself demonstrative: the card is a candidate
explanation to be checked against the transcript, not a verdict.

## Case 2 — transcript 176.0, surveillance, turn 2 (con), confidence 0.311 — contestable, informative

**Card:** `strategic_persuasion` (only signal fired). Shift: causal-heavy prev turn → emotional/moral
(0.5/0.5) flagged turn.

**1. Rhetorical shift (quoted).**
- Prev (pro, causal): *"National security is of utmost importance. With the rise of terrorist
  attacks, we need to be sure we can be kept safe…"*
- Flagged (con, emotional/moral): *"National security is extremely important. However, for citizens
  like myself who have a lifetime history of no terrorist involvement, history of a criminal record,
  we should be given our right to privacy."*

**2. Stance movement.** Con: 0.291 → 0.521 → 0.564 — strengthens, but with diminishing increments.

**3. Alternative explanations considered.** The flagged turn *partially concedes the opponent's
premise* ("National security is extremely important. However, …"), so accommodation/accord is a
plausible alternative reading that the card does not exclude. Echo: partial mirroring of the
opponent's opening phrase, but below the signal threshold. Evidence adoption: none.

**4. Provisional verdict: ambiguous.** The affective pivot is real, but the concession framing makes
accommodation equally defensible. This case is valuable precisely because it shows what the system
does *not* establish: the card reports that the strategic-persuasion signal fired and that no
alternative signal did — it does not, and should not, adjudicate between strategy and accommodation.

**Full-text note.** The complete turn also challenges the opponent's evidence (*"where is the
proof of that?"*) and inverts the parent analogy (*"As a parent, I would not listen in on my
child's private conversations. The government is certainly not my parent."*), which makes the
strategic reading more substantive than the snippet alone suggested — but the opening concession
stands, so the verdict remains ambiguous.

## Case 3 — transcript 125.0, internet censorship, turn 3 (pro), confidence 0.459 — weakest

**Card:** `strategic_persuasion` (only signal fired). Shift: uniform baseline (0.25 on all four
rhetoric dimensions) → pure moral (1.0).

**1. Rhetorical shift (quoted).**
- Prev (con, uniform baseline): *"The US government has no ownership over online domains. Posting
  information to the World of Tanks forum is akin to having a conversation in the middle of the
  ocean…"*
- Flagged (pro, pure moral): *"The government has ownership of certain types of information
  regardless of physical location. They may not have the right to censor anything they want, but
  they are allowed to censor pieces of information that are considered government property."*

**2. Stance movement.** Pro: 0.374 (flagged turn) → 0.434 (turn 5) — small; and the pro's turn 5 is
itself labeled `echo`, further weakening a strategic reading of the subsequent movement.

**3. Alternative explanations considered.** The shift magnitude is computed against a *maximally
uninformative* previous-turn baseline (equal mass on all four dimensions), so the 0→1 moral jump
overstates how deliberate the pivot looks to a human reader. Echo/anchoring/evidence: all below
their thresholds.

**4. Provisional verdict: ambiguous (lean refute-as-strategic).** A moral counter-argument it is;
evidence that it was *strategic* persuasion is thin. Reported honestly, this is the case where the
signal most likely over-fires — and it is useful to show reviewers that the workbench's artifacts
let a reader reach exactly that conclusion.

**Full-text note.** The flagged turn is a direct ownership/property-rights rebuttal (*"The
government has ownership of certain types of information regardless of physical location"*); a
human reader might categorize it as legal/causal rather than pure moral (1.0), so the computed
shift magnitude partly rides on the rhetoric annotator's categorization — a second, independent
reason to distrust the strength of this flag.

---

## What this case study establishes (and does not)

- **Establishes:** every flagged attribution is inspectable down to the transcript text, the
  per-turn rhetoric distribution, and the signal vector; a reader can argue for or against each
  label without re-running the system.
- **Does not establish:** that the strategic-persuasion signal is accurate in general (n = 3, no
  statistical claim intended), or that confidence is calibrated (H6 already reports it is not).
- **Framing for the paper:** consistent with the evaluation decision that *ambiguous is a legitimate
  system output* — the workbench makes candidate explanations and their evidence inspectable; it
  does not certify causes.
