# Annotator Instructions — Blind Explanation-Quality Study

**Study:** Why Did You Change Your Mind? — evaluating the Belief-Tracking Debate Analyzer workbench
**Time commitment:** ~45–75 minutes total (8 judgments, roughly 5–10 minutes each)
**Materials you receive:** two personal packets — `P*_A` (4 transcripts, transcript only) and `P*_B` (4 transcripts, transcript + workbench panel)

---

## 1. What this study is about

We are testing whether an interactive analysis workbench helps researchers understand *why* a debater's position changed during a debate. You will read short 6-turn debates and answer a fixed set of questions for each. Some packets include extra analysis produced by our system; some contain only the raw transcript. **Both kinds of packets are equally important to us** — a null result is as informative as a positive one, so please answer with your honest best judgment in both.

You are **not** being tested. There are no "correct answers" we expect you to know in advance. We care about what *you* can extract from the material in front of you.

## 2. Before you start

1. Find a quiet place; budget one uninterrupted block of 45–75 minutes.
2. You will need a text editor or spreadsheet program for the CSV (or a pen if you use the printed Markdown version).
3. **Do not open both packets until you have finished the first one.** Complete your two packets in separate sittings if possible (a short break of at least 30 minutes between them is ideal).
4. Have the demo site open **only if your condition-B instructions tell you to** (see §5).

## 3. The golden rules

- **Work completely independently.** Do not discuss the transcripts, your answers, or the system with anyone — including other annotators — until all packets have been returned and you are told the study is closed.
- **Do not look up the project, its code, its demo site, or the DebateGPT dataset** beyond what your packet explicitly instructs.
- **Do not skip items.** If you are unsure, write your best guess plus the word "unsure". "Unsure" answers are useful data; empty cells are not.
- **Answer from the material in the packet only** — not from what you imagine the system probably computed, and not from prior beliefs about the debate topic.

## 4. Packet A — transcript only

Each item shows a debate topic and six turns (two speakers, three rounds). For each transcript, fill in:

| Field | Question | How to answer |
|---|---|---|
| `q1_change` | What changed at the inflection point? | 1–2 sentences in your own words. If you see no clear inflection point, write "no clear change" and say why. |
| `q2_mechanisms` | Which candidate mechanism(s) are plausible? | Pick any of: `evidence_adoption` (convinced by a substantive argument), `anchoring` (stuck to prior position despite counter-evidence), `echo` (repeating/mirroring the opponent's phrasing or points), `strategic_persuasion` (shift following a *rhetorical* move rather than substance), `no_clear_change`. Separate multiple choices with semicolons, e.g. `echo;anchoring`. Choose as many or as few as you genuinely find plausible. |
| `q3_evidence` | What textual evidence supports that interpretation? | Quote a short phrase from the transcript **verbatim** (copy-paste), or paraphrase and point to the turn number. |
| `q4_confidence_1to5` | How confident are you? | 1 = not at all confident … 5 = very confident. |
| `q5_workbench_useful_1to5_or_NA` | Was the workbench useful? | Write `NA` in packet A. You did not see the workbench. |

## 5. Packet B — transcript + workbench panel

Your packet adds a **WORKBENCH PANEL** block above each transcript. It shows, produced by our system:

- the **stance trajectory** (a score from −1 to +1 per turn; higher = more supportive of the proposition),
- per-turn **rhetoric labels** (causal / empirical / emotional / moral),
- the **triggered signals** for each turn,
- the **"Why?" card**: the system's single best-guess mechanism with a confidence value (0–1),
- the **alternative**: the system's runner-up mechanism.

Rules for packet B:

1. Read the workbench panel **first**, then the transcript.
2. Treat the panel as *another researcher's hypotheses*, not as ground truth. If you disagree with the "Why?" card after reading the transcript, **say so** — in `q2_mechanisms` pick what *you* find plausible, even when it differs from the card, and note the disagreement in `q1_change` or `q3_evidence`.
3. Everything from §4 applies unchanged, **except** `q5`: here, answer genuinely on the 1–5 scale — how useful was the panel for reaching your judgment?
4. If your packet's cover page says to use the **live workbench** instead of the embedded panel: open the site, find each transcript by its topic, explore freely (click turns, inspect the "Why?" card), then answer. Otherwise do **not** open the site.

## 6. How to fill the CSV (recommended)

1. Open your packet CSV (e.g. `P1_B.csv`) in Excel/LibreOffice/Google Sheets or a text editor.
2. Fill **only** the five answer columns; leave the metadata columns untouched.
3. For `q2_mechanisms`, use the exact option strings from §4, semicolon-separated, no spaces needed.
4. Save the file **in place, same name, same format** (CSV).
5. Filename reminder: `_A` = transcript only, `_B` = with workbench panel.

If you were given the Markdown version instead: write your answers on the `Answer:` lines under each question and return the file.

## 7. Returning your packets

- Return **both** CSVs (A and B) to the sender by the agreed deadline.
- Before sending, sanity-check: 8 items answered across the two files, no empty answer cells, no extra columns added.
- After returning, please **do not discuss the study** until the sender confirms it is closed — late-returning annotators must face the same conditions you did.

## 8. Questions, problems, withdrawal

- If a transcript seems broken (missing turns, garbled text), note it in `q1_change` and move on — do not guess what it "should" say.
- If anything is unclear *before* you start, ask the sender now (questions are fine before the study begins, not during).
- Participation is voluntary; you may stop at any point and return whatever you completed. Partial data is usable.

Thank you — your judgments are a real part of the published evaluation, reported exactly as measured.
