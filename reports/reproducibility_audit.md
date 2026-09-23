# Reproducibility Audit

Root: `/home/pakdd/github repo/EACL DEMO 2027/belief_debate_analyzer`

**Authentic-results eligible:** `False`

## required_directories — PASS
Required directories exist.

## required_files — PASS
Core files exist.

## primary_debategpt — PASS
DebateGPT files and four agreement fields are present.
- data/raw/debategpt/debategpt.csv

## human_study_inputs — FAIL
Human annotation/user-study inputs are missing or invalid.

## processed_evaluation_inputs — FAIL
Processed DebateGPT evaluation inputs are missing.
- data/processed/debategpt_instances.csv
- data/processed/cw_por.csv
- data/processed/convergence.csv

## proxy_scan — PASS
No known proxy-generation markers found.

## python_syntax — PASS
Python source parses successfully.

## notebook_evaluation_integrity — PASS
Notebook evaluation is clean.

## real_results_gate — BLOCKED
Authentic metrics remain blocked until all observed artifacts pass.
