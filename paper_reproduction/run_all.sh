#!/usr/bin/env bash
# Recomputes every derived statistic and figure from the result files in data/.
# Requires: python>=3.10, numpy, pandas, scipy, matplotlib (see ../requirements.txt). ~2-4 min on a laptop.
set -euo pipefail
PY="${PYTHON:-python3}"   # override with PYTHON=python if needed
cd "$(dirname "$0")"
mkdir -p outputs/analysis outputs/figures
"$PY" scripts/facts.py data/export outputs/analysis/facts.json
"$PY" scripts/facts_ext.py data outputs/analysis/facts_ext.json
"$PY" scripts/tost.py data/export/grid_runs.csv outputs/analysis/tost_fedadapt.csv
"$PY" scripts/new_tables.py > outputs/analysis/new_tables_log.txt
"$PY" scripts/make_figures.py data/export outputs/figures
"$PY" scripts/make_new_figures.py outputs/figures
echo "done: figures in outputs/figures, statistics in outputs/analysis"
