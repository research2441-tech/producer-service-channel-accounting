#!/usr/bin/env bash
# Reproduces every estimate, table input and figure from the assembled panel.
# All inputs and outputs live in this single folder.
set -e
cd "$(dirname "$0")"
python analysis.py . .          # writes results.json and the CSV inputs of the figures
NREP=500 NBOOT_P=199 python placebo.py .   # placebo size experiment -> placebo.json
python figures.py . .           # writes fig1..fig9 PNG files
