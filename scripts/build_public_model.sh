#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
rugby-value fetch-data
rugby-value fit data/raw/phase_2018-19.csv \
  --output models/premiership-2018-19 \
  --bootstrap "${RPV_BOOTSTRAP:-500}"
