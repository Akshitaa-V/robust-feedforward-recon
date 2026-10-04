#!/usr/bin/env bash
# Trains every variant for every seed, then evaluates and builds the report.
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
export PYTHONPATH=src
for seed in 0 1; do
  for v in single_clean multi_clean multi_aug robust; do
    if [ ! -f "runs/${v}_seed${seed}/model.pt" ]; then
      python -m robustrecon.train --config "$CONFIG" --variant "$v" --seed "$seed"
    fi
  done
done
python -m robustrecon.evaluate --config "$CONFIG"
python -m robustrecon.report --config "$CONFIG"
