#!/usr/bin/env bash
# End to end on one dataset: protect the training set, then fine-tune on clean
# and on protected data and compare the test accuracy.
#
#   bash scripts/run.sh arc                 # configs/arc.yaml, seed 42
#   bash scripts/run.sh arc --run.seed=43   # extra arguments go to every stage
#
# Results: outputs/<dataset>/seed<seed>/results_{clean,protected}.json
set -euo pipefail
cd "$(dirname "$0")/.."
ds=${1:?usage: bash scripts/run.sh DATASET [--key=value ...]}
shift
cfg=configs/$ds.yaml
python -m textcloak.protect "$cfg" "$@"
python -m textcloak.finetune "$cfg" --data clean "$@"
python -m textcloak.finetune "$cfg" --data protected "$@"
python scripts/summarize.py "$cfg" "$@"
