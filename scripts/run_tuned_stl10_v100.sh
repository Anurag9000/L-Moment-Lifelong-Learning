#!/usr/bin/env bash
set -euo pipefail

log_dir=results/stl10_hypothesis26_multifidelity_v2/logs
mkdir -p "$log_dir"
log_file="$log_dir/tuner_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/envs/tfcl-btp/bin/python" ]]; then
    PYTHON="$HOME/envs/tfcl-btp/bin/python"
  else
    PYTHON=python3
  fi
fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m lmoment_lifelong.cifar10_tuning \
  --dataset stl10 \
  --train-per-class 400 \
  --screen-train-per-class 100 \
  --val-per-class 100 \
  --test-per-class 800 \
  --batch-size 64 \
  --device cuda \
  --extraction-microbatch 4 \
  --download \
  --output-root results/stl10_hypothesis26_multifidelity_v2 \
  --max-rounds 20 2>&1 | tee -a "$log_file"
