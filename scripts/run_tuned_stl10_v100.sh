#!/usr/bin/env bash
set -euo pipefail

log_dir=results/stl10_hypothesis26_tuned/logs
mkdir -p "$log_dir"
log_file="$log_dir/tuner_$(date -u +%Y%m%dT%H%M%SZ).log"

exec python -m lmoment_lifelong.cifar10_tuning \
  --dataset stl10 \
  --train-per-class 400 \
  --val-per-class 100 \
  --test-per-class 800 \
  --batch-size 64 \
  --device cuda \
  --extraction-microbatch 4 \
  --download \
  --max-rounds 20 2>&1 | tee -a "$log_file"
