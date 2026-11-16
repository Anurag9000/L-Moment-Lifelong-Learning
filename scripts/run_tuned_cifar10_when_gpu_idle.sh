#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="/home/anurag24092/Projects/L-Moment-Lifelong-Learning"
python_bin="/home/anurag24092/envs/tfcl-btp/bin/python"
cd "$repo_root"
mkdir -p results/cifar10_hypothesis26_tuned/logs
log_file="results/cifar10_hypothesis26_tuned/logs/gpu_idle_launcher.log"

idle_samples=0
while (( idle_samples < 3 )); do
    sample="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r used_mib utilization <<< "$sample"
    used_mib="${used_mib// /}"
    utilization="${utilization// /}"
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s memory_mib=%s utilization_pct=%s\n' \
        "$timestamp" "$used_mib" "$utilization" | tee -a "$log_file"
    if (( used_mib < 2048 && utilization < 5 )); then
        ((idle_samples += 1))
    else
        idle_samples=0
    fi
    if (( idle_samples < 3 )); then
        sleep 60
    fi
done

printf '%s Starting complete 26-track validation tuning campaign.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_file"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m lmoment_lifelong.cifar10_tuning \
    --output-root results/cifar10_hypothesis26_tuned \
    --feature-cache-root results/cifar10_dinov2_cache \
    --data-root datasets --device cuda --extraction-microbatch 16 \
    --max-rounds 4 2>&1 | tee -a "$log_file"
