#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="/home/anurag24092/Projects/L-Moment-Lifelong-Learning"
python_bin="/home/anurag24092/envs/tfcl-btp/bin/python"
output_root="results/stl10_baseline_independent_v2"
log_dir="$output_root/logs"
log_file="$log_dir/gpu_idle_launcher.log"

cd "$repo_root"
mkdir -p "$log_dir"

# Keep at least half the V100's VRAM free and avoid launching during active
# compute. Stopped processes can retain CUDA allocations while using no GPU.
idle_samples=0
while (( idle_samples < 3 )); do
    sample="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r used_mib utilization <<< "$sample"
    used_mib="${used_mib// /}"
    utilization="${utilization// /}"
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s memory_mib=%s utilization_pct=%s\n' \
        "$timestamp" "$used_mib" "$utilization" | tee -a "$log_file"
    if (( used_mib < 16384 && utilization < 5 )); then
        ((idle_samples += 1))
    else
        idle_samples=0
    fi
    if (( idle_samples < 3 )); then
        sleep 60
    fi
done

printf '%s Starting independent baseline search for all 26 STL-10 tracks.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_file"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m lmoment_lifelong.cifar10_tuning \
    --dataset stl10 \
    --batch-size 64 \
    --train-per-class 400 \
    --screen-train-per-class 100 \
    --val-per-class 100 \
    --test-per-class 800 \
    --output-root "$output_root" \
    --feature-cache-root results/stl10_dinov2_cache_v1 \
    --data-root datasets/stl10_binary \
    --device cuda \
    --extraction-microbatch 4 \
    --seed 0 \
    --variant baseline \
    --max-rounds 30 \
    --no-download 2>&1 | tee -a "$log_file"
