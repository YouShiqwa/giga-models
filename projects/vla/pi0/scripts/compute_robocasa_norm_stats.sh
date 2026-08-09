#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "${script_dir}")"
repo_root="$(cd -- "${project_dir}/../../.." && pwd)"
data_path="/vepfs-cnbje63de6fae220/chengy/code/mobile_pi/datasets/v1.0/target/composite/SetUpCuttingStation/20250817/lerobot"
output_path="${data_path}/meta/giga_pi05_robocasa_norm_stats.json"

export PYTHONPATH="${repo_root}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_dir}"
exec python scripts/compute_norm_stats.py \
    --data-paths "${data_path}" \
    --output-path "${output_path}" \
    --sample-rate 1.0 \
    --action-chunk 50 \
    --action-dim 32 \
    --dataset-type robocasa \
    --num-workers 16
