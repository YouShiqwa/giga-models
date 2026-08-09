#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "${script_dir}")"
repo_root="$(cd -- "${project_dir}/../../.." && pwd)"

export PYTHONPATH="${repo_root}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_dir}"
exec python scripts/train.py --config configs.pi05_robocasa_set_up_cutting_station.config
