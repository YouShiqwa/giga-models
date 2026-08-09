#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "${script_dir}")"
giga_root="$(cd -- "${project_dir}/../../.." && pwd)"
mobile_pi_root="$(cd -- "${project_dir}/../../../.." && pwd)"
openpi_root="${mobile_pi_root}/openpi"

train_config="configs.pi05_robocasa_set_up_cutting_station_dual_16query_action_random.config"
checkpoint="${project_dir}/experiments/vla/pi05/robocasa_set_up_cutting_station_dual_16query_action_random/models/checkpoint_epoch_10_step_50000/model_ema"
eval_launcher="${openpi_root}/examples/robocasa/eval_giga_dual_query_action_robocasa.sh"
eval_config="${openpi_root}/examples/robocasa/configs/giga_pi05_dual_16query_action_random.yaml"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run training and evaluation" >&2
  exit 1
fi
if [[ ! -f "${project_dir}/configs/pi05_robocasa_set_up_cutting_station_dual_16query_action_random.py" ]]; then
  echo "Training config does not exist" >&2
  exit 1
fi
if [[ ! -f "${eval_launcher}" || ! -f "${eval_config}" ]]; then
  echo "Evaluation launcher or config does not exist" >&2
  exit 1
fi

conda_base="$(conda info --base)"
source "${conda_base}/etc/profile.d/conda.sh"
conda activate giga_pi0

export PYTHONPATH="${giga_root}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

log "Starting random-action-head dual_16query_action training"
cd "${project_dir}"
python scripts/train.py --config "${train_config}"

if [[ ! -d "${checkpoint}" ]]; then
  echo "Training returned successfully, but the final EMA checkpoint is missing: ${checkpoint}" >&2
  exit 1
fi

log "Training completed; starting aligned RoboCasa evaluation"
cd "${openpi_root}"
exec bash "${eval_launcher}" "${eval_config}"
