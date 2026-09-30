#!/bin/bash
#SBATCH --job-name=so101-smolvla
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100-preempt
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/smolvla-%j.out
#SBATCH --error=outputs/logs/smolvla-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SVLA_TASK:?Set pick_lift, pick_place, or battery_insertion}"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
PY
extra=()
if [[ -n "${SVLA_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$SVLA_RESUME_FROM"); fi
trap 'if command -v my-job-stats >/dev/null 2>&1; then my-job-stats -a -n -s || true; fi' EXIT
.venv/bin/python baselines/smolvla/run_smolvla.py --task "$SVLA_TASK" \
  --output "${SVLA_RUN_OUTPUT:-$PWD/outputs/smolvla/${SVLA_TASK}_${SLURM_JOB_ID}}" \
  --steps 20000 "${extra[@]}" --execute
