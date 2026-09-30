#!/bin/bash
#SBATCH --job-name=so101-diffusion
#SBATCH --account=punim2341
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=diffusion-%j.out
#SBATCH --error=diffusion-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch; do not run training on a login node}"
: "${BENCHMARK_SCRIPTS:?Set absolute path to so_arm101/baselines/diffusion}"
: "${DP_TASK:?Set DP_TASK to pick_lift or pick_place}"
: "${DP_OUTPUT_BASE:?Set absolute path to persistent project storage}"
case "$DP_TASK" in pick_lift|pick_place) ;; *) echo 'Invalid DP_TASK' >&2; exit 2;; esac
case "${DP_SMOKE:-0}" in 0|1) ;; *) echo 'DP_SMOKE must be 0 or 1' >&2; exit 2;; esac
case "$BENCHMARK_SCRIPTS:$DP_OUTPUT_BASE" in /*:/*) ;; *) echo 'Use absolute script/output paths' >&2; exit 2;; esac

# Either initialize a dedicated Conda environment inside the job, or use a
# verified absolute Python executable. Do not combine both selectors.
if [[ -n "${LEROBOT_CONDA_ENV:-}" ]]; then
  [[ -z "${LEROBOT_PYTHON:-}" ]] || { echo 'Select CONDA_ENV or PYTHON, not both' >&2; exit 2; }
  module purge
  module load "${LEROBOT_CONDA_MODULE:-Anaconda3/2024.02-1}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$LEROBOT_CONDA_ENV"
  LEROBOT_PYTHON=$(command -v python)
else
  : "${LEROBOT_PYTHON:?Set absolute environment python, or LEROBOT_CONDA_ENV}"
fi
[[ "$LEROBOT_PYTHON" == /* && -x "$LEROBOT_PYTHON" ]] || { echo 'Python must be an executable absolute path' >&2; exit 2; }
[[ -f "$BENCHMARK_SCRIPTS/run_diffusion.py" ]] || { echo 'Training runner not found' >&2; exit 2; }

mkdir -p "$DP_OUTPUT_BASE"
# Keep model/dataset caches on project storage instead of the home quota.
export HF_HOME="${HF_HOME:-$DP_OUTPUT_BASE/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DP_OUTPUT_BASE/cache/torch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
"$LEROBOT_PYTHON" - <<'PY'
import sys, torch
from lerobot.utils.import_utils import require_package
require_package("diffusers", extra="diffusion")
from importlib.metadata import version
print('Python:', sys.executable, 'LeRobot:', version('lerobot'), 'Torch:', torch.__version__)
assert version('lerobot') == '0.5.2', 'Expected the pinned LeRobot environment'
assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU training'
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
x=torch.ones((32,32),device='cuda');assert torch.isfinite(x@x).all()
PY
extra=()
if [[ -n "${DP_STEPS:-}" ]]; then extra+=(--steps "$DP_STEPS"); fi
if [[ "${DP_SMOKE:-0}" == 1 ]]; then extra+=(--smoke); fi
if [[ -n "${DP_DATASET_ROOT:-}" ]]; then extra+=(--dataset-root "$DP_DATASET_ROOT"); fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then extra+=(--entity "$WANDB_ENTITY"); fi
# LeRobot saves native checkpoints. Do not use the skill's generic DMTCP
# process-checkpoint example for this CUDA training job.
trap 'if command -v my-job-stats >/dev/null 2>&1; then my-job-stats -a -n -s || true; fi' EXIT
"$LEROBOT_PYTHON" "$BENCHMARK_SCRIPTS/run_diffusion.py" --task "$DP_TASK" \
  --output "$DP_OUTPUT_BASE/${DP_TASK}_${SLURM_JOB_ID}" \
  --wandb-mode "${WANDB_MODE:-online}" "${extra[@]}" --execute
