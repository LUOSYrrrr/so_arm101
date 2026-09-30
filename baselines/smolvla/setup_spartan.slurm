#!/bin/bash
#SBATCH --job-name=so101-smolvla-setup
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/smolvla-setup-%j.out
#SBATCH --error=outputs/logs/smolvla-setup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export UV_CACHE_DIR="$PWD/cache/uv"
export HF_HOME="$PWD/cache/huggingface"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
outputs/setup/bootstrap/bin/uv pip install --python "$PWD/.venv/bin/python" \
  'transformers==5.3.0' 'num2words>=0.5.14,<0.6.0'
.venv/bin/python baselines/smolvla/prepare.py
