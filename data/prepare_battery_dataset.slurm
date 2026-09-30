#!/bin/bash
# Download the pinned public dataset on a CPU node before GPU jobs start.
#SBATCH --job-name=so101-battery-data
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/battery-data-%j.out
#SBATCH --error=outputs/logs/battery-data-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the so_arm101 project root}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
export HF_HOME="$PWD/cache/huggingface"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
import pandas as pd

root = Path.cwd()
entry = json.loads((root / 'data/datasets.json').read_text())['battery_insertion']
assert entry['revision'] and len(entry['success_episodes']) == 100
dest = root / 'datasets' / entry['repo_id'].split('/')[-1]
snapshot_download(repo_id=entry['repo_id'], repo_type='dataset', revision=entry['revision'], local_dir=dest, max_workers=4)
info = json.loads((dest / 'meta/info.json').read_text())
report = json.loads((dest / 'meta/validation.json').read_text())
tasks = pd.read_parquet(dest / 'meta/tasks.parquet')
assert info['codebase_version'] == 'v3.0'
assert (info['total_episodes'], info['total_frames']) == (entry['total_episodes'], entry['total_frames'])
assert report['passed'] and report['valid_success_episodes'] == entry['success_episodes']
assert len(tasks) == 1 and 'battery' in tasks.index[0].lower()
print(f'Dataset ready: {dest} ({len(entry["success_episodes"])} selected successes, revision {entry["revision"]})')
PY
