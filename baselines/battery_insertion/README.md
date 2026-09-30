# Battery insertion: ACT, Diffusion Policy, SmolVLA

Dataset: `LUOSYrrrrr/so101_battery_insertion_v2` (LeRobot v3.0, dual RGB, 20 FPS). The exact revision and 100 human-labeled successful episode indices are pinned in `data/datasets.json`. The upload retains 7 excluded and 1 failed episode for audit; all three training entries pass only the success indices. The export corrects the collector's accidental Pick-and-Lift task text; never train from the uncorrected raw folder.

The success labels are human annotations. The technical validator verifies frames, timestamps, and video decoding, not physical insertion depth. Evaluate actual success on the robot after training. Keep the charger and cameras at the recorded positions during initial evaluation.

## Prepare once on Spartan

From `/data/gpfs/projects/punim2341/siyuanluo/so_arm101`, after pulling the code and activating the existing LeRobot `.venv` setup:

```bash
cd /data/gpfs/projects/punim2341/siyuanluo/so_arm101
git pull --ff-only
mkdir -p outputs/logs outputs/setup
.venv/bin/wandb login
data_job=$(sbatch --parsable data/prepare_battery_dataset.slurm)
setup_job=$(sbatch --parsable baselines/smolvla/setup_spartan.slurm)
```

The CPU jobs download the pinned dataset and prepare the pinned SmolVLA base model. They do not train on a login node. Check with `squeue --me` and inspect `outputs/logs/` before using the GPU outputs.

## Submit three independent training runs

Submit from the same project root in the same shell as the preceding commands. ACT and Diffusion use the existing LeRobot 0.5.2 environment; SmolVLA uses its existing setup and pretrained base. Each model writes to a separate output directory and W&B project.

```bash
export LEROBOT_PYTHON="$PWD/.venv/bin/python"
unset LEROBOT_CONDA_ENV
export BENCHMARK_SCRIPTS="$PWD/baselines/act"
export ACT_TASK=battery_insertion ACT_SMOKE=0 WANDB_MODE=online
export ACT_OUTPUT_BASE="$PWD/outputs/act"
export ACT_DATASET_ROOT="$PWD/datasets/so101_battery_insertion_v2"
sbatch --dependency=afterok:"$data_job" --job-name=act-battery \
  baselines/act/train_act.slurm

export BENCHMARK_SCRIPTS="$PWD/baselines/diffusion"
export DP_TASK=battery_insertion DP_SMOKE=0 WANDB_MODE=online
export DP_OUTPUT_BASE="$PWD/outputs/diffusion"
export DP_DATASET_ROOT="$PWD/datasets/so101_battery_insertion_v2"
sbatch --dependency=afterok:"$data_job" --job-name=dp-battery \
  baselines/diffusion/train_diffusion.slurm

export SVLA_TASK=battery_insertion
unset SVLA_RUN_OUTPUT SVLA_RESUME_FROM
sbatch --dependency=afterok:"${data_job}:${setup_job}" --kill-on-invalid-dep=yes \
  --job-name=svla-battery baselines/smolvla/train_smolvla.slurm
```

ACT: 100k optimizer steps, batch 8, A100 short partition; DP: 100k steps, batch 8, A100 partition; SmolVLA: 20k steps, batch 8, A100 preempt partition. These are starting budgets, not performance guarantees. Use `squeue --me` and the W&B projects `so101-act-baselines`, `so101-diffusion-baselines`, and `so101-smolvla-baselines` to monitor progress. A preempted SmolVLA job requires resubmission from a complete checkpoint; see `baselines/smolvla/README.md`.
