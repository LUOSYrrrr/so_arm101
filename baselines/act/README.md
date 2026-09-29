# Two independent SO-101 ACT baselines

Each task has 30 human-success, technically validated demonstrations. The raw datasets also retain failures/excluded episodes; `datasets.json` pins the Hub commit and successful indices. `run_act.py` passes the indices explicitly. No RL, no automatic robot evaluation, no automatic model publication.

## Reproduce code

Use the official LeRobot repository at the exact clean local commit:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout a07f22e22ce88cddff1f6eddced9ea008fbfc37c
```

Install that checkout in a dedicated Python 3.12 environment on an allocated compute node (`pip install -e '.[training]'`). Match the pinned code, check Torch CUDA and video decoding before training. `local_versions.json` records this machine's packages as a reference, not a portable lockfile. Do not clone the floating main branch and assume it matches. Do not train/install heavy environments on the login node.

Keep this `baselines/act` directory in your project Git repository. Do not commit datasets, caches, credentials, or checkpoints. The local robot LeRobot checkout is unchanged; a personal fork is needed only if future upstream code changes are required.

## Data and training

Download the exact `repo_id` / `revision` from `datasets.json` with `hf download REPO_ID --repo-type dataset --revision COMMIT --local-dir DATA_DIR`. Test one decoded sample from each view on a compute node. Initial ResNet18 pretrained weights may need a network download/cache before an offline job.

A single A100 per run is the starting allocation, batch size 8. Keep official ACT model/optimizer defaults: chunk size 100, ResNet18, learning rate 1e-5, KL weight 10. Baseline: 100,000 steps, checkpoints every 20,000; these are initial defaults, not a claim about convergence or required runtime. A 100-step smoke test precedes each full run. The 4-hour Slurm limit is a template; estimate full wall time from measured smoke throughput and request sufficient time before submitting the baseline. All output directories must be distinct.

```bash
python run_act.py --task pick_lift --output /project/outputs/lift_smoke --smoke
```

This only prints the command. Add `--execute` on an allocated GPU node. Use `--task pick_place` for the second model. Use `--dataset-root DATA_DIR` for predownloaded data. All 30 successful episodes are used for baseline training; there is no held-out imitation-loss validation split in this initial configuration. Real success must be measured later with at least 20 independent robot trials per policy.

## Spartan

Current SSH from this computer failed authentication. No transfer, GPU job, or training run has been submitted. Partition/account availability must be verified after login:

```bash
sinfo -p gpu-a100,gpu-a100-short -o '%P %G %l'
```

After code/environment/data setup:

```bash
# Option A: initialize the dedicated environment inside the job:
export LEROBOT_CONDA_ENV=/absolute/path/to/lerobot/env
# Option B: omit LEROBOT_CONDA_ENV and set a verified absolute LEROBOT_PYTHON.
# Do not use the audio_llm environment from generic skill examples.
export BENCHMARK_SCRIPTS=/absolute/path/to/so_arm101/baselines/act
export ACT_OUTPUT_BASE=/data/gpfs/projects/punim2341/siyuanluo/so101/outputs
export ACT_TASK=pick_lift
export ACT_SMOKE=1
export WANDB_MODE=offline
sbatch "$BENCHMARK_SCRIPTS/train_act.slurm"
```

Repeat for pick_place after checking the first smoke test. The script defaults to `gpu-a100-short` and 100 steps. After smoke validation, submit a separate full run with explicit partition/time overrides (the duration below is an example, not a runtime estimate):

```bash
export ACT_SMOKE=0
sbatch --job-name=so101-act-baseline --partition=gpu-a100 --time=12:00:00 \
  "$BENCHMARK_SCRIPTS/train_act.slurm"
```

Measure throughput and choose sufficient time before submitting. Each run uses one GPU, eight CPUs and 40 GB host RAM (not GPU memory). HF/Torch caches default to project output storage. The job checks CUDA before training; uses native LeRobot checkpoints rather than DMTCP. The downloaded Spartan skill lists partition/module settings, but actual availability, permissions and GPU memory must be verified on Spartan; cluster authentication has not been restored.

## W&B

One project `so101-act-baselines`, separate named runs for each task and smoke/baseline. LeRobot logs total loss, ACT L1/KL outputs, learning rate, gradient norm and timing; W&B can also capture GPU metrics. No real-world success rate exists during offline training. Model artifact upload is disabled; checkpoints stay in the output directory.

For live charts, run `wandb login` interactively (never paste the key into chat) and use `--wandb-mode online` / WANDB_MODE=online. Optionally pass your entity. No account, project or run has been created by these scripts yet. If compute nodes cannot reach W&B, use offline mode, then `wandb sync PATH_TO_OFFLINE_RUN` from a network-capable environment. That gives delayed charts, not real-time monitoring. Share the accessible run URL or local logs when asking for analysis.
