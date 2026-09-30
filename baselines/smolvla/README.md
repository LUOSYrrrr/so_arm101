# SO-101 双视角 SmolVLA 微调

从 `lerobot/smolvla_base` 微调 Pick-and-Lift 和 Pick-and-Place，复用 `data/datasets.json` 的固定 HF revision 和各 30 条成功 episode。预训练模型 revision 固定为 `d9f33c94a60fb382c90dea2164c96845bd955e28`。

默认每个任务 20,000 步、batch 8、seed 1000，每 5,000 步保存 checkpoint。冻结 VLM，训练约 100M 参数的 action expert 和 state projection。第三人称相机映射为 `camera1`，腕部相机映射为 `camera2`；沿用官方 512×512 等比例缩放与 padding、50 步 action chunk，以及 10 步 flow matching 推理。

W&B 使用 online 模式，项目 `so101-smolvla-baselines`，不自动上传模型。HF 元数据在 CPU 作业中缓存，训练时离线读取缓存；这不影响 W&B 联网。

## Spartan 提交

从项目根目录操作。要求已有 ACT/DP 使用的 `.venv`（LeRobot 0.5.2，源码 revision `a07f22e22ce88cddff1f6eddced9ea008fbfc37c`）、`outputs/setup/bootstrap/bin/uv` 和两个完整的本地数据集目录 `datasets/<repo basename>`。准备作业添加 `transformers==5.3.0`、`num2words`，缓存预训练权重与 tokenizer。

```bash
mkdir -p outputs/logs outputs/setup
# 首次使用或登录失效时，在终端交互登录；不将密钥写进脚本。
.venv/bin/wandb login

setup_job=$(sbatch --parsable baselines/smolvla/setup_spartan.slurm)
export SVLA_TASK=pick_place
sbatch --dependency=afterok:"$setup_job" --kill-on-invalid-dep=yes \
  --job-name=svla-pick-place baselines/smolvla/train_smolvla.slurm
export SVLA_TASK=pick_lift
sbatch --dependency=afterok:"$setup_job" --kill-on-invalid-dep=yes \
  --job-name=svla-pick-lift baselines/smolvla/train_smolvla.slurm
```

默认申请单张 A100、8 CPU、40 GB RAM、24 小时上限，分区为 `gpu-a100-preempt`、QoS 为 `publicgpu`。输出为 `outputs/smolvla/<task>_<jobid>`，日志为 `outputs/logs/smolvla-<jobid>.out/.err`。训练成功后写入 `training_run.json`。

## 抢占后恢复

该分区的 `PreemptMode=CANCEL` 会取消作业，不会自动重排队。检查原作业已经停止，并且 `checkpoints/last` 包含完整模型和训练状态后重新提交：

```bash
export SVLA_TASK=pick_place
export SVLA_RUN_OUTPUT="$PWD/outputs/smolvla/pick_place_<原jobid>"
export SVLA_RESUME_FROM="$SVLA_RUN_OUTPUT/checkpoints/last"
sbatch --job-name=svla-pick-place-resume baselines/smolvla/train_smolvla.slurm
```

恢复原生模型、优化器、scheduler、RNG 和步数，并续接原 W&B run。开始新任务前运行 `unset SVLA_RUN_OUTPUT SVLA_RESUME_FROM`。

此训练不控制机械臂。真机部署需沿用训练的相机映射、语言任务、校准和归一化参数，单独检查推理延迟及成功率；训练 loss 不等于抓取成功率。
