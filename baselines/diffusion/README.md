# Diffusion Policy 双视角基线

复用 `data/datasets.json` 中固定 HF revision 和每个任务 30 条成功 episode；两个任务分别训练。独立输出，不修改 ACT 或机器人控制。该脚本不会启动机械臂。

实现：LeRobot 0.5.2，已在本地 checkout `a07f22e22ce88cddff1f6eddced9ea008fbfc37c` 检查。服务器使用训练 ACT 的环境，先确认 diffusion 依赖可导入：

```bash
python -c 'import diffusers; print(diffusers.__version__)'
```

若缺 diffusers，在实际 LeRobot checkout 路径安装其 extra：`python -m pip install -e '/实际/lerobot路径[diffusion]'`。不要重新安装或替换整套 CUDA 环境。

## 配置

- batch 8、seed 1000；双 RGB 原始 640×480，未额外裁剪或缩放。
- 沿用官方 Diffusion 默认：ResNet18（无预训练权重）、共享视觉编码器、2 帧观测、horizon 16、执行 8 步、DDPM 100 diffusion timesteps、学习率 1e-4、cosine scheduler、warmup 500。
- 默认不启用 AMP，不改 loss。与 ACT 的骨干预训练和优化器不同，因此这是默认方法对照，不是只改一种模型结构的消融。
- smoke 1,000 步；正式 100,000 步，每 5,000 步存 checkpoint。可用 DP_STEPS 覆盖。100k 是初始预算，不保证达到最优。
- W&B 默认 online，项目 `so101-diffusion-baselines`。模型和视频不会自动上传。
- 未设置仿真环境，`eval_freq=0`；训练 loss 不是成功率，也不能与 ACT loss 数值直接比较。保存完整预处理/后处理文件用于后续真机评估。

## Spartan 提交

在已训练 ACT 的环境激活后，从项目根目录：

```bash
cd /data/gpfs/projects/punim2341/siyuanluo/so_arm101
git pull --ff-only
export LEROBOT_PYTHON="$(command -v python)"
unset LEROBOT_CONDA_ENV
export BENCHMARK_SCRIPTS="$PWD/baselines/diffusion"
export DP_OUTPUT_BASE="$PWD/outputs/diffusion"
export DP_TASK=pick_place
export DP_SMOKE=0
export WANDB_MODE=online
wandb login
sbatch baselines/diffusion/train_diffusion.slurm
```

要求 `LEROBOT_PYTHON` 来自实际 lerobot 环境。脚本检查 LeRobot 版本与 GPU，只在 Slurm 作业内训练。已有下载的数据可以设置 `DP_DATASET_ROOT` 指向同一个任务的完整数据集目录；否则按固定 revision 从 HF 获取。切换任务时应同时修改或取消该变量。

默认直接正式训练 100,000 步，申请 24 小时（分区以账户实际权限为准）。另一个任务独立提交：

```bash
export DP_SMOKE=0
sbatch --partition=gpu-a100 --time=24:00:00 --job-name=dp-pick-place baselines/diffusion/train_diffusion.slurm
# 另一个任务独立作业：
export DP_TASK=pick_lift
unset DP_DATASET_ROOT
sbatch --partition=gpu-a100 --time=24:00:00 --job-name=dp-pick-lift baselines/diffusion/train_diffusion.slurm
```

输出 `${DP_OUTPUT_BASE}/${DP_TASK}_${SLURM_JOB_ID}`，终端日志 `diffusion-<jobid>.out/.err`。保留检查点，续训需使用 LeRobot 原生 resume 并确认 train_config；本入口拒绝覆盖已有输出目录。

离线 W&B 后续用同一环境的 `wandb sync <输出目录>/wandb/offline-run-*` 同步；在线日志可在已登录且计算节点网络允许时设置 `WANDB_MODE=online`。

## 时间与运行记录

不安排测速作业，直接正式训练。当前没有 A100 的实测 Diffusion 吞吐，不承诺固定完成时间。在线 W&B 的稳定 update_s + dataloading_s 可用于估计剩余时间。24 小时是申请上限，不是预计耗时；若稳定每步 0.1 / 0.2 / 0.5 秒，100k 步的计算时间分别约 2.8 / 5.6 / 13.9 小时，另加初始化和保存。训练期间不要因短期 loss 波动重复启动作业。

运行前在同一环境执行 wandb login，API key 仅输入终端交互，不提交到 Git。计算节点需能访问 W&B；online 模式不会自动降为 offline。保存本地日志和 checkpoint，即使网络中断也保留产物。

DDPM 的 100 个去噪步骤主要增加推理成本，训练每个样本随机采样 diffusion timestep，不是每次更新都完整去噪 100 次。真机前必须单独验证推理延迟，不能直接复制 ACT 部署参数。
