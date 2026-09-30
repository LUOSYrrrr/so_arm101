# SO-101 实验项目

本目录集中管理 SO-101 的采集、数据发布、模仿学习基线和后续 RL 实验，使用独立 Git 仓库 [LUOSYrrrr/so_arm101](https://github.com/LUOSYrrrr/so_arm101)。

```text
so_arm101/
├── hardware/dashboard/     # 双视角预览、遥操录制、续录、数据校验
├── data/
│   ├── datasets.json       # 两个 HF 数据集的固定版本与成功 episode 索引
│   └── publish_datasets.py # 制作上传副本、校验并发布 HF
├── baselines/
│   └── act/
│       ├── run_act.py      # 两个任务分别训练 ACT
│       ├── train_act.slurm # Spartan 单卡训练模板
│       ├── local_versions.json
│       └── README.md
└── rl/README.md            # 后续官方 HIL-SERL 阶段说明，尚未实现
```

## 当前状态

- 双视角：腕部 `observation.images.wrist`、第三视角 `observation.images.third_person`。
- 两个任务：Pick-and-Lift；抓取黑方块放到蓝区并松开。各有 30 条通过技术校验的人工成功示范。
- ACT 使用同一实现、独立数据与 checkpoint；先 batch size 8、100 步测试，再正式训练。没有开始训练或 RL。
- LeRobot 使用外层 `lerobot/` 官方独立 checkout，固定 commit `a07f22e22ce88cddff1f6eddced9ea008fbfc37c`。不复制或重写机器人控制代码。

## 使用入口

从本仓库根目录运行：

```bash
python hardware/dashboard/server.py
# 下条只打印训练命令，不执行训练：
/home/siyuanluo/anaconda3/envs/lerobot/bin/python baselines/act/run_act.py \
  --task pick_lift --output /tmp/act_lift_smoke --smoke
```

采集说明见 [hardware/dashboard/README.md](hardware/dashboard/README.md)，训练说明见 [baselines/act/README.md](baselines/act/README.md)。

## 文件与数据的边界

代码、配置和 HF 版本索引放在本目录并由 Git 管理。原始录制仍保留在本工作站外层 `datasets/`，会话日志和上传副本仍保留在外层 `outputs/`，避免破坏已有续录配置；二者均被 Git 忽略，不复制进代码仓库。`data/` 存储元数据与发布工具，不存原始视频。

HF 已公开发布：
- https://huggingface.co/datasets/LUOSYrrrrr/so101_pick_lift_dual_rgb_20260929
- https://huggingface.co/datasets/LUOSYrrrrr/so101_pick_place_blue_dual_rgb_20260929

旧 `tools/hardware_dashboard/` 的命令入口和旧 ACT Python 入口保留为转发脚本；后续开发修改本目录内的实现。

独立克隆：`git clone https://github.com/LUOSYrrrr/so_arm101.git`。新机器默认使用本仓库内 `datasets/`、`outputs/`；可用 `SO101_STORAGE_ROOT` 指定其他存储根目录。服务器只训练时无需启动硬件面板。

## Diffusion Policy 基线

新增双视角 Diffusion Policy 独立训练入口和 Spartan 作业脚本，见 [baselines/diffusion/README.md](baselines/diffusion/README.md)。复用相同 30 条成功示范与固定数据版本，默认直接分别训练两个任务 100,000 步，并在线同步 W&B。
