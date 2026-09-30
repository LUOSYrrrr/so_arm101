# 数据版本与发布

`datasets.json` 固定各 HF 数据集的 commit 与可用于模仿学习的成功 episode 索引。全部失败/排除记录也保存在 HF，但不作为成功示范使用。

`publish_datasets.py` 是本次 2026-09-29 数据发布的专用脚本（包含经用户确认的第二任务文字修正），不是任意新数据集的通用上传器。默认只生成本地上传副本；加 `--upload` 才会上传，`--public` 会公开仓库。不要为新一批数据直接覆盖已固定的基线版本。

`publish_battery_insertion.py` 专用于 2026-09-30 的电池插入 v2 录制：100 条人工成功、7 条排除、1 条失败；只在导出副本中修正采集器误写的 Pick-and-Lift 任务文字。`prepare_battery_dataset.slurm` 在 Spartan CPU 节点下载固定 revision 后校验关键元数据。三个模型的作业命令见 `baselines/battery_insertion/README.md`。

原始记录位于外层 `datasets/`，导出副本位于 `outputs/hf_exports/`。此目录不存录像或凭证。数据格式为 LeRobot v3.0。
