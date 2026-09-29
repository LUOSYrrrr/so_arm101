# 数据版本与发布

`datasets.json` 固定两个 HF 数据集的 commit 与可用于 ACT 的成功 episode 索引。全部失败/排除记录也保存在 HF，但不作为本次 ACT 示范使用。

`publish_datasets.py` 是本次 2026-09-29 数据发布的专用脚本（包含经用户确认的第二任务文字修正），不是任意新数据集的通用上传器。默认只生成本地上传副本；加 `--upload` 才会上传，`--public` 会公开仓库。不要为新一批数据直接覆盖已固定的基线版本。

原始记录位于外层 `datasets/`，导出副本位于 `outputs/hf_exports/`。此目录不存录像或凭证。数据格式为 LeRobot v3.0。
