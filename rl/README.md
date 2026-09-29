# 后续真实机器人 RL（尚未实现）

这里预留官方 LeRobot HIL-SERL 集成。当前只有阶段说明，无训练脚本、奖励分类器、actor/learner 或机器人自动运动入口。

开始条件：先完成 ACT 基线与真机评估，并完成可复现的硬件 benchmark 验收。

1. 用示范构建兼容的 offline replay；ACT checkpoint **不能直接当作 stock HIL-SERL SAC 的续训 checkpoint**。
2. 配置 SO-101 leader 人工接管、EE 工作空间、最大 EE 单步位移、关节边界、固定复位姿态与 episode 超时。
3. 从人工成功/失败奖励开始，不先训练奖励分类器。
4. 分别验证 actor 与 learner，先运行极小规模真机测试。
5. 稳定后按 20 → 100 → 200–600 episodes 扩展；保留全部在线与人工接管 transition。
6. 记录奖励、critic/actor loss、接管次数、成功率、episode 时长。人工奖励流程工作后再评估奖励分类器。

未来其他 RL 方法可放独立子目录，避免与 `baselines/act/` 的训练输出和配置混用。sim-to-real 暂不实现。
