# outputs/ 目录约定

本目录存放全部实验产物与图表，整体被 gitignore（不入库）。

## 目录结构

```
outputs/
├── runs/                          # 实验运行产物
│   └── <experiment>/<mode>/<timestamp>/
│       ├── config.yaml            # 实验配置快照
│       ├── experiment.log         # 执行日志
│       ├── evaluation.log         # 评估日志（评估任务）
│       ├── metrics.json           # 指标摘要
│       ├── training_history.csv   # 训练过程明细（训练任务）
│       ├── checkpoints/           # best_model.pth / final_model.pth / checkpoint.pth
│       └── events.out.tfevents.*  # TensorBoard 事件
└── figures/                       # 绘图脚本统一输出
```

## 约定

- **运行目录**：`runs/<experiment>/<mode>/<timestamp>/`
  - `<experiment>`：实验名（lstm / dqn / gate / control_compare / event_driven / ablation 等）
  - `<mode>`：train / test / eval / optimize 等
  - `<timestamp>`：`YYYYMMDD_HHMMSS`，每次运行唯一
- **图表输出**：所有绘图脚本统一写 `figures/`，不再散落到 `docs/pics/`（论文图从 `figures/` 归档）。
- **TensorBoard**：`tensorboard --logdir outputs/runs/` 查看训练/评估曲线。
- **存量迁移**：旧 `Event-DQN/logs/` 已全量复制为 `outputs/runs/` 存量，checkpoint 可被新代码直接加载。

## 说明

- 路径一律基于 `et_prl.config.base.project_root()` 锚定，不依赖当前工作目录。
- 本目录内容可随时删除重建（重新运行实验即可再生），故不纳入版本控制。
