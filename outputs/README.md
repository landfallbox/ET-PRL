# outputs/ 目录约定

本目录存放全部实验产物与图表，整体被 gitignore（不入库）。

## 目录结构

```
outputs/
├── runs/                          # 实验运行产物
│   └── <experiment>/[<grouping>/]<timestamp>/
│       ├── config.yaml            # 实验配置快照
│       ├── run.log                # 执行日志（训练/评估统一）
│       ├── metrics.json           # 指标摘要
│       ├── checkpoints/           # best_model.pth / final_model.pth / checkpoint.pth
│       ├── results/               # 结果文件
│       │   ├── training_history.csv   # 训练过程明细（训练任务）
│       │   ├── *_step_results.csv     # 逐步评估结果（评估/对比任务）
│       │   └── ...                    # 优化 trials.csv、best_config.json 等
│       ├── figures/               # 本次运行生成的图表（png）
│       └── tb/                    # TensorBoard 事件（events.out.tfevents.*）
└── figures/                       # 绘图脚本统一输出
```

## 约定

- **运行目录**：`runs/<experiment>/[<grouping>/]<timestamp>/`
  - `<experiment>`：实验名（lstm / dqn / gate / control_compare / event_driven / ablation 等）
  - `<grouping>`：可选分组层（train / eval / optimization / sensitivity 等），由配置 `TRAIN_SUBDIR` / `EVAL_SUBDIR` 等控制；一次性对比实验（如 control_compare）无此层
  - `<timestamp>`：`YYYYMMDD_HHMMSS`，每次运行唯一
- **入口文件**：`config.yaml` / `run.log` / `metrics.json` 固定位于运行目录顶层
- **产物分层**：结果数据（CSV/JSON/报告）入 `results/`，图表入 `figures/`，TensorBoard 事件入 `tb/`，模型入 `checkpoints/`
- **图表输出**：所有绘图脚本统一写 `figures/`，不再散落到 `docs/pics/`（论文图从 `figures/` 归档）。
- **TensorBoard**：`tensorboard --logdir outputs/runs/` 查看训练/评估曲线。
- **存量迁移**：旧 `Event-DQN/logs/` 已全量复制为 `outputs/runs/` 存量，checkpoint 可被新代码直接加载。

## 说明

- 路径一律基于 `et_prl.config.base.project_root()` 锚定，不依赖当前工作目录。
- 本目录内容可随时删除重建（重新运行实验即可再生），故不纳入版本控制。
