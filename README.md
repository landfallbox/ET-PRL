# Event-DQN

事件驱动 DQN 控制项目，包含以下主流程：
- LSTM 负荷预测（训练/验证/测试/超参优化）
- DQN 控制策略训练与测试
- 固定间隔 vs 事件驱动控制策略对比

## 快速开始

### 1) 环境准备
- 安装依赖：`uv sync`
- 所有命令统一使用 `uv run`

### 2) 统一入口（Applications）
- DQN 训练：`uv run event-dqn-train-dqn`
- DQN 测试：`uv run event-dqn-test-dqn --fixed_interval 4`
- DQN 超参优化：`uv run event-dqn-optimize-dqn --n_trials 30 --max_episodes 30`
- LSTM 训练：`uv run event-dqn-train-lstm`
- LSTM 测试：`uv run event-dqn-test-lstm`
- LSTM 超参优化：`uv run event-dqn-optimize-lstm`
- LSTM 预测 CL_next：`uv run event-dqn-predict-lstm`
- 控制策略对比：`uv run event-dqn-compare-control --fixed_interval 4`
- 事件驱动测试：`uv run event-dqn-test-event-driven`
- 在线门控预热：`uv run event-dqn-prewarm-gate`

## 目录约定

- `src/cli/`：统一命令入口（CLI）
- `src/dqn/`、`src/cl_predict/`、`src/control_evaluation/`：可复用业务模块
- `src/online_anomaly_detection/`：在线异常门控与事件触发逻辑
- `config/`：配置层（仅放配置，不放业务流程）
- `data/`：输入数据与预处理输出
- `logs/`：实验产物（训练/评估/优化）

## 实验产物说明

每次实验会写入 `logs/<experiment_name>/<mode>/<timestamp>/`：
- `config.yaml`：实验配置快照
- `experiment.log` / `evaluation.log`：执行日志
- `metrics.json`：指标摘要
- `training_history.csv`：训练过程明细（训练任务）
- `checkpoints/`：`best_model.pth` 与 `final_model.pth`

## 导入与工程规范

- 统一通过包路径导入（例如 `src.*`、`config.*`），不要依赖当前工作目录的脚本相对导入
- 可执行逻辑放在 `src/cli/`，业务目录仅保留可复用函数/类
- 通用编排能力优先沉淀到 `ml-toolkit`
