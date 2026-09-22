# ET-PRL

事件驱动预测-强化学习（Event-driven PRL）项目：冷冻站供水温度控制。整合自 `Event-DQN`（业务/论文代码）与 `ml-toolkit`（通用 ML 库），采用标准 src 布局 + YAML 单一事实源配置 + 实验结果统一管理。

主流程：
- LSTM 负荷预测（训练/验证/测试/超参优化/CL_next 预测）
- DQN 控制策略训练与测试
- 在线异常门控（gate）预热、超参优化与敏感性分析
- 固定间隔 vs 事件驱动控制策略对比
- 消融实验（full_et_prl / wo_dual_threshold / wo_local_threshold / wo_multi_scale / 单尺度变体）

## 快速开始

### 1) 环境准备
- 安装依赖：`uv sync`
- 所有命令统一使用 `uv run`（禁止直接 `python`/`pip`）

### 2) 统一入口（CLI）
- DQN 训练：`uv run etprl-train-dqn`
- DQN 测试：`uv run etprl-test-dqn`
- DQN 超参优化：`uv run etprl-optimize-dqn`
- LSTM 训练：`uv run etprl-train-lstm`
- LSTM 测试：`uv run etprl-test-lstm`
- LSTM 超参优化：`uv run etprl-optimize-lstm`
- LSTM 预测 CL_next：`uv run etprl-predict-lstm`
- 控制策略对比：`uv run etprl-compare-control`
- 事件驱动测试：`uv run etprl-test-event-driven`
- 在线门控预热：`uv run etprl-prewarm-gate`
- 门控超参优化：`uv run etprl-optimize-gate`
- 门控超参敏感性分析：`uv run etprl-analyze-gate-sensitivity`
- 消融实验：`uv run etprl-run-ablation-all`（及各 `etprl-run-ablation-*` 单项）

所有入口均支持 `--help` 查看参数。

### 3) 论文导出（Markdown -> Word）
- 前置：已安装 Pandoc（`pandoc --version` 检查）；若要从 `docs/pics/fig2.drawio` 重新导出 Figure 2，还需 draw.io/diagrams.net 与 Inkscape 命令行工具
- 若修改了 `docs/pics/fig2.drawio`，先导出 Word 友好的 plain SVG：`uv run python docs/papers/_docx_build/export_drawio_plain_svg.py`
- 若缺少 draw.io 或 Inkscape，按脚本提示安装后重试；也可用 `DRAWIO_EXE`、`INKSCAPE_EXE` 指定可执行文件路径
- 生成 Word：`uv run python docs/papers/_docx_build/merge_paper.py --input docs/papers/小论文-en.md`

## 配置（YAML 单一事实源）

- 配置文件位于 `configs/`，以 `base.yaml` 为根，各实验配置通过 `extends` 继承并只写差异字段
- 配置 schema 由 `src/et_prl/config/` 下的 frozen dataclass 定义（无默认值，缺字段/未知字段均 fail-fast）
- 继承支持单字符串或列表（列表按"后者覆盖"合并）；`TIMESTAMP` 运行时注入
- 加载入口：`et_prl.config.loader.load_config(name)` / `load_config_path(path)`

| 配置文件 | 说明 |
| --- | --- |
| `base.yaml` | 公共字段根配置 |
| `lstm.yaml` | LSTM 预测 |
| `dqn.yaml` | DQN 控制 |
| `gate.yaml` | 在线异常门控 |
| `control_compare.yaml` | 控制策略对比（extends [gate, dqn]） |
| `event_driven.yaml` | 事件驱动 DQN 评估（extends [gate, dqn]） |
| `ablation/*.yaml` | 消融实验变体（extends control_compare） |

## 目录约定

- `src/et_prl/`：唯一代码包
  - `config/`：配置 schema（frozen dataclass）+ YAML 加载器
  - `cli/`：统一命令入口（CLI）
  - `models/`：网络结构（LSTM / QNetwork / gate MLP）
  - `agents/`：RL 智能体（通用 DQN 基类 + 业务特化）
  - `environments/`：序列控制环境
  - `training/`：训练器（LSTM / DQN / gate value）
  - `evaluation/`：评估器
  - `experiments/`：实验编排
  - `data/`：数据加载/归一化
  - `detection/`：在线异常门控与事件触发
  - `plotting/`：绘图脚本（输出统一写 `outputs/figures/`）
  - `utils/`：通用工具（日志/指标/checkpoint/可复现性）
- `configs/`：YAML 配置（单一事实源）
- `data/`：数据（gitignore）
  - `shanghai_chiller/`：上海冷冻站实测数据（2021.7-10，5min）与预处理输出（`raw_data.csv`、`action_all.npy`、`coeff_date.npy`、`dqn/`、`lstm/`）
  - `external/`：外部公开数据集（一个数据集一个目录，目录内附来源/许可/字段说明）
- `outputs/`：实验结果与图表（gitignore，约定见 `outputs/README.md`）
- `docs/`：论文与图表源文件

## 实验产物说明

每次实验写入 `outputs/runs/<experiment>/[<grouping>/]<timestamp>/`（`<grouping>` 为可选分组层，如 train / eval）：
- `config.yaml`：实验配置快照
- `run.log`：执行日志（训练/评估统一）
- `metrics.json`：指标摘要
- `checkpoints/`：`best_model.pth` 与 `final_model.pth`
- `results/`：`training_history.csv`（训练）、`*_step_results.csv`（评估/对比）、优化 trials 等结果文件
- `figures/`：本次运行生成的图表
- `tb/`：TensorBoard 事件（`tensorboard --logdir outputs/runs/` 查看曲线）

## 导入与工程规范

- 统一通过包路径导入（`et_prl.*`），不依赖当前工作目录的脚本相对导入
- 可执行逻辑放在 `src/et_prl/cli/`，其余目录仅保留可复用函数/类
- 算法与通用工具能力沉淀在 `et_prl` 包内；业务编排/配置/流程保留在本项目
- 路径一律基于 `et_prl.config.base.project_root()` 锚定，不依赖 cwd
