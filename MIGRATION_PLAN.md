# ET-PRL 代码整合迁移计划

> 将 `Event-DQN`（业务项目，62 个 py 文件）与 `ml-toolkit`（通用算法库，54 个 py 文件）
> 整合为单一项目 `ET-PRL`，按标准研究项目架构重新组织。

---

## 一、现状盘点

### 1.1 两个项目的职责

| 项目 | 职责 | 内容 |
|---|---|---|
| `Event-DQN` | 论文业务 | LSTM 预测、DQN 控制、事件门控、消融/对比实验、绘图、CLI 入口 |
| `ml-toolkit` | 通用组件 | 模型（LSTM/GateValueMLP/QNetwork）、RL（agent/buffer/env）、训练器、评估器、异常检测、校准、数据增强、通用工具 |

### 1.2 耦合点（迁移工作量来源）

- `Event-DQN` 中 **49 处** `from ml_toolkit.* import ...` 引用
- `ml-toolkit` 内部仅 **1 处** 绝对自引用（`training/gate_value_trainer.py` → `ml_toolkit.models`），其余均为相对导入，迁移成本低
- `Event-DQN` 通过 `pyproject.toml` 以 git 依赖方式引用 ml-toolkit（迁移后删除）
- 两个项目均无 pytest 测试套件（`*test*.py` 均为评估脚本），回归验证依赖实际运行

### 1.3 依赖合并（取最严格版本）

| 依赖 | Event-DQN | ml-toolkit | 合并结果 |
|---|---|---|---|
| torch | `>=2.10.0,<2.11` | `>=2.10.0` | `>=2.10.0,<2.11` |
| numpy | `>=2.4.2` | `>=1.24.0` | `>=2.4.2` |
| pandas | `>=3.0.0` | `>=3.0.0` | `>=3.0.0` |
| matplotlib | `>=3.10.0` | `>=3.8.0` | `>=3.10.0` |
| optuna | `>=4.7.0` | `>=4.0.0` | `>=4.7.0` |
| pyyaml | `>=6.0.3` | `>=6.0.3` | `>=6.0.3` |
| scikit-learn | — | `>=1.3.0` | `>=1.3.0`（校准模块需要） |
| python-docx | optional | — | optional（论文导出） |

### 1.4 非代码资产

| 资产 | 体积 | 处理 |
|---|---|---|
| `Event-DQN/data/` | 32.5 MB | 复制到新项目（raw_data.csv、.npy、预处理输出） |
| `Event-DQN/logs/` | 26.4 MB | 复制到 `outputs/runs/`（论文结果与 checkpoint，可复现性依赖） |
| `Event-DQN/docs/` | 34.8 MB | 整体复制（7 个期刊 LaTeX 目录 + pics 图） |
| `Event-DQN/AGENTS.md`、`.github/`（copilot 指令、skills） | — | 迁移并按新结构更新 |
| `ml-toolkit/examples/` | — | **不迁移**（库使用示例，失去意义） |
| `ml-toolkit/LICENSE` | — | 保留（MIT，个人项目） |

---

## 二、目标架构

### 2.1 目录结构（src layout，单一包 `et_prl`）

```
ET-PRL/
├── pyproject.toml              # 项目名 et-prl；合并依赖；[project.scripts] CLI 入口
├── README.md
├── AGENTS.md
├── .python-version             # 3.13
├── .gitignore
├── configs/                    # 参数单一事实来源（YAML 为准，extends 继承）
│   ├── base.yaml               # 全量字段规范（默认值载体 / 参数文档）
│   ├── default.yaml
│   ├── lstm.yaml
│   ├── dqn.yaml
│   ├── gate.yaml
│   └── ablation/               # 消融变体：extends: dqn + 开关
├── src/et_prl/
│   ├── __init__.py
│   ├── config/                 # 配置层（Python 默认值 + YAML 覆盖）
│   │   ├── base.py             # ← config/common_config.py（CommonConfig → BaseConfig）
│   │   ├── lstm.py             # ← config/lstm_config.py
│   │   ├── dqn.py              # ← config/dqn_config.py + event_driven_dqn_config.py
│   │   ├── gate.py             # ← config/online_anomaly_detection_config.py
│   │   ├── comparison.py       # ← config/compare_dqn_config.py + control_compare_config.py
│   │   └── ablation.py
│   ├── models/                 # 网络结构
│   │   ├── lstm.py             # ← ml_toolkit/models/lstm.py
│   │   ├── gate_value_mlp.py   # ← ml_toolkit/models/gate_value_mlp.py
│   │   └── q_network.py        # ← ml_toolkit/rl/networks/q_network.py
│   ├── agents/                 # RL 智能体
│   │   └── dqn/
│   │       ├── base.py         # ← ml_toolkit/rl/agents/dqn_agent.py（通用 DQN）
│   │       ├── agent.py        # ← Event-DQN src/dqn/agent.py（项目特化子类）
│   │       ├── rewards.py      # ← src/dqn/rewards.py（物理奖励函数）
│   │       └── replay_buffer.py# ← ml_toolkit/rl/buffers/*（合并 fixed 版）
│   ├── environments/           # ← ml_toolkit/rl/envs/
│   │   ├── env.py
│   │   └── sequence_env.py
│   ├── data/                   # 数据处理
│   │   ├── dataset.py          # ← ml_toolkit/data_processing/dataset_loader.py
│   │   ├── normalizer.py       # ← ml_toolkit/data_processing/normalizer.py
│   │   ├── tensor_loader.py    # ← ml_toolkit/data_processing/tensor_loader.py
│   │   ├── preprocess_lstm.py  # ← src/cl_predict/preprocess_data.py
│   │   ├── preprocess_dqn.py   # ← src/dqn/preprocess_dqn_data.py
│   │   └── augmentation/       # ← ml_toolkit/data_augmentation/
│   ├── detection/              # 异常检测 / 事件门控（两项目合并）
│   │   ├── streaming_gate.py   # ← src/online_anomaly_detection/streaming_anomaly_gate.py
│   │   ├── prewarm.py          # ← src/online_anomaly_detection/prewarm_streaming_gate.py
│   │   ├── multi_scale_tracker.py  # ← ml_toolkit/anomaly_detection/*
│   │   ├── streaming_isolation.py
│   │   ├── threshold_optimizer.py
│   │   └── metrics.py          # ← ml_toolkit/evaluation/event_gate_metrics.py
│   ├── training/               # 训练
│   │   ├── base.py             # ← ml_toolkit/training/trainer.py
│   │   ├── lstm_trainer.py     # ← ml_toolkit/training/lstm_trainer.py
│   │   ├── gate_value_trainer.py
│   │   ├── dqn_trainer.py      # ← src/dqn/dqn_train.py
│   │   └── hyperparam.py       # ← ml_toolkit/utils/hyperparameter_optimizer.py
│   ├── evaluation/             # 评估
│   │   ├── base.py             # ← ml_toolkit/evaluation/evaluator.py
│   │   ├── metrics.py          # ← ml_toolkit/evaluation/metrics.py
│   │   ├── lstm_evaluator.py
│   │   ├── dqn_evaluator.py    # ← src/dqn/dqn_test.py
│   │   ├── event_driven_eval.py# ← src/online_anomaly_detection/event_driven_test.py
│   │   └── control/            # ← src/control_evaluation/（评估框架：common + strategies/）
│   ├── calibration/            # ← ml_toolkit/calibration/
│   ├── experiments/            # 实验编排（依赖 evaluation/ 框架，方向单向）
│   │   ├── ablation/           # ← src/ablation/（消融变体实验）
│   │   └── comparison.py       # ← src/comparison/compare_control_strategies.py（策略对比实验）
│   ├── plotting/               # ← src/plotting/（独立入口脚本，按数据源分子目录）
│   │   ├── dataset/            # 数据集特征图（读 data/）
│   │   ├── training/           # 训练曲线（读 outputs/runs/<lstm|dqn>）
│   │   ├── control/            # 控制策略/触发对齐图（读 outputs/runs/<compare_dqn|control_compare>）
│   │   ├── ablation/           # 消融 Pareto 图
│   │   └── methods/            # 方法示意图（合成数据，无实验依赖）
│   ├── utils/                  # ← ml_toolkit/utils/（checkpoint/logger/metrics_recorder/
│   │                           #    orchestration/visualizer/reproducibility 等 12 个文件）
│   └── cli/                    # ← src/cli/（入口函数不变，仅改导入）
├── data/                       # 输入数据（gitignore）
├── outputs/                    # 实验结果（gitignore）
│   ├── runs/<experiment>/<mode>/<timestamp>/
│   │   ├── config.yaml         # 配置快照（已有机制，保留）
│   │   ├── experiment.log
│   │   ├── metrics.json
│   │   ├── training_history.csv
│   │   └── checkpoints/
│   └── figures/                # 论文图（SVG/PNG）
├── docs/                       # ← Event-DQN/docs（papers/、pics/）
├── scripts/                    # 一次性脚本（docx 构建、drawio 导出等）
└── tests/                      # 新增：核心模块 pytest 单测（可选，P7 后补充）
```

**分层约定（代码平面依赖方向单向向下）**：

```
cli/ ──> experiments/ ──> evaluation/ ──> training/ ──> agents|models|environments
              │                 │
              └─────────> data/ <──────────────────────┘
utils/、config/ 可被任意层引用
```

- `evaluation/`：可复用的评估**框架**（给定训练好的模型，怎么评估、算什么指标），
  不含"跑哪些实验"的编排逻辑
- `experiments/`：具体**实验编排**（跑哪组对比/消融、怎么汇总成表），依赖 `evaluation/`
- 判定标准：被多个实验脚本复用 → 归框架层；只服务某一篇论文实验 → 归 `experiments/`
- **`plotting/` 是 `cli/` 的对等入口层，不是被调用的业务层**（已核实：全项目无任何
  代码 import plotting，15 个脚本均为带 `argparse main()` 的独立入口，事后手动运行）：
  - 代码平面：只依赖 `utils/` 与自身内部绘图 helper；不被 `cli/`、`experiments/` 或
    其他业务层 import，仅通过 `uv run python -m et_prl.plotting.<script>` 手动触发
  - 数据平面：读 `data/`（数据集特征图）与 `outputs/runs/`（训练曲线、触发对齐等
    实验产物），统一写 `outputs/figures/`——文件 I/O 不经过代码层，不构成跨层依赖
  - 放顶层而非 `experiments/` 下的理由：① 不被实验代码调用（放下面会制造虚假耦合）；
    ② 数据源异构（训练 run / control_compare run / 原始数据 / 合成数据），不隶属单一实验；
    ③ 出图是实验完成后的人工独立步骤
  - 内部按实验/数据源分子目录兼顾就近内聚（见目录树）

### 2.2 模块映射总表

| 来源 | 目标 | 说明 |
|---|---|---|
| `ml_toolkit/models/*` | `et_prl/models/` | 直接迁移 |
| `ml_toolkit/rl/networks/q_network.py` | `et_prl/models/q_network.py` | 网络结构归入 models |
| `ml_toolkit/rl/agents/*` | `et_prl/agents/dqn/base.py` | 通用 DQN 基类 |
| `ml_toolkit/rl/buffers/*` | `et_prl/agents/dqn/replay_buffer.py` | 合并 fixed/普通版 |
| `ml_toolkit/rl/envs/*` | `et_prl/environments/` | 直接迁移 |
| `ml_toolkit/training/*` | `et_prl/training/` | 直接迁移 |
| `ml_toolkit/evaluation/*` | `et_prl/evaluation/`、`et_prl/detection/metrics.py` | event_gate_metrics 归入 detection |
| `ml_toolkit/data_processing/*` | `et_prl/data/` | 直接迁移 |
| `ml_toolkit/anomaly_detection/*` | `et_prl/detection/` | 与 Event-DQN 门控模块合并 |
| `ml_toolkit/calibration/*` | `et_prl/calibration/` | 直接迁移 |
| `ml_toolkit/data_augmentation/*` | `et_prl/data/augmentation/` | 归入 data |
| `ml_toolkit/utils/*` | `et_prl/utils/` | 迁移 + 基础设施重构（见 2.7）：`metrics_recorder` 换 SummaryWriter、`logger` 修命名 |
| `Event-DQN/src/dqn/agent.py` | `et_prl/agents/dqn/agent.py` | 继承关系保留（base + 特化子类） |
| `Event-DQN/src/dqn/{dqn_train,dqn_test,rewards,preprocess,plot_training}.py` | 分拆至 `training/`、`evaluation/`、`agents/dqn/`、`data/`、`plotting/` | 按职责归位 |
| `Event-DQN/src/cl_predict/*` | `training/`、`evaluation/`、`data/`、`cli/` | 按职责归位 |
| `Event-DQN/src/online_anomaly_detection/*` | `et_prl/detection/`、`evaluation/` | 与 toolkit 异常检测合并 |
| `Event-DQN/src/control_evaluation/*` | `et_prl/evaluation/control/` | **评估框架**（common 公共组件 + strategies/ 各策略评估函数） |
| `Event-DQN/src/comparison/*` | `et_prl/experiments/comparison.py` | **实验编排**：调用评估框架跑多策略对比并汇总 |
| `Event-DQN/src/ablation/*` | `et_prl/experiments/ablation/` | **实验编排**：消融变体实验，复用同一评估框架 |
| `Event-DQN/src/plotting/*` | `et_prl/plotting/{dataset,training,control,ablation,methods}/` | 直接迁移 + 按数据源归入子目录 |
| `Event-DQN/src/cli/*` | `et_prl/cli/` | 直接迁移，改导入 |
| `Event-DQN/config/*` | `et_prl/config/` | 直接迁移，改导入 |

### 2.3 导入重写规则（脚本化执行）

| 原导入 | 新导入 | 数量 |
|---|---|---|
| `from ml_toolkit.X import Y` | `from et_prl.<映射目标> import Y` | 49 处（按映射表逐项替换） |
| `from src.X import Y` | `from et_prl.X import Y` | 全部 |
| `from config.X import Y` | `from et_prl.config.X import Y` | 全部 |
| `ml_toolkit.models`（toolkit 内部 1 处） | `et_prl.models` | 1 处 |
| `pyproject [project.scripts]` 中 `src.cli.*` | `et_prl.cli.*` | 16 条 |
| `pyproject [tool.hatch.build.targets.wheel]` | `packages = ["src/et_prl"]` | — |

### 2.4 参数配置方案（已定：YAML 为准 + dataclass schema 校验）

**原则：YAML 是单一事实来源。所有生效参数值必须出现在 YAML 中；
字段缺失或出现未知键时 `load_config` 立即报错（fail-fast），代码中不留"隐式默认值"。**

- **dataclass 是 schema，不是默认值载体**：每个配置类重写为 frozen dataclass，
  字段**不设默认值**（`TIMESTAMP`/`DEVICE`/路径等派生值由 `__post_init__`/`@property`
  计算）；方法（`get_train_experiment_dir()`、`to_dict()` 等）保留；
  `config.FIELD` 实例访问语义不变，调用点基本无需改动；
- **YAML 采用扁平结构，键与 dataclass 字段名一一对应**（与现类属性名一致，
  迁移映射与快照 diff 最直观）；配置规模增长后可再引入分节嵌套
- **`extends` 三层继承解决样板问题**（默认值由 YAML 承载，而非代码）：
  - `configs/base.yaml`：全量字段规范（"默认值"的载体，兼作参数文档）
  - `configs/<experiment>.yaml`（`lstm`/`dqn`/`gate`）：`extends: base` + 只写覆盖项
  - `configs/ablation/<variant>.yaml`：`extends: dqn` + 只写消融开关（如 `wo_local_threshold: true`）
  - 合并顺序 base → experiment → variant，深合并（dict 合并，标量/列表整体覆盖）
- 加载器 `et_prl/config/loader.py`：`load_config("dqn") -> DQNConfig`
  1) 解析 extends 链并深合并；2) 对照 dataclass schema 严格校验：缺字段 → 报错
  （列出全部缺失字段）、未知键 → 报错（防拼写错误）；3) 构造 frozen dataclass
- **取消空构造入口**：`LSTMConfig()` 不再可用，CLI `--config` 为必选参数
  （或默认 `default.yaml`）；测试使用 `tests/configs/` 下的最小 YAML
- **前置审计（P4 第一步）**：grep 全库找出**类级别**直接引用
  （如 `LSTMConfig.RANDOM_STATE` 而非 `config.RANDOM_STATE`）与 `config.X = ...`
  运行期改写的调用点，逐一改为实例访问或 YAML 覆盖；
- 快照机制不变：`to_dict()` 写全量生效值（即合并后结果），与输入 YAML 一致，
  保证结果可追溯；
- 路径修正：`DATA_ROOT`/`LOG_ROOT_DIR` 从 CWD 相对改为**项目根相对**
  （环境变量 `ET_PRL_ROOT` 或 `Path(__file__).resolve().parents[2]`），消除对运行目录的依赖。

> 注意：这是本次迁移中改动面最大的部分（6 个配置类 + 所有实例化点），
> 因此 P4 安排在 P3（纯移动）之后、P6（回归验证）之前，且 P6 必须覆盖配置重写的影响。

### 2.5 实验结果组织

- `logs/` 更名 `outputs/`，运行目录结构不变：`outputs/runs/<experiment>/<mode>/<timestamp>/`
- 新增 `outputs/figures/`：绘图脚本统一输出到这里（论文图再从 `docs/pics/` 归档）
- `outputs/` 整体 gitignore；`outputs/README.md` 说明目录约定
- 旧 `logs/`（26.4 MB）复制为 `outputs/runs/` 存量，checkpoint 可被新代码直接加载
  （模型结构不变，state_dict 兼容）

### 2.6 CLI 入口（已定：前缀改为 etprl-*）

- 16 个入口保留，函数路径改为 `et_prl.cli.*`
- 命令前缀 `event-dqn-*` → `etprl-*`，如 `etprl-train-dqn`、`etprl-test-event-driven`
- 所有入口新增统一 `--config` 参数（必选，或默认 `default.yaml`；指向 `configs/` 下的预设名或 YAML 路径）

### 2.7 基础设施重构（迁移时做，业务逻辑不动）

**原则**：业务代码（agent/trainer/evaluator/detection 等）纯移动不改；
日志/实验记录等基础设施趁迁移重构，换掉手动实现的薄弱部分。

#### 2.7.1 MetricsRecorder 重构（核心改动）

**已核实的调用事实**（决定安全边界）：
- `save_metrics()` 在全部 6 个调用点**均为训练/评估结束时一次性调用**，
  不存在逐 epoch 追加 → 现实现无性能问题，`metrics.json` 格式可自由调整
- `load_metrics()` **全项目零调用**；`metrics.json` 无任何程序化读取方
  （`lstm_test.py` 读的是 `EVALUATION_METRICS_FILENAME`，恰同名但只写不读）
- `training_history.csv` 是**唯一被绘图脚本消费的产物**
  （`plot_training.py` 依赖列名：`epoch`/`train_reward`/`train_avg_comfort_score`/
  `train_avg_energy_score`/`train_loss`/`train_steps`/`val_reward`/`val_steps`），
  **列名必须逐字保持**

现实现真正的问题：
- `save_training_history()` 硬编码 key 名（`train_loss`/`train_metrics`/`val_metrics`...）
  拼 CSV，指标结构一变就要改记录器；且 DQN trainer 被迫在 `_save_training_history`
  里手动把 `list[dict]` 重排成这个硬编码结构再传入（`trainer.py:286`）

重构方案（**已实现**）：

> 实现时两处修正（相对本计划初稿）：
> ① `tensorboard` 是**独立依赖**（torch 不强制安装），已加入 `pyproject.toml`
>    （`tensorboard>=2.16.0`）；`MetricsRecorder` 内 SummaryWriter 做 try-import 降级
>    （TB 缺失时静默跳过，不影响 CSV）双保险。
> ② 采用**批量写 TB**（`save_training_history` 末尾一次性写）而非逐 epoch `add_epoch`，
>    使 trainer 训练循环**零改动**，更符合"业务代码保持原样"总原则。

| 产物 | 现状 | 重构后 |
|---|---|---|
| `training_history.csv` | 通用记录器硬编码 key 名拼列 | **列名逐字不变**（已验证 `epoch/train_loss/train_reward/train_steps/val_reward/val_steps`，绘图兼容）；逻辑保留，末尾追加 TB 写入 |
| TB event 文件 | 无 | **新增**：`save_training_history` 末尾 `_log_records_to_tb` 批量 `add_scalar`（跳过 NaN/inf），`tensorboard --logdir outputs/runs/` 看曲线 |
| `metrics.json` | 结束时一次性写入 | 保持原样（已核实无读取方） |

实现要点（`et_prl/utils/metrics_recorder.py`）：
- `MetricsRecorder` 保留同名同接口（`save_metrics`/`save_training_history`/`load_metrics`）
- `__init__` 新增 `_init_tb_writer()`（try-import SummaryWriter，失败返回 None）+ `close()`
- `save_training_history` 写完 CSV 后调 `_log_records_to_tb(records)` 批量写 TB
- `ExperimentContext` 新增 `close()` 方法（调 `metrics_recorder.close()`）
- **trainer 调用点零改动**（DQN trainer 的 `_save_training_history` 手动重排保留，
  因其产出的 dict 格式正是记录器消费的格式）

#### 2.7.2 Logger 修复（小改动）

- logger name：`f"exp_{dir.name}"` → `f"et_prl.{uuid4().hex[:8]}"`
  （消除同名实验目录共享 logger 导致的日志串流）
- 去掉 `handlers.clear()` hack：改为 `if not logger.handlers` 幂等初始化
- 格式、级别策略（console INFO / file DEBUG）保持不变

#### 2.7.3 ExperimentContext 调整

- `create_experiment_context()` 新增 `SummaryWriter(experiment_dir)` 注入
  （`ExperimentContext.writer` 字段），`close()` 方法负责 `writer.close()`
- 所有训练/评估入口在结束时调用 `context.close()`（P3 迁移时随调用点一起加）

#### 2.7.4 保持原样

- `ConfigManager`：薄封装功能正确；P4 配置重写后由新 loader 自然替代
- `CheckpointManager`：标准 `torch.save`/`load` 模式，无问题
- `reproducibility.py`（种子设置）、`visualizer.py`：保持原样

---

## 三、分阶段执行步骤

### P0 准备（~1–2 h，含历史迁移）
1. **迁移 Git 历史**（已定：保留 Event-DQN 历史）：
   - `git clone --no-checkout D:\Code\Projects\Event-DQN D:\Code\Projects\ET-PRL`
     （复用全部提交历史，工作区为空）
   - 删除 `.git` 外残留后，把当前 `MIGRATION_PLAN.md` 之外的骨架文件放入；
     首次提交前删除 Event-DQN 时代不再需要的文件（P2/P3 会整体替换 `src/`、`config/`）
   - 备选：`git filter-repo --to-squash` 可把旧历史压成单个初始提交（若希望历史更干净）
2. 写入 `.gitignore`（沿用 Event-DQN 版，`/data/`、`/outputs/`、缓存等）
3. 复制 `.python-version`（3.13）
4. 复制 `data/`、`docs/`、旧 `logs/`（→ `outputs/runs/`）——已定：全量复制
5. 复制 `AGENTS.md`、`.github/`（copilot-instructions、instructions、skills）
- **验收**：`git log` 可见 Event-DQN 历史；`git status` 中 data/outputs 被正确忽略

### P1 脚手架（~1 h）
1. 新建 `pyproject.toml`：name=`et-prl`，合并依赖（见 1.3），删除 ml-toolkit git 依赖与
   `[tool.uv.sources]`，保留清华 index，`[project.scripts]` 按 2.6 更新
2. 创建 `src/et_prl/` 包骨架（全部子包 `__init__.py`）
3. `uv sync` 验证环境
- **验收**：`uv run python -c "import et_prl"` 成功

### P2 迁移 ml-toolkit + 基础设施重构（~3–4 h）
1. 按映射表复制文件到 `et_prl/{models,agents,environments,training,evaluation,data,detection,calibration,utils}`
2. 修正唯一 1 处内部自引用 + 各 `__init__.py` 的 re-export
3. 合并 `anomaly_detection` 与 `data_processing` 与 Event-DQN 对应模块的命名冲突
   （`streaming_*` 文件归 `detection/`，`dataset_loader` 等归 `data/`）
4. **基础设施重构**（详见 2.7，**已完成**）：
   - `MetricsRecorder` → 新增 SummaryWriter（批量写 TB，trainer 零改动）；
     `training_history.csv` 列名逐字保持（已验证）；`tensorboard` 加入依赖
   - `Logger` → 修复 logger 命名（UUID 替代目录名）、去掉 `handlers.clear()` hack
   - `ExperimentContext` → 新增 `close()` 方法
   - `ConfigManager`/`CheckpointManager` → 保持原样（P4 后 ConfigManager 被新 loader 替代）
- **验收**（**已通过**）：`compileall` 通过；11 个 toolkit 包 import 冒烟通过；
  功能测试确认：TB event 文件生成 + `training_history.csv` 列名与旧版一致
  + `metrics.json` 正常写入 + Logger UUID 命名 + `close()` 正常

### P3 迁移 Event-DQN 业务代码（~3–4 h）
1. 按映射表复制 `src/dqn`、`src/cl_predict`、`src/online_anomaly_detection`、
   `src/comparison`、`src/control_evaluation`、`src/ablation`、`src/plotting`、`src/cli`、`config`
2. `src/plotting` 的 15 个脚本按数据源归入子目录（`dataset/`、`training/`、`control/`、
   `ablation/`、`methods/`），修正包内互引（`plot_t_chws_comparison` → 同包相对导入）
3. 脚本化重写导入（2.3 规则），`git diff` 逐文件审查
4. `compileall` + 全模块 import 冒烟
- **验收**：`import et_prl.cli.*` 全部成功，无残留 `ml_toolkit`/`from src`/`from config` 引用
  （grep 验证为 0）

### P4 配置层重写：YAML 为准 + dataclass schema（~4–6 h，本次最大改动）
1. **前置审计**：grep 全库找出类级别直接引用（`LSTMConfig.X`）与运行期改写
   （`config.X = ...`）的调用点，列出改造清单
2. 6 个配置类重写为 frozen dataclass（**字段无默认值** + 方法保留 + 派生值改 property/`__post_init__`），
   `CommonConfig` → `config/base.py` 的 `BaseConfig`
3. 实现 `et_prl/config/loader.py`（extends 深合并 + 严格校验：缺字段报错 / 未知键报错）
4. 编写 `configs/base.yaml`（全量规范）+ `configs/{lstm,dqn,gate}.yaml`
   + `configs/ablation/*.yaml`（extends 继承）
5. 路径改为项目根相对
- **验收**：`load_config("dqn")` 返回完整 dataclass；缺字段/未知键均报错；
  `to_dict()` 快照与输入 YAML 一致；实验目录生成在 `outputs/runs/` 下；类级别引用 grep 为 0

### P5 CLI 与入口验证（~1 h）
1. `uv sync` 后逐个跑 `uv run etprl-* --help`（16 个入口）
2. 每个入口做最小冒烟：LSTM 训 1 epoch、DQN 训 1–2 episodes、一次 gate prewarm、一次绘图，
   均通过 `--config` 传入对应预设
3. 绘图脚本路径专项：逐一检查 15 个脚本的数据读取路径（`logs/` → `outputs/runs/`）、
   硬编码绝对路径与时间戳默认值，输出统一指向 `outputs/figures/`
- **验收**：16 个入口全部可执行，产物落在 `outputs/runs/` 正确结构；绘图脚本在换目录运行后仍可出图

### P6 回归验证（~2 h）
1. 固定随机种子，在**旧项目**与**新项目**各跑同一实验（LSTM 训练 + DQN 训练 + 事件驱动测试）
2. 对比 `metrics.json` 关键指标（MAE/RMSE、验证奖励、动作频率）应在浮点误差内一致
3. 用新代码加载旧 checkpoint（`best_model.pth`）跑一次评估，确认权重兼容
4. 新旧 `config.yaml` 快照逐项 diff，确认 dataclass 重写未改变任何生效值
- **验收**：指标一致；checkpoint 可加载；配置快照无差异

### P7 文档与收尾（~1 h）
1. 重写 `README.md`（快速开始、目录约定、实验产物说明）
2. 更新 `AGENTS.md`、`structure.md`、`.github/copilot-instructions.md` 到新架构
3. `outputs/README.md` 说明结果目录约定
4. 旧项目 `Event-DQN`、`ml-toolkit` 保留原样作为只读备份（不删除），README 顶部加归档说明
- **验收**：文档与新结构一致；`git log` 干净的首次提交

**预计总工作量：2–3 天**（P4 配置重写为最大头；不含 P6 中完整训练的时间）

---

## 四、风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 49 处导入重写遗漏/错映射 | 运行时 ImportError | 脚本化替换 + `compileall` + 逐模块 import 冒烟 + grep 残留检查 + git diff 审查 |
| 配置类方法签名在归位时被改坏 | 训练/评估流程断裂 | P3 只做"移动 + 改导入"，不做逻辑重构；重构留到整合后单独进行 |
| 配置类重写为 dataclass 时遗漏类级别引用/运行期改写 | 运行时 AttributeError/TypeError | P4 前置审计（grep `Config\.` 类级访问与 `config\.\w+ =` 赋值）+ P6 回
| extends 合并语义（list vs dict）产生隐蔽 bug | 配置被静默覆盖 | loader 仅支持 dict 深合并，标量/列表整体覆盖；为合并行为补充单测 |归覆盖 |
| dataclass 派生字段（TIMESTAMP/DEVICE/路径）语义漂移 | 实验目录/设备选择错误 | 派生值统一改 property/`__post_init__`，P4 验收中用 `to_dict()` 快照与旧配置逐项比对 |
| `data/`、`logs/` 相对路径依赖 CWD | 换目录运行即失败 | P4 统一改为项目根相对路径 |
| 旧 checkpoint 不兼容 | 论文结果无法复现 | P6 显式验证加载；模型结构迁移中不改层名/维度 |
| `src/dqn/agent.py` 与 toolkit `DQNAgent` 继承关系 | 合并后命名混乱 | 保留 base/子类两层，命名 `base.py`/`agent.py`，继承关系不变 |
| 绘图脚本硬编码路径（读 `logs/`、绝对路径 `d:/code/projects/Event-DQN`、具体实验时间戳 `20260402_222140`） | 换目录/换机器即失败 | P5 逐一替换为 `outputs/runs/` + 项目根相对路径；时间戳默认值改为"自动选最新实验目录"（现有 `--experiment_dir` 机制已支持） |
| `plot_ablation_pareto.py` 等脚本硬编码论文指标数字 | 结果更新后图不更新，可复现性隐患 | 迁移时保持原样（不阻塞）；迁移后改进为从 `outputs/tables/*.csv` 读数字，列入后续待办 |
| 两项目异常检测模块文件重名 | 复制覆盖 | P2 先做文件名冲突清单，逐一确认归属 |
| training_history.csv 列名被绘图脚本依赖 | 重构后列名变化导致出图失败 | 列名逐字保持；P2 已验证列名与旧版一致 |
| tensorboard 是独立依赖（torch 不强制安装） | 未安装时 TB 写入失败 | 已加入 pyproject 依赖；MetricsRecorder 内 try-import 降级（TB 缺失仅跳过 TB，CSV 不受影响） |
| Logger 用目录名做 logger name | 同名实验共享 logger，日志串流 | 改为 UUID 命名 |

---

## 五、决策记录（2026-09-14 已确认）

1. **配置方案**：YAML 为准（缺字段/未知键立即报错）+ dataclass schema 校验（字段无默认值）
   + extends 三层继承 → 见 2.4，P4 实施
2. **旧数据/结果**：全量复制（data/ 32.5 MB + logs/ 26.4 MB → outputs/runs/）
3. **CLI 前缀**：`event-dqn-*` → `etprl-*`
4. **Git 历史**：保留 Event-DQN 历史（clone --no-checkout 方式），旧仓库保留作只读备份
5. **plotting 组织**：顶层独立入口层（不并入 experiments/ 或 cli/），内部按数据源分子目录
   （dataset/training/control/ablation/methods）——2026-09-14 自主决策，待用户确认
