# 文献调研：自迭代 RL × 冷站控制 × 阈值自适应

> 新论文（next-paper 分支）专属文献笔记。
> 调研日期：2026-09-22。检索方式：arXiv API（含 2025–2026 最新工作），覆盖四个方向：自改进 RL、冷站/建筑 RL 控制、非平稳 RL、自适应阈值。
> 状态：v8（2026-09-28 第五节重构为"会前准备计划"：分档 🔴会前必须/🟠强烈建议/🟡可进行中/⚪会后，含 5 个方法设计问题、4 个开放问题；2.1–2.5 表统一为五列）。

## 一、研究方向与检索方法

**方向**：自迭代（self-iterative）让 RL agent 自我进化，两个应用模式——
- **模式 A（直接控制）**：策略自进化，闭环经验累积 + 周期性更新
- **模式 B（调整阈值）**：门控双阈值在线自适应，区别于旧论文（ET-PRL）的离线贝叶斯调参

**检索词**（arXiv API）：
- `self-improving` + `reinforcement learning`（250 篇）
- `chiller` + `reinforcement learning`（6 篇）、`data center cooling` + `reinforcement learning`（7 篇）
- `non-stationary` + `reinforcement learning` + `adaptive`（212 篇）
- `anomaly detection` + `adaptive threshold`（25 篇）
- `event-triggered` + `reinforcement learning`（40 篇）

**检索词**（Semantic Scholar bulk API，2026-09-28 补充，14 组，本地按 2022–2026 过滤）：
- `chiller AND "reinforcement learning"`（72）、`"chiller plant" AND "reinforcement learning"`（17）、`"cooling tower" AND "reinforcement learning"`（18）
- `non-stationary AND "reinforcement learning" AND building`（58）、`"reinforcement learning" AND building AND HVAC`（386）
- `event-triggered AND "reinforcement learning" AND building`（15）、`adaptive threshold AND "reinforcement learning" AND anomaly`（71）
- 另加 web 检索（Google Scholar 式）覆盖 ASHRAE / Applied Energy / Building and Environment / Energy and Buildings

## 二、各方向现状

### 2.1 自改进/自迭代 RL（方法侧，最拥挤）

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴 |
|---|---|---|---|---|
| RLoop (arXiv 2511.04285, 2025.11) | arXiv（未发表） | LLM 推理自迭代中的遗忘与过特化 | 迭代策略初始化：RL 探索 → 筛选成功轨迹 → RFT 重训 → 下一轮 | 最接近"自迭代"表述；**外层环结构模板** |
| SELFI (arXiv 2403.00991, CoRL 2024) | CoRL 2024 | 在线机器人学习不稳定 | 离线预训练 + 在线经验微调，离线目标稳定在线学习 | 模式 A（离线→在线迭代）直接参照 |
| MEDAL++ (arXiv 2303.01488, 2023) | arXiv | 端到端自改进 | 自主练习 + 从演示推断奖励 | 机器人域，机制不同，仅旁证 |
| ProteinZero (arXiv 2506.07459, TMLR 2026) | TMLR 2026 | 持续自改进的漂移与模式坍缩 | 在线 RL 持续自改进 + KL 正则防漂移 + 多样性正则防坍缩 | **防遗忘/防漂移设计可借鉴** |
| SIBRE (arXiv 2004.09846, 2020) | arXiv | 自改进奖励如何设计 | 奖励相对自身历史改进（self-improvement-based rewards） | 奖励设计思路 |
| SAGE (arXiv 2512.17102, 2025.12) | arXiv | 技能自进化 | 技能库 + 顺序 rollout 的自进化 agent | LLM agent 域，旁证 |
| SERL (arXiv 2511.07922, AAAI 2026) | AAAI 2026 | 自奖励机制 | 模型自当 actor + judge 的自检验 RL | LLM 域，自奖励参照 |

**判断**：方法侧拥挤且 2025–2026 是热词，"self-improving / self-iterative" 撞名严重（RLoop/SERL/SIRLC/SiMT 等）；但**全部在 LLM/机器人/蛋白设计域，没有建筑能源控制**。

### 2.2 冷站/建筑 RL 控制（应用侧，arXiv 少、期刊多）

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴 |
|---|---|---|---|---|
| Zhan et al. (arXiv 2501.15085, **ICLR 2025**) | ICLR 2025 | 数据中心冷却控制 | **离线 RL**：GNN 物理感知模型 + latent 空间策略；生产 DC 部署 2000h，省 14–21% | **应用侧强基线，必引** |
| CQD-ERL (arXiv 2608.11324, 2026.08) | arXiv | 热带水冷冷站群非平稳 | 质量-多样性进化 + 按天气/负荷工况的策略档案（archive）+ 安全盾 | 用"多策略档案"应对非平稳，**与单策略自迭代形成对比点** |
| Chiller+TES 共设计 (arXiv 2601.22880, 2026.01) | arXiv | 冷机+蓄能共设计 | DQN 控制冷机 PLR + 30 年生命周期成本共设计 | 动作空间设定最接近 |
| 澳洲大学校园冷机 (arXiv 2511.14160, 2025.11) | arXiv | 真实校园冷机节能 | PPO + 滚动时域 + 约束优先奖励，省 28% | 真实案例参照 |
| Guo et al. (arXiv 2203.07500 / 2310.03814) | arXiv | 区域供冷控制 | Q-learning vs MPC，省 8–17% | 经典基线 |
| Wong et al. (arXiv 2209.08112, 2022) | arXiv | 工业 HVAC 多时间尺度控制 | 分层 RL（多时间尺度动作） | 动作分层参照 |
| DCVerse (arXiv 2604.07559, 2026.04) | arXiv | 数据中心数据稀缺 | 数字孪生 + DRL 策略池双环控制 | 用"孪生重训"应对数据稀缺，与闭环自迭代对比 |
| IBM RL Testbed (arXiv 1808.10427) | arXiv | 数据中心冷却 RL 测试床 | EnergyPlus 数据中心冷却 RL 测试床 | 经典 |

**判断**：应用侧全是**静态策略**（离线训练一次部署）；非平稳靠多策略档案（CQD-ERL）或数字孪生重训（DCVerse）解决，**没有闭环自迭代**。

### 2.3 非平稳 RL（理论侧，成熟）

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴 |
|---|---|---|---|---|
| 高效重启 (arXiv 2510.11933, 2025.10) | arXiv | 非平稳下重启策略的缺陷 | 指出 RestartQ-UCB 两大缺陷：**完全遗忘**（重启丢全部历史）与**定时重启**（不看策略与环境是否失配）；提出部分/自适应/选择性重启，动态遗憾降 91% | **迭代单元设计的直接约束**：每次迭代不能全量重训丢历史，也不能固定周期盲目重启 |
| FANS-RL (arXiv 2203.16582, NeurIPS 2022) | NeurIPS 2022 | 非平稳环境适应 | 因果分解的非平稳适应（潜变量变化因子） | 理论参照 |
| BORL (arXiv 2006.14389, ICML 2020) | ICML 2020 | 漂移非平稳 MDP 调参 | 无变化预算的自适应调参 | 理论参照 |
| ESN 在线适应 (arXiv 2602.06326, 2026.02) | arXiv | 轻量在线适应 | reservoir + RLS 在线适应模块，免反传 | 部署侧轻量方案参照 |
| PPO 自适应滤波 (arXiv 2506.06323, 2025) | arXiv | 非平稳噪声 | RL 直接调滤波器系数 | "RL 调参数"的邻近范式 |

### 2.4 自适应阈值（模式 B 侧）

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴 |
|---|---|---|---|---|
| Tri-CRLAD (arXiv 2405.06925, 2024) | arXiv | 半监督传感器异常检测 | 因果 RL + **自适应阈值平滑调整** + 自适应决策奖励 | **模式 B 最近邻，必须精读区分** |
| ATH (arXiv 2308.10504, 2023) | arXiv | KPI 异常检测 concept drift | 周期性 + 异常比例的自适应阈值 | 规则式强基线 |
| ReRe (arXiv 2004.02319, 2020) | arXiv | 实时流式异常检测 | LSTM + 双自适应用阈值 | 门控结构近亲 |
| ADALog (arXiv 2505.13496, 2025) | arXiv | 日志异常检测 | 正常数据上的自适应分位阈值 | 阈值校准思路 |

**判断**：自适应阈值全在异常检测域，**没有作为 RL 控制回路的一部分在线调整**。

### 2.5 事件触发 RL（旧论文 ET-PRL 根基）

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴 |
|---|---|---|---|---|
| Baumann et al. (arXiv 1809.05152, 2018) | arXiv | 事件触发控制 | DRL + 事件触发控制 | 开创工作 |
| Adaptive ETRL (arXiv 2409.19769, 2024.09) | arXiv | 非平稳策略自适应 | 联合学习控制策略与触发策略 | 触发策略与策略联合学习 |
| ET-MAPG (arXiv 2509.20338, 2025.09) | arXiv | 多 agent 事件触发 | 多 agent 事件触发策略梯度 | 多 agent 版本参照 |
| 建筑微气候事件触发 RL (arXiv 2001.10505, **Applied Energy 2020**) | Applied Energy 2020 | 建筑微气候事件触发 | SMDP 框架 + 事件触发学习/控制 | 建筑域直接参照 |
| eMPC+DRL (arXiv 2208.10302, 2022) | arXiv | 事件触发时机 | RL 学事件触发策略（自动驾驶） | 触发时机学习参照 |

**判断**：旧论文已站在这条线上；新论文要往前推"**触发阈值本身可进化**"，这是与上述工作的差异点。

### 2.6 补充检索：期刊文献（2026-09-28，Semantic Scholar bulk API + web 检索）

> 方法：Semantic Scholar bulk search（14 组关键词，含 venue/年份本地过滤，2022–2026）+ web 检索（Google Scholar 式）。
> 说明：S2 对 ASHRAE 期刊覆盖弱（ASHRAE Journal 论文基本未收录），ASHRAE 侧靠会议论文（ASHRAE Annual / IBPSA）补位。

#### 2.6.1 冷站/建筑 RL 控制（应用侧期刊，arXiv 漏掉的主力）

| 工作 | 期刊/会议 | 要点 | 与我们的关系 |
|---|---|---|---|
| Predictive control optimization of chiller plants based on DRL (JBE 2023, 被引 52) | J. Building Engineering | 冷站预测控制 DRL | 期刊经典基线 |
| Efficient model-free control of chiller plants via cluster-based DRL (JBE 2023, 被引 26) | J. Building Engineering | 聚类分工况的 model-free DRL | 分工况思路，与"多策略档案"同族 |
| Leveraging RL for optimal control of chiller plant with complex hydraulic and thermodynamic characteristics (EAB 2025, 被引 13) | Energy and Buildings | 复杂水力/热工特性冷站 RL 最优控制 | 真实冷站建模参照 |
| Knowledge-guided hierarchical RL for robust operational control of chiller plants: field deployment in a hotel (EAB 2026) | Energy and Buildings | 知识引导分层 RL，酒店现场部署 | **分层 + 现场部署，强参照** |
| Systematic evaluation of DQNs for multi-chiller system RL control (ATE 2025) | Applied Thermal Engineering | 多冷机系统 DQN 系统评估 | 动作/算法设定参照 |
| Multi-agent RL for chiller system prediction and energy-saving in semiconductor manufacturing (IJPE 2025) | Int. J. Production Economics | 半导体厂冷机群多 agent RL | 工业冷站参照 |
| Multi-agent RL control for large chiller plant (IBPSA BS 2025) | Building Simulation 2025 | 广州商业综合体 8 个月现场，比规则省 10.6%、比单 agent 省 5% | **真实长期运行参照** |
| RL-based optimal control of condenser water system in chiller plant (ASHRAE Annual 2025, PX-25-C024) | ASHRAE | 冷凝水系统 Q-learning，省 14.7–19.4% | ASHRAE 侧代表 |
| Multi-Agent Optimal Control for Central Chiller Plants Using RL and Game Theory (Systems 2023, 被引 17) | Systems | 博弈论 + 多 agent 冷站群控 | 群控机制参照 |
| Optimal Control of District Cooling Energy Plant with RL and MPC (ASME JESBC 2023) | ASME J. Eng. for Sustainable Buildings & Cities | 区域供冷 RL+MPC | 区域供冷参照 |
| Key Lessons From Improving the Energy Efficiency of Hospitals Using RL: 39 Deployments in India (SEEBE 2024) | SEEBE 2024 | 印度 39 家医院 RL 部署经验教训 | **大规模部署教训，必引** |
| Deep clustering of cooperative multi-agent RL to optimize multi chiller HVAC (JBE 2022, 被引 57) | J. Building Engineering | 深度聚类合作多 agent | 经典高引基线 |
| Toward Model-Assisted Safe RL for Data Center Cooling Control: Lyapunov-based (EECN 2023, 被引 19) | Energy-Efficient Computing & Networking | Lyapunov 安全约束 RL 数据中心冷却 | 安全机制参照 |
| LC-Opt: Benchmarking RL and Agentic AI for End-to-End Liquid Cooling Optimization in DCs (NeurIPS 2025, arXiv 2511.00116) | NeurIPS 2025 | 液冷端到端 RL/agentic AI 基准 | 新基准，数据中心侧 |
| Real-Time Operational Strategy for Ice Thermal Storage District Cooling via Model-Free Safe DRL (IEEE TSG 2026) | IEEE Trans. Smart Grid | 冰蓄能区域供冷安全 DRL | 安全 DRL 期刊代表 |
| An innovative heterogeneous transfer learning framework to enhance the scalability of DRL controllers in buildings (Building Simulation 2024, 被引 46) | Building Simulation | 异构迁移学习提升 DRL 跨建筑可扩展性 | 迁移学习路线，与自迭代对比 |
| A Safe and Data-Efficient Model-Based RL System for HVAC Control (IEEE IoT-J 2024) | IEEE Internet of Things J. | 安全 + 数据高效 model-based RL | 部署侧参照 |
| ORCHID: Offline RL for Control of HVAC in Buildings using Historical and Low-Fidelity Simulation Data (IC3SE 2024) | IC3SE 2024 | 历史数据 + 低保真仿真离线 RL | 离线 RL 路线代表 |
| Development of Chiller Plant Models in OpenAI Gym Environment for Evaluating RL Algorithms (Energies 2025) | Energies | 冷站 OpenAI Gym 环境，DQN/DDQN 省 14% | 评测环境参照 |

**判断（更新）**：应用侧期刊确认了 arXiv 侧的结论——**全部是静态策略**（离线训练一次部署），非平稳靠分工况聚类、多策略档案、迁移学习或数字孪生重训解决，**仍无闭环自迭代**。

#### 2.6.2 非平稳建筑 RL（新发现，修正 arXiv 侧"理论侧"认知）

| 工作 | 期刊/会议 | 要点 | 与我们的关系 |
|---|---|---|---|
| Deep RL Control for Non-stationary Building Energy Management (EAB 2022) | Energy and Buildings | 非平稳建筑能源管理 DRL 控制 | **非平稳建筑 RL 期刊代表** |
| Towards optimal HVAC control in non-stationary building environments combining active change detection and DRL (BAE 2022, vol 211, 108680；清华 Deng/Zhang/Qi) | Building and Environment | **主动变化点检测** + 每个工况上下文单独学 DRL 策略，检测到漂移即切换 | **非平稳侧最近邻**：多模型切换，与 CQD-ERL 同族；非闭环自迭代 |
| Continual adaptation in DRL-based control applied to non-stationary building environments (2020) | 1st Workshop on RL for Energy Management in Buildings & Cities | 非平稳建筑环境 DRL 持续适应 | 早期工作 |
| A Model-free RL Approach for the Energetic Control of a Building with Non-stationary User Behaviour (2020) | Smart Grid and Smart Cities | 非平稳用户行为 model-free RL | 早期工作 |
| Non-Stationary Policy Learning for Multi-Timescale Multi-Agent RL (CDC 2023) | IEEE CDC | 多时间尺度多 agent 非平稳策略学习 | 理论参照 |
| Meta-RL with Shared Representations Enables Fast Adaptation in Energy Systems (PAKDD 2026) | PAKDD 2026 | 共享表征 meta-RL 加速能源系统适应 | meta-RL 路线参照 |

**判断（更新）**：非平稳建筑 RL 在期刊有 2020–2022 的明确工作，主流做法是**变化点检测 + 多模型切换**（或 meta-RL 快速适应），**没有闭环自迭代**；这给"为什么不直接变化点检测 + 重训"的审稿人问题提供了必须回应的最近邻。

#### 2.6.3 自迭代/再学习/持续学习（新发现，模式 A 最近邻）

| 工作 | 期刊/会议 | 要点 | 与我们的关系 |
|---|---|---|---|
| An End-to-End Relearning Framework for Building Energy Optimization (Energies 2025, 18(6):1408；Naug/Quinones-Grueiro/Biswas) | Energies | RL 控制器**自监控 + 自适应**，性能退化时触发再学习策略，应对设备老化/使用模式变化 | **模式 A 应用侧最近邻**：再学习是周期性/退化触发，非经验闭环自迭代，且不调阈值 |
| Building a Self-Evolving Digital Twin System with Bayesian Optimization and DRL for Complex Equipment Optimization and Control (Tsinghua Sci. Technol. 2026, 31(1):199–216；Wang/Chen/Zhang/Obaidat/Cui/Cheng/Lu) | Tsinghua Science and Technology | "自进化数字孪生"：贝叶斯更新持续融合实时数据 + SAC agent 在线更新 | **命名撞车**（self-evolving）+ 孪生在线更新机制参照 |
| Continual RL for HVAC Systems Control: Integrating Hypernetworks and Transfer Learning (arXiv 2503.19212, 2025；Bekal et al.) | arXiv | 超网络建模不同 HVAC 环境动力学，任务序列持续学习 + 迁移，抗灾难性遗忘 | 持续学习路线：任务序列，非在线工况漂移 |
| Efficient and assured RL-based building HVAC control with heterogeneous expert-guided training (Sci. Rep. 2025, 15:7677；Xu/Fu/Wang/Yang/Huang/O'Neill/Wang/Zhu) | Scientific Reports | 物理模型 + 历史数据 + 专家规则异构引导加速在线 DRL | 在线 DRL 加速路线参照 |
| A Relearning Approach to Reinforcement Learning for Control of Smart Buildings (PHM 2020) | PHM Society | 智能建筑 RL 再学习 | 早期再学习工作 |

**判断（更新）**：建筑域已有"再学习框架"（Energies 2025）与"自进化数字孪生"（清华 2026），说明**"在线更新策略"本身不再是空白**；我们的差异点必须落在"**经验闭环自迭代 + 策略与触发阈值共同进化**"，且"self-evolving"命名撞车升级。

#### 2.6.4 自适应阈值（新发现，模式 B 最近邻）

| 工作 | 期刊/会议 | 要点 | 与我们的关系 |
|---|---|---|---|
| A physics-guided self-adaptive chiller sequencing controller of enhanced robustness and energy efficiency accommodating measurement uncertainties (Applied Energy 2025, 389:125718；Zou/Li/Gao/Wang) | Applied Energy | FDD 监督器识别冷机启停故障类型 + **自适应调整冷机切换阈值**，鲁棒性提升、节能至 7.46% | **模式 B 冷站侧最近邻**：阈值自适应真实存在且有效，但是物理引导规则式，非 RL 回路 |
| Agent-based dynamic thresholding for adaptive anomaly detection using RL (Neural Computing & Applications 2024) | NCA | agent 式动态阈值 + RL 自适应异常检测 | 阈值 + RL 组合（异常检测域） |
| ADT: Agent-based Dynamic Thresholding for Anomaly Detection (arXiv 2023) | arXiv | agent 动态阈值异常检测 | 同上 |
| Semi-supervised Anomaly Detection via Adaptive RL-Enabled Method with Causal Inference for Sensor Signals (arXiv 2024) | arXiv | 自适应 RL + 因果推断传感器异常检测 | 传感器信号侧 |
| Predictive Maintenance and Cooling Demand Forecasting in Data Centre Thermal Control: Hybrid RL and AI Framework (ICOSAAS 2026) | ICOSAAS 2026 | 数据中心热控预测性维护 + 冷却需求预测混合 RL | 数据中心侧 |

**判断（更新）**：自适应阈值在**冷站启停控制**（Applied Energy 2025）已有规则式实现并验证有效，在异常检测域有 RL 动态阈值；**RL 控制回路内在线自迭代调整事件触发阈值**仍是空白，但"阈值自适应"本身的价值已被同行认可（降低被质疑风险）。

#### 2.6.5 事件触发 RL（确认）

- Applied Energy 2020（arXiv 2001.10505）确认被引 48，仍是建筑微气候事件触发 RL 的开创工作。
- 2022–2026 建筑/冷站域**未发现新的事件触发 RL 期刊论文**（IEEE 侧的 event-triggered RL 全在非线性控制/网络域）——**事件触发 + 建筑能源仍是空白**，旧论文 ET-PRL 的根基稳固。

### 2.7 线 A 精读：RSI 环机制提炼（怎么在 ET-PRL 之上搭出自迭代环）

> 目的：不是找建筑域现成 RSI 抄（没有），而是拆 5 篇机制论文的"环怎么转"，回答 4 个设计决策，拼出冷站 RSI 环。
> 前一篇 ET-PRL 结构 = 事件触发控制（策略 π + 触发阈值 τ，τ 离线贝叶斯调参）。引入 RSI = 在外层套"部署 → 收集在线经验 → 迭代触发判断 → 更新 π/τ → 再部署"的环。

#### 2.7.1 五篇机制拆解（按"环怎么转"）

| 论文 | 环的骨架 | 关键机制 | 可移植到冷站 RSI 的点 |
|---|---|---|---|
| **RLoop** (2511.04285) | 探索 → 筛成功轨迹 → RFT 重训 → 下一轮 | 迭代策略初始化：用 Rejection-sampling Fine-Tuning 把成功轨迹炼成专家数据集，**以上一轮策略为起点**重训，保留步间策略多样性 | ① 迭代单元 = 一批"成功轨迹"；② 更新 = RFT 微调而非从头训；③ 防过特化/遗忘靠"保留多样性 + 以上轮为起点" |
| **高效重启** (2510.11933) | 检测失配 → 重启 | 指出 RestartQ-UCB 两缺陷：**完全遗忘**（重启丢全部历史）+ **定时重启**（不看策略-环境是否失配）；给部分/自适应/选择性重启，动态遗憾降 91% | **迭代触发条件**的直接约束：不能固定周期盲目迭代，必须"失配才迭代"；更新要部分式，不能全量重训丢历史 |
| **SELFI** (2403.00991) | 离线预训练 → 在线微调 | 在线 model-free RL 叠在离线 model-based 预训练之上；把**离线预训练目标锚进在线 Q 值**，稳定在线学习 | **最贴"延续前一篇"**：前一篇的静态策略 = 离线预训练产物，作为在线迭代的锚点/起点，不被丢弃 |
| **ProteinZero** (2506.07459) | 在线 RL 持续自改进 | 多目标奖励 + **KL 偏离参考模型** + **嵌入级多样性正则**（防模式坍缩） | **迭代安全性正则**：每轮更新加 KL 锚到参考策略 + 多样性项，防止自迭代把策略带崩 |
| **Adaptive ETRL / ATPPO** (2409.19769) | 联合学控制策略 + 触发策略 | **联合学习**触发条件与控制策略；把**累积奖励（整段轨迹性能）增广进状态空间** → 价值函数天然编码历史 → 无需显式学触发条件即得自适应非平稳策略 | **τ 与 π 共同进化的桥**：证明"触发条件可学"是成立的；其"累积奖励增广状态"是隐式触发机制，与 ET-PRL 的显式双阈值形成两种路线。2026-10-05 精读完成：机制四件套与三点差异、可借鉴机制、"不换底座"决策见 `第二篇-方法设计骨架.md` 2.7 节 |

#### 2.7.2 四个设计决策（RSI 环怎么定）

| 设计决策 | 选项 | 文献依据 | 推荐（延续 ET-PRL） |
|---|---|---|---|
| **① 迭代单元** | episode / 时间窗 / 步数 | RLoop 用"一批成功轨迹" | **时间窗**（如 1 天 / 1 个工况周期）：冷站有日/季节周期，时间窗是自然单元，且便于"滚动评估" |
| **② 迭代触发条件** | 固定周期 / 性能退化 / 分布漂移 | 高效重启：固定周期=盲目，必须"失配才迭代" | **事件触发（性能退化 ∨ 工况漂移）**：直接延续 ET-PRL 的事件触发哲学到外层环——这是"自迭代"区别于"定期重训"的核心 |
| **③ 更新方式** | 全量重训 / 微调 / 经验回放混合 | RLoop(RFT)+SELFI(锚定)+高效重启(部分式)：全量重训=遗忘 | **以上轮 π 为起点的微调（RFT/KL 锚定）**：保留前一篇策略，部分式更新不丢历史 |
| **④ τ 与 π** | 联合学 / 分开学 | Adaptive ETRL(联合,隐式)+Tri-CRLAD(回路内调阈值) | **显式 τ 作为可学参数，与 π 在外层环共同更新**：延续 ET-PRL 的显式双阈值结构（而非隐式状态增广），让"阈值共同进化"成为清晰贡献点 |

#### 2.7.3 拼出的冷站 RSI 环（设计骨架，待 Phase 0 验证）

```
外层环（RSI，事件触发）：
  初始化：π₀, τ₀ = ET-PRL 策略 + 离线贝叶斯阈值   ← 前一篇结果作为起点，不丢弃
  循环：
    1. 部署 πₖ + τₖ，在一个时间窗内收集在线经验
    2. [迭代触发判断]（事件触发，非固定周期）：
         性能退化？（奖励/KPI 跌破基线）
         工况漂移？（运行分布变化）
         都不满足 → 保持 πₖ+τₖ 继续部署（不迭代）
    3. 触发则更新（部分式，不丢历史）：
         筛成功轨迹 → RFT 微调 πₖ → πₖ₊₁（KL 锚到 πₖ，防漂移/过特化）
         同步更新 τₖ → τₖ₊₁（显式可学阈值，与 π 共同进化）
    4. 再部署 πₖ₊₁ + τₖ₊₁
  安全：KL 锚定 + 安全盾（参照 CQD-ERL），防止自迭代把策略带崩
```

**这个骨架同时满足**：延续前一篇（π₀,τ₀ 即 ET-PRL 产物）+ 引入 RSI（外层事件触发迭代环）+ 阈值共同进化（显式 τ 可学）+ 规避文献指出的坑（完全遗忘/定时重启/漂移/模式坍缩）。

**待 Phase 0 回答的前置问题**：冷站数据上"静态 π₀+τ₀ 随时间退化"是否真实成立——不成立则外层环无动机，需先找非平稳性来源（工况漂移 / 设备老化 / 使用模式变化，对应 Energies 2025 的退化触发）。

### 2.8 RSI 概念深挖结论（2026-09-28 讨论；交互概念卡：`next-paper/assets/rsi-concept.html`）

**2.8.1 事件触发 vs RL 的理论定位**
- 无根本冲突：RL 原则上能自己学会 hold。三种 formulation：(a) hold 作动作（MDP）；(b) 阈值 ETC（SMDP）；(c) 隐式触发（状态增广）。表达能力大致等价。
- ETC 不冗余的理由：连续动作空间下 hold 不自然；样本效率；τ 可解释、可独立学（"π–τ 共同进化"贡献点的前提）；执行成本是一等约束。
- SMDP 是最干净的理论外壳（hold 作动作 = sojourn time 恒 1 步的特例）；Applied Energy 2020（arXiv 2001.10505）用 SMDP 框架，可引。
- 关键动作：把 τ 做成**显式可学参数**（而非隐式状态增广），是共同进化贡献点的前提。

**2.8.2 RSI 与训练阶段的关系（两层更新）**
- RSI 不参与"怎么训"（轮次内 = 标准 RL，不变）；它是决定"何时训、用什么训、怎么初始化"的元机制 / 生命周期协议。
- 第一层（轮次内）：标准梯度训练，不变；第二层（轮次间）：事件触发 + 热启动 + 在策略数据。
- 原 ET-PRL"训练→部署→冻结"一次性生命周期 → 重复生命周期；触发是"按需"（事件触发），非"始终在线"。

**2.8.3 加 RSI 后的训练/测试流程（对齐原 ET-PRL 三步口径）**
- 原流程：① 梯度下降训 RL agent；② 异常检测算法得阈值判断方式；③ 部署后异常检测门控决定何时触发 agent。
- 加 RSI：初始化同 ①②（π₀, τ₀）→ 自迭代环：③ 部署 πₖ+τₖ 收集经验 + 性能指标 → ④ 迭代触发判断（否→回③；是→⑤）→ ⑤ RFT 微调 πₖ→πₖ₊₁ 且 τₖ→τₖ₊₁ → ⑥ 再部署回 ③。
- 测试流程：一次性验证 → **滚动评估**（非平稳条件下 πₖ 性能随时间序列，对比静态 π₀ 基线）。
- 两个未定设计点：(a) τ 更新机制——每轮重跑异常检测重新推导 vs 做成可学参数随 RFT 更新（后者是共同进化贡献的前提）；(b) 外层触发判据——对窗口内性能残差做漂移/异常检测，与内层门控同族。

**2.8.4 价值边界（"10000 条数据"论证）**
- 若数据是静态历史且环境平稳：RSI ≈ 把一次训练拆成多次，无价值。
- RSI 价值由两个假设买来：① 非平稳性（顺序适应让最新工况赢得话语权；一次性全训 = 各工况折中模型）；② 在策略自产数据（每轮数据由模型自己动作产生，环境须能评估"模型自己动作"的奖励）。
- Phase 0 验证 ①（承重墙）；② 由环境选择决定（2.8.5）。

**2.8.5 前置条件：环境必须对模型自己的动作有响应**
- 静态历史数据序列是数据集，不是环境：奖励固定在数据里，模型动作不影响奖励。
- 四条路径：真实在线部署 / 物理仿真器（EnergyPlus 等）/ 学习型数字孪生 / 纯静态历史数据（仅保留顺序适应的弱化版）。
- 推荐路径：**滚动回放过仿真器**——历史时间线按序回放（→ 非平稳）+ agent 出动作、仿真器返奖励（→ 在策略），同时满足两个假设。
- 保真度风险 = RSI 链天花板：OOD 动作 → 仿真器预测不准 → 在错误奖励上自迭代，越迭代越偏；仿真器校验（对历史数据）/ 置信度加权是前置；接数字孪生文献线（DCVerse、清华 self-evolving digital twin）。

### 2.9 LLM 自进化环机制参照（2026-09-28 补充，源自 LLM agent 自进化论文清单）

**筛选标准**：清单主体是 LLM agent 自进化（prompt/workflow 优化、多智能体拓扑、skill 库、工具创建），优化对象是文本/图结构，不迁移到冷站控制；只保留解决**自迭代环通用机制问题**（何时触发 / 怎么更新 / 怎么防遗忘 / 跨轮次怎么算功劳 / 两个组件怎么共同进化）的论文。

**第一梯队（精读，机制可直接迁移到冷站 RSI 外层环）**：

| 论文 | arXiv | 机制 | 对应设计决策（见 2.7.2 / 2.8） |
|---|---|---|---|
| AEL: Agent Evolving Learning | 2604.21725 | 双时间尺度：快 Thompson 采样选检索策略 + 慢反思在 plateau 时"先诊断为什么退化、再开处方"注入新策略 | 外层触发判据（退化检测→定向更新）+ 内层事件触发 vs 外层 RSI 的快慢分离 |
| Do Self-Evolving Agents Forget? (CPE) | 2605.09315 | "自进化能力侵蚀"现象 + Capability-Preserving Evolution 通用稳定化原则（约束破坏性漂移，workflow/skill/model/memory 四通道验证） | 防遗忘/防模式坍缩（与 KL 锚定互补）+ "保留能力"怎么度量 |
| Skill-R1 | 2605.09359 | 双层信用分配：intra-generation（同轮 rollout 对比）+ inter-generation（奖励"让下一轮变好"的修订） | 外层环跨轮次信用分配——怎么判定迭代 k+1 比 k 好（环中尚未定的一环） |
| EvolveR | 2510.16079 | 闭环经验生命周期：离线蒸馏轨迹成可复用原则 + 在线检索 + 策略强化迭代更新 | 环结构模板（与 RLoop 同类，互为印证） |
| COSPLAY (Co-Evolving Decision & Skill Bank) | 2604.20987 | 决策 agent 与 skill bank 共同进化：一个检索使用，一个从 rollout 持续提炼更新 | π–τ 共同进化的机制参照（两组件耦合进化，与"显式可学 τ"同构） |

**第二梯队（值得读，概念/部分迁移）**：

| 论文 | arXiv | 参考点 |
|---|---|---|
| JitRL: Just-In-Time RL | 2601.18510 | 免梯度、非参数经验记忆 + KL 约束策略优化的闭式解；"低成本更新"思路（logit 调制为 LLM 特有，不直接搬） |
| Truly Self-Improving Agents (metacognition) | 2506.05109 | 把"何时学/学什么"形式化为元认知三组件（自评/规划/评估）；外层触发的概念外壳，无具体机制 |
| EvoLM | 2605.03871 | 判别式 rubric 与策略共同进化当自监督奖励；与"τ 作为评估信号与 π 共同进化"同构，可作类比引用 |

**未保留（不迁移）**：prompt/workflow 优化（DSPy / TextGrad / ProTeGi / OPRO / EvoPrompt / AFlow / ADAS / GPTSwarm / AgentSquare / MaAS / ScoreFlow）、多智能体拓扑进化（ST-EVO / MASPO / HiVA / EvoMAC / DyLAN / MAS-GPT 等）、skill 库管理与工具创建（Group of Skills / SkillOS / MemSkill / LATM / CREATOR / Alita 等）、语言反思（Reflexion / Self-Refine / Meta-Prompting）——优化对象为 LLM 文本/图结构或多智能体通信拓扑，冷站侧无对应物。

## 三、Gap 与定位

现有工作的组合方式（v2 更新）：
- 自迭代 RL → 全在 LLM/机器人/蛋白设计，**没有建筑能源控制**；建筑域出现的是"再学习框架"（Energies 2025，退化触发再训）与"自进化数字孪生"（清华 2026，孪生在线更新），均非经验闭环自迭代
- 冷站 RL → 期刊侧确认全是**静态策略**，非平稳靠"多策略档案/分工况聚类"（JBE 2023、CQD-ERL）、"变化点检测 + 多模型切换"（BAE 2022）、"迁移学习"（Building Simulation 2024）或"数字孪生重训"（DCVerse），**没有闭环自迭代**
- 自适应阈值 → 冷站侧已有**规则式**自适应阈值（Applied Energy 2025 冷机切换阈值，物理引导 + FDD）；异常检测域有 RL 动态阈值；**没有作为 RL 控制回路的一部分在线自迭代调整**
- 事件触发 RL → 建筑/冷站域 2022–2026 **无新工作**，Applied Energy 2020 仍是该交叉点唯一代表

**独占交叉点**：真实冷站数据上的闭环自迭代，同时进化**控制策略（A）和事件触发阈值（B）**；B 是旧论文已知弱点（离线调参）的自然延伸，且 Applied Energy 2025 已证明"冷站阈值自适应"本身有价值（规则式），RL 回路内在线自迭代调整阈值是空白。

**必须正面回应的最近邻（v2 新增）**：
1. Energies 2025 relearning framework——"在线更新策略"已有，差异点 = 经验闭环自迭代（非退化触发重训）+ 阈值共同进化
2. BAE 2022 变化点检测 + 多模型切换——差异点 = 单策略连续自迭代（非离散多模型切换），且阈值也进化
3. Applied Energy 2025 自适应冷机切换阈值——差异点 = RL 回路内生学习阈值（非物理规则 + FDD）

**贡献表述（暂定）**：

> 首个在真实冷站群控数据上验证的自迭代 RL 框架，通过闭环经验迭代同时适应策略与事件触发阈值，解决静态策略/静态阈值在非平稳运行工况下的退化。

## 四、风险点

1. **命名撞车（升级）**："RSI" 撞金融相对强弱指数，RL 领域已有 RLoop/SERL/SIRLC/SiMT 等缩写，且清华 2026 已用 "self-evolving digital twin"、Energies 2025 用 "relearning framework"——"self-evolving/self-improving/relearning" 在建筑能源域开始被占用。方法名**必须**走"机制描述"路线（强调 closed-loop / co-adaptation / experience-driven iteration），不用 self-* 前缀，不用缩写。
2. **审稿人必问（细化）**：
   - "为什么不直接用离线 RL + 定期重训？"——Phase 0 动机实验（静态策略随时间退化）+ 滚动评估协议回答。
   - "为什么不直接变化点检测 + 多模型切换（BAE 2022）？"——需论证单策略连续自迭代 vs 离散多模型切换的优劣（切换开销、上下文数爆炸、阈值无法随工况平滑变化）。
   - "阈值自适应已有规则式方案（Applied Energy 2025），为什么要 RL？"——需论证规则式阈值依赖故障类型先验与物理模型，RL 回路内生学习可覆盖未知工况。
   - "RSI 是不是把一次训练拆成多次？"——非平稳 + 在策略自产数据两个假设（2.8.4）+ 滚动评估对比（静态 vs 自迭代）回答。
3. **检索覆盖**：S2 对 ASHRAE Journal 覆盖弱（基本未收录），ASHRAE 侧仅能靠会议论文（ASHRAE Annual、IBPSA）补位；若需 ASHRAE Journal 全文级确认，后续可手动查 ASHRAE Digital Library。

## 五、会前准备计划（2026-09-28 定稿，目标：带着"做什么 + 怎么做"找老师讨论第二篇）

> 讨论目标：让老师 30 秒看懂**第二篇做什么**（延续 ET-PRL、引入闭环自迭代）与**怎么做**（方法骨架），并就 4 个开放问题拍板。
> 分档：🔴 会前必须 → 🟠 强烈建议 → 🟡 可进行中 → ⚪ 会后。

### 5.0 已完成（不用重做）
- 文献调研 72 篇 + Gap 定位（第三节）
- 风险点 / 审稿人必问（第四节）
- RSI 概念深挖 + 交互概念卡（2.8 + `next-paper/assets/rsi-concept.html`）
- ~~补 Google Scholar / Semantic Scholar 检索~~ **已完成（2026-09-28）**，结果见 2.6 节。

### 5.1 🔴 会前必须（否则讨论没基础）
1. **1 页定位陈述（做什么）**：把第三节"独占交叉点 + 贡献表述"浓缩成 1 页 A4——延续前一篇（ET-PRL：π+τ，τ 离线调参）→ 第二篇（闭环自迭代，π 与 τ 共同进化），并列 3 个最近邻差异（Energies 2025 / BAE 2022 / Applied Energy 2025）。
   *完成标准：老师 30 秒看懂"这篇做什么、和前一篇什么关系"。*
2. **方法设计骨架（怎么做）**：精读**线 A 7 篇 + 2.9 第一梯队 5 篇**拆机制，回答 5 个设计问题，每个给"暂定答案 + 依据"：
   - ① 外层环**何时触发**一次迭代
   - ② 每次迭代**怎么更新**（部分/选择性，不全量重训）
   - ③ 怎么**防遗忘/防漂移**（KL/正则）
   - ④ **跨轮信用分配**（哪轮经验归因给哪个更新）
   - ⑤ **π 与 τ 怎么共同进化**（同时 vs 交替）
   *完成标准：一张方法骨架图 + 5 问暂定答案。这是老师会重点挑战的部分。*
3. **开放问题清单**：3–5 个真正需要老师拍板的问题（见 5.5）。

### 5.2 🟠 强烈建议会前完成（让讨论有数据支撑）
4. **Phase 0 动机初步结果**：静态 π₀+τ₀ 在新数据上是否随时间退化？（不成立则 RSI 动机不成立，需重新找非平稳性来源）
   *完成标准：至少一条粗糙退化曲线，或"数据还差哪一步"的明确结论。*
5. **仿真器/数字孪生可行性结论**（RSI 价值链前置条件，见 2.8.5）：环境须能对模型自己的动作给出奖励。
   *完成标准：用哪个仿真器、保真度够不够、推荐路径（滚动回放过仿真器）。*

### 5.3 🟡 会前可"进行中"（带进展去讨论即可）
6. 数据清洗：按 `data/external/csge_chiller_competition/TODO.md` 推进（权限确认 → 选目标项目 train_A/train_B → 清洗 → 接入管线）
7. 线 B related work 精读（写论文用，非核心决策）

### 5.4 ⚪ 会后
8. 方法命名：查完文献后定名（避开 RSI 及已有缩写，走"机制描述"路线）
9. Phase 0 完整实验（5.2 第 4 条的正式版）

### 5.5 给老师的开放问题（需拍板）
1. **动机强度**：冷站数据上静态策略退化是否足够强，值不值得做闭环自迭代？
2. **范围**：π 与 τ 共同进化 vs 先只做 π 自迭代（τ 保持离线）——先做哪个？
3. **平台**：仿真器 vs 真实数据滚动回放，哪个作主实验平台？
4. **基线**：对比哪些（离线重训 / 多策略档案 CQD-ERL / 变化点检测多模型切换 BAE 2022）？

### 5.6 精读清单（支撑 5.1 第 2 步，v3 按目的分两条线）

**线 A——机制侧（当前优先）：怎么在 ET-PRL 之上搭出 RSI 环**
前一篇结构 = 事件触发控制（策略 π + 触发阈值 τ，τ 离线贝叶斯调参）。引入 RSI = 在外层套"部署→收集在线经验→迭代触发判断→更新 π/τ→再部署"的环。以下按 5 个设计问题对应：
- **RLoop (arXiv 2511.04285)**——最接近"自迭代"的表述：探索→筛成功轨迹→重训→下一轮。**外层环结构模板**
- **高效重启 (arXiv 2510.11933)**——指出全量重训（遗忘）与固定周期重启（盲目）两大缺陷，给部分/自适应/选择性重启。**"每次迭代怎么更新"的直接约束**
- **SELFI (arXiv 2403.00991)**——离线预训练+在线微调，离线目标锚定在线学习。**最贴"延续前一篇静态策略再适应"的范式**
- **ProteinZero (arXiv 2506.07459)**——KL 防漂移 + 多样性防坍缩。**迭代中不破坏策略的正则设计**
- **Adaptive ETRL (arXiv 2409.19769)**——联合学控制策略+触发策略。**"触发阈值本身可进化"的最近现有工作，ET-PRL→RSI 的桥**
- **Tri-CRLAD (arXiv 2405.06925)**——自适应阈值平滑调整 + 自适应决策奖励。**回路内调阈值的机制参照**
- **CQD-ERL (arXiv 2608.11324)**——多策略档案（QD 进化+工况档案+安全盾）。**主要对照：单策略自迭代 vs 多策略档案**
> 注意：线 A 全在 LLM/机器人/蛋白/非线性控制域，建筑域无现成 RSI 可抄；读法是"拆机制→判断能否移植到冷站事件触发结构"，这正是 gap。
> 补充（2026-09-28）：LLM 自进化环机制参照 5 篇（AEL / CPE / Skill-R1 / EvolveR / COSPLAY，见 2.9 第一梯队），对应触发 / 防遗忘 / 跨轮信用分配 / 共同进化四个机制点。

**线 B——差异化侧（写 Related Work 用）：怎么和已有工作划清界限**
- **Energies 2025 relearning framework (doi:10.3390/en18061408)**——模式 A 应用侧最近邻：退化触发再训，非经验闭环自迭代，不调阈值
- **BAE 2022 变化点检测 + 多模型切换 (doi:10.1016/j.buildenv.2021.108680)**——非平稳侧最近邻：离散多模型切换，非单策略连续自迭代
- **Applied Energy 2025 自适应冷机切换阈值 (doi:10.1016/j.apenergy.2025.125718)**——模式 B 冷站侧最近邻：规则式阈值（物理+FDD），非 RL 内生学习
- **Zhan et al. ICLR 2025 (arXiv 2501.15085)**——应用侧强基线，Phase 0 评测协议参照

## 六、论文总表（72 篇，2026-09-28 整理）

> 覆盖本文 2.1–2.6、2.9 全部论文；按类别分组（A–J），与上文各表一一对应。"可参考/借鉴"列是面向冷站 RSI 环的取舍结论。

| 工作 | 期刊/会议 | 解决的问题 | 方法 | 可参考/借鉴的内容 |
|---|---|---|---|---|
| **A. 自改进/自迭代 RL（方法侧，2.1）** | | | | |
| RLoop (arXiv 2511.04285) | arXiv 2025 | LLM 推理策略迭代改进，防遗忘/过特化 | 探索 → 筛成功轨迹 → RFT 重训 → 下一轮；以上轮策略为起点，保留步间策略多样性 | 外层环结构模板：迭代单元 = 一批成功轨迹；更新 = RFT 微调而非从头训 |
| SELFI (arXiv 2403.00991) | CoRL 2024 | 离线预训练后如何稳定在线微调（机器人） | 在线 model-free 叠离线 model-based 预训练，离线目标锚进在线 Q 值 | 最贴"延续前一篇"：前一篇静态策略 = 在线迭代的锚点/起点 |
| MEDAL++ (arXiv 2303.01488) | arXiv 2023 | 机器人域端到端自改进 | 自主练习 + 从演示推断奖励 | 奖励设计思路（机制不同，相关性低） |
| ProteinZero (arXiv 2506.07459) | TMLR 2026 | 在线 RL 持续自改进不漂移/不模式坍缩 | 多目标奖励 + KL 偏离参考模型 + 嵌入级多样性正则 | 防遗忘/防漂移设计：每轮更新加 KL 锚 + 多样性项 |
| SIBRE (arXiv 2004.09846) | arXiv 2020 | 自改进的奖励设计 | 奖励相对自身历史改进 | 奖励设计思路 |
| SAGE (arXiv 2512.17102) | arXiv 2025 | 自进化 LLM agent | 技能库 + 顺序 rollout | 相关性低（LLM 域） |
| SERL (arXiv 2511.07922) | AAAI 2026 | LLM 自改进的自奖励机制 | 模型自当 actor + judge 自检验 RL | 自奖励机制参照（LLM 域） |
| **B. 冷站/建筑 RL 控制（应用侧 arXiv，2.2）** | | | | |
| Zhan et al. (arXiv 2501.15085) | ICLR 2025 | 数据中心冷却真实工况控制 | 离线 RL：GNN 物理感知模型 + latent 空间策略；生产 DC 部署 2000h，省 14–21% | 应用侧强基线必引；Phase 0 评测协议参照 |
| CQD-ERL (arXiv 2608.11324) | arXiv 2026 | 热带水冷冷站群非平稳 | 质量-多样性进化 + 按天气/负荷工况策略档案 + 安全盾 | 对比点：多策略档案 vs 单策略自迭代；安全盾参照 |
| 冷机+TES 共设计 (arXiv 2601.22880) | arXiv 2026 | 冷机控制与生命周期成本共设计 | DQN 控制冷机 PLR + 30 年生命周期成本共设计 | 动作空间设定最接近 |
| 澳洲大学校园冷机 (arXiv 2511.14160) | arXiv 2025 | 真实冷机节能 | PPO + 滚动时域 + 约束优先奖励，省 28% | 真实案例参照 |
| Guo et al. (arXiv 2203.07500 / 2310.03814) | arXiv 2022/2023 | 区域供冷控制 | Q-learning vs MPC，省 8–17% | 经典基线 |
| Wong et al. (arXiv 2209.08112) | arXiv 2022 | 工业 HVAC 多时间尺度控制 | 分层 RL（多时间尺度动作） | 动作分层参照 |
| DCVerse (arXiv 2604.07559) | arXiv 2026 | 数据中心冷却数据稀缺 | 数字孪生 + DRL 策略池双环控制 | 对比：孪生重训 vs 闭环自迭代；数字孪生文献线 |
| IBM RL Testbed (arXiv 1808.10427) | arXiv 2018 | 数据中心冷却 RL 测试 | EnergyPlus 测试床 | 经典测试床（仿真器路线参照） |
| **C. 非平稳 RL（理论侧，2.3）** | | | | |
| 高效重启 (arXiv 2510.11933) | arXiv 2025 | RestartQ-UCB 完全遗忘 + 定时重启两缺陷 | 部分/自适应/选择性重启，动态遗憾降 91% | 迭代触发与更新方式的直接约束：失配才迭代；部分式更新 |
| FANS-RL (arXiv 2203.16582) | NeurIPS 2022 | 非平稳适应 | 因果分解（潜变量变化因子） | 理论参照 |
| BORL (arXiv 2006.14389) | ICML 2020 | 漂移非平稳 MDP | 无变化预算的自适应调参 | 理论参照 |
| ESN 在线适应 (arXiv 2602.06326) | arXiv 2026 | 部署侧轻量在线适应 | reservoir + RLS，免反传 | 部署侧轻量方案参照 |
| PPO 自适应滤波 (arXiv 2506.06323) | arXiv 2025 | 非平稳噪声 | RL 直接调滤波器系数 | "RL 调参数"邻近范式 |
| **D. 自适应阈值（模式 B，2.4）** | | | | |
| Tri-CRLAD (arXiv 2405.06925) | arXiv 2024 | 半监督传感器异常检测 | 因果 RL + 自适应阈值平滑调整 + 自适应决策奖励 | 模式 B 最近邻，必须精读区分 |
| ATH (arXiv 2308.10504) | arXiv 2023 | concept drift 下 KPI 异常检测 | 周期性 + 异常比例自适应阈值 | 规则式强基线 |
| ReRe (arXiv 2004.02319) | arXiv 2020 | 实时流式异常检测 | LSTM + 双自适应用阈值 | 门控结构近亲 |
| ADALog (arXiv 2505.13496) | arXiv 2025 | 日志异常检测 | 正常数据上自适应分位阈值 | 阈值校准思路 |
| **E. 事件触发 RL（ET-PRL 根基，2.5）** | | | | |
| Baumann et al. (arXiv 1809.05152) | arXiv 2018 | 事件触发控制开创 | DRL + 事件触发控制 | 奠基工作 |
| Adaptive ETRL / ATPPO (arXiv 2409.19769) | arXiv 2024 | 非平稳策略 + 触发 | 联合学控制策略与触发策略；累积奖励增广状态（隐式触发） | π–τ 共同进化的桥：证明"触发条件可学"；隐式路线 vs ET-PRL 显式双阈值 |
| ET-MAPG (arXiv 2509.20338) | arXiv 2025 | 多 agent 事件触发 | 事件触发策略梯度 | 多 agent 扩展参照 |
| 建筑微气候 ETRL (arXiv 2001.10505) | Applied Energy 2020 | 建筑微气候事件触发控制 | SMDP 框架 + 事件触发学习/控制 | 理论外壳（SMDP）；建筑侧必引 |
| eMPC+DRL (arXiv 2208.10302) | arXiv 2022 | 自动驾驶事件触发策略 | RL 学事件触发策略 | 触发策略学习参照 |
| **F. 期刊补充：冷站/建筑 RL 控制（2.6.1）** | | | | |
| 冷站预测控制 DRL (JBE 2023, 被引 52) | J. Building Engineering | 冷站预测控制 | DRL | 期刊经典基线 |
| 聚类 DRL 冷站 (JBE 2023, 被引 26) | J. Building Engineering | 分工况 model-free 冷站控制 | 聚类分工况 + DRL | 分工况思路，与多策略档案同族 |
| 冷站复杂特性 RL (EAB 2025, 被引 13) | Energy and Buildings | 复杂水力/热工特性冷站最优控制 | RL 最优控制 | 真实冷站建模参照 |
| 知识引导分层 RL 酒店部署 (EAB 2026) | Energy and Buildings | 冷站鲁棒控制 + 现场部署 | 知识引导分层 RL，酒店现场部署 | 分层 + 现场部署，强参照 |
| 多冷机 DQN 系统评估 (ATE 2025) | Applied Thermal Engineering | 多冷机系统控制 | DQN 系统评估 | 动作/算法设定参照 |
| 半导体厂多 agent RL (IJPE 2025) | Int. J. Production Economics | 半导体厂冷机群预测 + 节能 | 多 agent RL | 工业冷站参照 |
| 大冷机群多 agent RL (IBPSA BS 2025) | Building Simulation 2025 | 大冷机群长期运行 | 多 agent RL，广州商业综合体 8 个月现场，比规则省 10.6%、比单 agent 省 5% | 真实长期运行参照 |
| 冷凝水 Q-learning (ASHRAE Annual 2025) | ASHRAE | 冷凝水系统控制 | Q-learning，省 14.7–19.4% | ASHRAE 侧代表 |
| 博弈论多 agent 冷站群控 (Systems 2023, 被引 17) | Systems | 多冷机群控 | 博弈论 + 多 agent RL | 群控机制参照 |
| 区域供冷 RL+MPC (ASME JESBC 2023) | ASME J. Eng. for Sustainable Buildings & Cities | 区域供冷能源站 | RL+MPC | 区域供冷参照 |
| 印度 39 家医院 RL 部署 (SEEBE 2024) | SEEBE 2024 | 大规模 RL 部署经验 | 39 家医院部署经验教训 | 大规模部署教训，必引 |
| 深度聚类合作多 agent (JBE 2022, 被引 57) | J. Building Engineering | 多冷机 HVAC 优化 | 深度聚类 + 合作多 agent | 经典高引基线 |
| Lyapunov 安全 RL DC 冷却 (EECN 2023, 被引 19) | Energy-Efficient Computing & Networking | 数据中心冷却安全控制 | Lyapunov 安全约束 RL | 安全机制参照 |
| LC-Opt (NeurIPS 2025) | NeurIPS 2025 | 液冷端到端优化基准 | RL/agentic AI 基准 | 新基准，数据中心侧 |
| 冰蓄能区域供冷安全 DRL (IEEE TSG 2026) | IEEE Trans. Smart Grid | 冰蓄能区域供冷 | model-free 安全 DRL | 安全 DRL 期刊代表 |
| 异构迁移学习 DRL (Building Simulation 2024, 被引 46) | Building Simulation | DRL 跨建筑可扩展性 | 异构迁移学习 | 迁移学习路线，与自迭代对比 |
| 安全数据高效 model-based RL HVAC (IEEE IoT-J 2024) | IEEE Internet of Things J. | 安全 + 数据高效 HVAC 控制 | model-based RL | 部署侧参照 |
| ORCHID (IC3SE 2024) | IC3SE 2024 | 历史数据 + 低保真仿真下的 HVAC 控制 | 离线 RL | 离线 RL 路线代表 |
| 冷站 Gym 环境 (Energies 2025) | Energies | 冷站 RL 评测环境 | OpenAI Gym 环境，DQN/DDQN 省 14% | 评测环境参照 |
| **G. 期刊补充：非平稳建筑 RL（2.6.2）** | | | | |
| 非平稳建筑能源 DRL (EAB 2022) | Energy and Buildings | 非平稳建筑能源管理 | DRL 控制 | 非平稳建筑 RL 期刊代表 |
| 变化检测 + DRL (BAE 2022, 108680) | Building and Environment | 非平稳 HVAC 控制 | 主动变化点检测 + 每工况上下文单独学 DRL，漂移即切换 | 非平稳侧最近邻：多模型切换，必须回应 |
| 非平稳 DRL 持续适应 (2020) | 1st Workshop on RL for Energy Management in Buildings & Cities | 非平稳建筑环境 DRL 持续适应 | DRL 持续适应 | 早期工作 |
| 非平稳用户行为 model-free RL (2020) | Smart Grid and Smart Cities | 非平稳用户行为 | model-free RL | 早期工作 |
| 多时间尺度多 agent 非平稳策略 (CDC 2023) | IEEE CDC | 多时间尺度多 agent 非平稳 | 非平稳策略学习 | 理论参照 |
| Meta-RL 共享表征 (PAKDD 2026) | PAKDD 2026 | 能源系统快速适应 | 共享表征 meta-RL | meta-RL 路线参照 |
| **H. 期刊补充：自迭代/再学习/持续学习（2.6.3）** | | | | |
| 端到端再学习框架 (Energies 2025, 18(6):1408) | Energies | 设备老化/使用模式变化下建筑能源优化 | RL 控制器自监控 + 自适应，性能退化触发再学习 | 模式 A 应用侧最近邻：退化触发再训，不调阈值 |
| 自进化数字孪生 (Tsinghua Sci. Technol. 2026, 31(1):199–216) | Tsinghua Science and Technology | 复杂设备优化与控制 | 贝叶斯更新持续融合实时数据 + SAC agent 在线更新 | 命名撞车（self-evolving）+ 孪生在线更新机制参照 |
| 持续 RL HVAC 超网络 (arXiv 2503.19212) | arXiv 2025 | HVAC 任务序列持续学习 | 超网络建模环境动力学 + 持续学习 + 迁移，抗灾难性遗忘 | 持续学习路线：任务序列，非在线工况漂移 |
| 异构专家引导在线 DRL (Sci. Rep. 2025, 15:7677) | Scientific Reports | 在线 DRL 加速 | 物理模型 + 历史数据 + 专家规则异构引导 | 在线 DRL 加速路线参照 |
| 智能建筑 RL 再学习 (PHM 2020) | PHM Society | 智能建筑 RL 再学习 | 再学习 | 早期再学习工作 |
| **I. 期刊补充：自适应阈值（2.6.4）** | | | | |
| 物理引导冷机排序 (Applied Energy 2025, 389:125718) | Applied Energy | 测量不确定下冷机启停 | FDD 识别故障类型 + 自适应调整冷机切换阈值，节能至 7.46% | 模式 B 冷站侧最近邻：规则式阈值，非 RL 回路 |
| agent 式动态阈值 RL (NCA 2024) | Neural Computing & Applications | 自适应异常检测 | agent 式动态阈值 + RL | 阈值 + RL 组合（异常检测域） |
| ADT (arXiv 2023) | arXiv | 异常检测 | agent-based 动态阈值 | 动态阈值参照 |
| 半监督异常检测自适应 RL+因果 (arXiv 2024) | arXiv | 传感器信号异常检测 | 自适应 RL + 因果推断 | 传感器信号侧 |
| 预测性维护 + 冷却需求预测 (ICOSAAS 2026) | ICOSAAS 2026 | 数据中心热控维护 + 预测 | 混合 RL+AI | 数据中心侧 |
| **J. LLM 自进化环机制参照（2.9）** | | | | |
| AEL (arXiv 2604.21725) | arXiv 2026 | agent harness 固定下经验怎么用 | 双时间尺度：快 Thompson 采样选检索策略 + 慢反思"先诊断后开方"注入新策略 | 外层触发判据 + 快慢分离 |
| Do Self-Evolving Agents Forget? (arXiv 2605.09315) | arXiv 2026 | 自进化中的能力侵蚀（遗忘） | Capability-Preserving Evolution 稳定化原则 | 防遗忘/防模式坍缩 + 保留能力度量 |
| Skill-R1 (arXiv 2605.09359) | arXiv 2026 | 迭代式 skill 优化的跨轮信用分配 | 双层 GRPO：intra-generation + inter-generation advantage | 跨轮信用分配：怎么判定 k+1 比 k 好 |
| EvolveR (arXiv 2510.16079) | arXiv 2025 | 从自身经验系统学习 | 闭环经验生命周期：离线蒸馏 + 在线检索 + 策略强化 | 环结构模板（与 RLoop 同族） |
| COSPLAY (arXiv 2604.20987) | arXiv 2026 | 长时程任务 skill 发现/保持/复用 | 决策 agent 与 skill bank 共同进化 | π–τ 共同进化机制参照 |
| JitRL (arXiv 2601.18510) | arXiv 2026 | 部署后冻结权重 LLM 持续适应 | 免梯度非参数经验记忆 + KL 约束策略优化闭式解 | 低成本更新思路 |
| Metacognition (arXiv 2506.05109) | arXiv 2025 | 自改进过程刚性、难泛化 | 元认知三组件（自评/规划/评估） | 外层触发的概念外壳 |
| EvoLM (arXiv 2605.03871) | arXiv 2026 | 外部监督奖励的天花板 | 判别式 rubric 与策略共同进化当自监督奖励 | 与"τ 作为评估信号与 π 共同进化"同构 |
