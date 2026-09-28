# 文献调研：自迭代 RL × 冷站控制 × 阈值自适应

> 新论文（next-paper 分支）专属文献笔记。
> 调研日期：2026-09-22。检索方式：arXiv API（含 2025–2026 最新工作），覆盖四个方向：自改进 RL、冷站/建筑 RL 控制、非平稳 RL、自适应阈值。
> 状态：初版，arXiv 覆盖不全（建筑能源领域大量文献在 ASHRAE / Applied Energy / Building and Environment），需补 Google Scholar / Semantic Scholar。

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

## 二、各方向现状

### 2.1 自改进/自迭代 RL（方法侧，最拥挤）

| 工作 | 机制 | 与我们的关系 |
|---|---|---|
| RLoop (arXiv 2511.04285, 2025.11) | 迭代策略初始化：RL 探索 → 筛选成功轨迹 → RFT 重训 → 下一轮；缓解遗忘与过特化 | 最接近"自迭代"表述，但用于 LLM 推理；**机制模板** |
| SELFI (arXiv 2403.00991, CoRL 2024) | 离线预训练 + 在线机器人经验微调，离线目标稳定在线学习 | 模式 A 的直接参照 |
| MEDAL++ (arXiv 2303.01488, 2023) | 自主练习 + 从演示推断奖励，端到端自改进 | 机器人域，机制不同 |
| ProteinZero (arXiv 2506.07459, TMLR 2026) | 在线 RL 持续自改进 + KL 正则防漂移 + 多样性正则防模式坍缩 | **防遗忘/防漂移设计可借鉴** |
| SIBRE (arXiv 2004.09846, 2020) | 奖励相对自身历史改进（self-improvement-based rewards） | 奖励设计思路 |
| SAGE (arXiv 2512.17102, 2025.12) | 技能库 + 顺序 rollout 的自进化 agent | LLM agent 域 |
| SERL (arXiv 2511.07922, AAAI 2026) | 模型自当 actor + judge 的自检验 RL | LLM 域，自奖励机制 |

**判断**：方法侧拥挤且 2025–2026 是热词，"self-improving / self-iterative" 撞名严重（RLoop/SERL/SIRLC/SiMT 等）；但**全部在 LLM/机器人/蛋白设计域，没有建筑能源控制**。

### 2.2 冷站/建筑 RL 控制（应用侧，arXiv 少、期刊多）

| 工作 | 要点 | 与我们的关系 |
|---|---|---|
| Zhan et al. (arXiv 2501.15085, **ICLR 2025**) | 数据中心冷却**离线 RL**：GNN 物理感知模型 + latent 空间策略；生产 DC 部署 2000h，省 14–21% | **应用侧强基线，必引** |
| CQD-ERL (arXiv 2608.11324, 2026.08) | 热带水冷冷站群：质量-多样性进化 + 按天气/负荷工况的策略档案（archive）+ 安全盾 | 用"多策略档案"应对非平稳，**与单策略自迭代形成对比点** |
| Chiller+TES 共设计 (arXiv 2601.22880, 2026.01) | DQN 控制冷机 PLR + 30 年生命周期成本共设计 | 动作空间设定最接近 |
| 澳洲大学校园冷机 (arXiv 2511.14160, 2025.11) | PPO + 滚动时域 + 约束优先奖励，省 28% | 真实案例参照 |
| Guo et al. (arXiv 2203.07500 / 2310.03814) | 区域供冷 Q-learning vs MPC，省 8–17% | 经典基线 |
| Wong et al. (arXiv 2209.08112, 2022) | 分层 RL 控制工业 HVAC（多时间尺度动作） | 动作分层参照 |
| DCVerse (arXiv 2604.07559, 2026.04) | 数字孪生 + DRL 策略池双环控制 | 用"孪生重训"应对数据稀缺，与闭环自迭代对比 |
| IBM RL Testbed (arXiv 1808.10427) | EnergyPlus 数据中心冷却 RL 测试床 | 经典 |

**判断**：应用侧全是**静态策略**（离线训练一次部署）；非平稳靠多策略档案（CQD-ERL）或数字孪生重训（DCVerse）解决，**没有闭环自迭代**。

### 2.3 非平稳 RL（理论侧，成熟）

| 工作 | 要点 | 与我们的关系 |
|---|---|---|
| 高效重启 (arXiv 2510.11933, 2025.10) | 指出 RestartQ-UCB 两大缺陷：**完全遗忘**（重启丢全部历史）与**定时重启**（不看策略与环境是否失配）；提出部分/自适应/选择性重启，动态遗憾降 91% | **迭代单元设计的直接约束**：每次迭代不能全量重训丢历史，也不能固定周期盲目重启 |
| FANS-RL (arXiv 2203.16582, NeurIPS 2022) | 因果分解的非平稳适应（潜变量变化因子） | 理论参照 |
| BORL (arXiv 2006.14389, ICML 2020) | 漂移非平稳 MDP，无变化预算的自适应调参 | 理论参照 |
| ESN 在线适应 (arXiv 2602.06326, 2026.02) | 轻量在线适应模块（reservoir + RLS），免反传 | 部署侧轻量方案参照 |
| PPO 自适应滤波 (arXiv 2506.06323, 2025) | RL 直接调滤波器系数应对非平稳噪声 | "RL 调参数"的邻近范式 |

### 2.4 自适应阈值（模式 B 侧）

| 工作 | 要点 | 与我们的关系 |
|---|---|---|
| Tri-CRLAD (arXiv 2405.06925, 2024) | 半监督传感器异常检测：因果 RL + **自适应阈值平滑调整** + 自适应决策奖励 | **模式 B 最近邻，必须精读区分** |
| ATH (arXiv 2308.10504, 2023) | KPI 异常检测自适应阈值：周期性 + 异常比例，处理 concept drift | 规则式强基线 |
| ReRe (arXiv 2004.02319, 2020) | LSTM + 双自适应用阈值的实时流式异常检测 | 门控结构近亲 |
| ADALog (arXiv 2505.13496, 2025) | 日志异常检测，正常数据上的自适应分位阈值 | 阈值校准思路 |

**判断**：自适应阈值全在异常检测域，**没有作为 RL 控制回路的一部分在线调整**。

### 2.5 事件触发 RL（旧论文 ET-PRL 根基）

- Baumann et al. (arXiv 1809.05152, 2018)：DRL + 事件触发控制开创工作
- Adaptive ETRL (arXiv 2409.19769, 2024.09)：联合学习控制策略与触发策略，自适应非平稳策略
- ET-MAPG (arXiv 2509.20338, 2025.09)：多 agent 事件触发策略梯度
- 建筑微气候事件触发 RL (arXiv 2001.10505, **Applied Energy 2020**)：SMDP 框架 + 事件触发学习/控制
- eMPC+DRL (arXiv 2208.10302, 2022)：RL 学事件触发策略（自动驾驶）

**判断**：旧论文已站在这条线上；新论文要往前推"**触发阈值本身可进化**"，这是与上述工作的差异点。

## 三、Gap 与定位

现有工作的组合方式：
- 自迭代 RL → 全在 LLM/机器人/蛋白设计，**没有建筑能源控制**
- 冷站 RL → 全是**静态策略**，非平稳靠"多策略档案"或"数字孪生重训"，**没有闭环自迭代**
- 自适应阈值 → 全在异常检测域，**没有作为 RL 控制回路的一部分在线调整**

**独占交叉点**：真实冷站数据上的闭环自迭代，同时进化**控制策略（A）和事件触发阈值（B）**；B 是旧论文已知弱点（离线调参）的自然延伸。

**贡献表述（暂定）**：

> 首个在真实冷站群控数据上验证的自迭代 RL 框架，通过闭环经验迭代同时适应策略与事件触发阈值，解决静态策略/静态阈值在非平稳运行工况下的退化。

## 四、风险点

1. **命名撞车**："RSI" 撞金融相对强弱指数，且 RL 领域已有 RLoop/SERL/SIRLC/SiMT 等一串缩写；"self-improving/self-iterative" 是 2025–2026 热词。建议方法名走"机制描述"路线（强调 closed-loop / co-adaptation），不用缩写。
2. **审稿人必问**："为什么不直接用离线 RL + 定期重训？"——Phase 0 动机实验（静态策略随时间退化）+ 滚动评估协议就是答案，优先级不变。
3. **arXiv 覆盖不全**：建筑能源领域主力期刊（ASHRAE、Applied Energy、Building and Environment、Energy and Buildings）大量相关工作不在 arXiv，必须补检索。

## 五、下一步做什么

1. **补 Google Scholar / Semantic Scholar 检索**（arXiv 漏期刊），关键词：
   - `self-evolving reinforcement learning building`
   - `continual reinforcement learning data center`
   - `event-triggered control threshold learning HVAC`
   - `online reinforcement learning chiller plant`
   - 重点查 Applied Energy / Building and Environment / ASHRAE 2023–2026
2. **精读三篇**：
   - Tri-CRLAD (arXiv 2405.06925)——模式 B 最近邻，明确区分点
   - RLoop (arXiv 2511.04285)——自迭代机制模板
   - Zhan et al. ICLR 2025 (arXiv 2501.15085)——应用侧强基线
3. **进入 Phase 0 动机实验**（与数据清洗并行）：验证"静态策略/静态阈值在新数据上随时间退化"是否成立——不成立则 RSI 动机不成立，需重新找非平稳性来源
4. **方法命名**：查完文献后定名（避开 RSI 及已有缩写）
5. 数据侧按 `data/external/csge_chiller_competition/TODO.md` 推进：权限确认 → 选目标项目（train_A / train_B）→ 清洗 → 接入管线
