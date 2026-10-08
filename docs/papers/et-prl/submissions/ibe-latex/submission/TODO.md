# Indoor and Built Environment 投稿检查清单

更新时间：2026-09-12

依据：iBE Guide for Authors（Sage，`https://journals.sagepub.com/author-instructions/IBE`）、当前投稿目录 `docs/papers/et-prl/submissions/ibe-latex/submission/`、`docs/papers/期刊投稿选择.md`。

## 当前状态

- [x] 在独立目录 `ibe-latex/submission/` 中从 IJR 版改造，不覆盖原稿。
- [x] 主稿目标期刊改为 Indoor and Built Environment（`\journal{}`）。
- [x] 摘要精简至 193 词（满足 Sage Track 200 词硬上限），主稿与投稿系统已同步。
- [x] 关键词沿用 6 个（iBE 要求至少 5 个），列在摘要后。
- [x] 末尾声明按 iBE 要求重组为 `Statements and Declarations` 节（含 6 个子标题，不适用项写 Not applicable），并保留 CRediT / Acknowledgements / 生成式 AI 声明。
- [x] 准备 iBE 定制封面信（突出室内环境/建筑能效与舒适性侧契合）。
- [x] 复制 elsarticle 类文件、bst、10 幅图 PDF 到本目录。

## 投稿前阻断项

- [x] **摘要字数**：Sage Track 硬上限 200 词，已精简至 193 词并同步主稿 `ET_PRL_IBE.tex` 与投稿系统。
- [ ] **确认 iBE 中科院分区**：官网 IF 2.7 / 5 年 2.9，大致在 4 区附近；用当年官方《中科院期刊分区表》或 LetPub 核对最新分区，确认符合目标定位。
- [ ] **参考文献 Sage Vancouver 细节核对**：当前沿用 IJR 版编号 `\bibitem` 格式，需逐条按 Sage Vancouver 规范核对作者、题名、期刊、年;卷(期):页、DOI 的精确排布，并确认正文引用与文献列表一一对应（当前 29 条引用 / 29 条 bibitem）。
- [ ] 确认通讯作者邮箱、电话、邮寄地址在 Sage Track 系统中完整填写。
- [ ] 确认研究数据与代码的长期归档（DOI）：当前 GitHub 地址不一定满足 iBE Research Data 要求，建议归档到 Zenodo / Mendeley Data 并绑定 DOI 后更新 Data availability。
- [ ] 确认本稿未在其他期刊审理（IJGE 已拒稿，确认无其他在审流程）。
- [ ] 由全体作者确认作者顺序、单位、CRediT 贡献与最终投稿版本。

## 技术筛选检查

- [ ] 每张图作为独立文件上传，检查缩放后文字可读性与色觉无障碍（Sage artwork guidelines）。
- [ ] 表格作为独立可编辑文件上传（Sage Track 通常要求表格单独成页）。
- [ ] 确认使用美式拼写与双引号。
- [ ] 确认声明节各子标题齐全，不适用项均已写 "Not applicable"。

## 内容风险

- [ ] 现有结果来自单一商业建筑的校准仿真，缺少现场闭环验证；正文已保持边界清晰，不得表述为现场实验。
- [ ] iBE 偏室内环境/建成环境质量与舒适性取向，本文偏控制算法 + 仿真；封面信已突出室内热环境稳定性与建筑能效贡献，审稿阶段需准备回应"室内环境/舒适性侧证据"的质疑。
- [ ] 节能幅度偏保守（2.13%），需强调"以 53.54% 更少的策略执行换取同等性能"的执行成本-性能权衡这一核心卖点。

## 投稿系统内需要人工确认

> 以下事项均在 Sage Track（ScholarOne）网页中完成，本地无需新增或修改文件。

- [ ] 确认 submission declaration：稿件未发表、未一稿多投、所有作者同意投稿、机构/项目负责人同意发表。
- [ ] 确认通信作者和所有作者信息与投稿系统完全一致（作者顺序提交后原则上不应再变更）。
- [ ] 选择出版模式：选 **Subscription（非开放获取）** 免 APC。
- [ ] 完成推荐审稿人或回避审稿人信息（如系统要求）。
- [ ] 如使用 preprint，在系统指定字段填写 preprint DOI 并告知编辑部。

## 建议但非阻断项

- [ ] 给 GitHub 仓库创建 release，并用 Zenodo 或同类服务生成长期 DOI。
- [ ] 将实验环境、数据字段说明、运行命令和随机种子整理到仓库 README，以增强可复现性。
- [ ] 如需完全贴合 Sage 官方排版，可下载 Sage Journal Author Gateway 的官方 LaTeX 模板替换现有 elsarticle 基础设施。
