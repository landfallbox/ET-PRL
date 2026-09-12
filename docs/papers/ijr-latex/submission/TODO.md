# International Journal of Refrigeration 投稿检查清单

更新时间：2026-09-08

依据：IJR Guide for Authors（Elsevier）、当前投稿目录 `docs/papers/ijr-latex/submission/`、`docs/papers/期刊投稿选择.md`。

## 当前状态

- [x] 在独立目录 `ijr-latex/submission/` 中从 Applied Energy 版改造，不覆盖原稿。
- [x] 主稿目标期刊改为 International Journal of Refrigeration（`\journal{}`）。
- [x] 通讯作者邮箱改为机构邮箱（`fqm@mail.usts.edu.cn`、`alan@mail.usts.edu.cn`）。
- [x] 摘要（≤ 250 词，当前 248 词）、关键词（6 个）、Highlights、参考文献 DOI、AI 声明沿用 Applied Energy 版。
- [x] 准备 IJR 定制封面信（Elsevier 简洁风格：无加粗分节标题、无作者表格，作者信息在投稿系统录入），突出冷冻水/制冷系统侧契合。
- [x] 复制 elsarticle 类文件、bst、10 幅图 PDF、利益冲突声明 docx、Highlights 到本目录。

## 投稿前阻断项

- [ ] **确认 IJR 中科院分区版本**：LetPub 最新为 3 区（新锐版），基础版为 2 区；若按基础版则超出"三区及以下"红线。投稿前用当年官方《中科院期刊分区表》或 LetPub 核对，确认所用版本符合目标。
- [ ] 确认通讯作者邮箱、电话、邮寄地址在投稿系统中完整填写。
- [ ] 确认研究数据与代码的长期归档（DOI）：当前 GitHub 地址不一定满足 IJR Research Data 要求，建议归档到 Zenodo / Mendeley Data 并绑定 DOI 后更新 Data availability。
- [ ] 确认本稿未在其他期刊审理（IJGE 已拒稿，确认无其他在审流程）。
- [ ] 由全体作者确认作者顺序、单位、CRediT 贡献与最终投稿版本。

## 技术筛选检查

- [ ] 正文引用与参考文献逐项对应，人工核对作者、题名、年份、卷期、文章号和 DOI（当前 29 条引用 / 29 条 bibitem）。
- [ ] 每张图作为独立文件上传，检查缩放后文字可读性与色觉无障碍。
- [ ] 表格作为独立可编辑文件上传（Elsevier 要求表格单独成页）。
- [ ] 按投稿系统要求决定是否上传 graphical abstract（IJR 鼓励但非强制）。
- [ ] 确认使用美式拼写与双引号。

## 内容风险

- [ ] 现有结果来自单一商业建筑的校准仿真，缺少现场闭环验证；正文已保持边界清晰，不得表述为现场实验。
- [ ] IJR 偏制冷机理/实验取向，本文偏控制算法 + 仿真；封面信与投稿信已突出冷冻水供水温度控制这一制冷循环核心变量的贡献，审稿阶段需准备回应"制冷系统侧机理/实验证据"的质疑。
- [ ] 节能幅度偏保守（2.13%），需强调"以 53.54% 更少的策略执行换取同等性能"的执行成本-性能权衡这一核心卖点。

## 投稿系统内需要人工确认

> 以下事项均在 Elsevier Editorial Manager 网页中完成，本地无需新增或修改文件。

- [ ] 确认 submission declaration：稿件未发表、未一稿多投、所有作者同意投稿、机构/项目负责人同意发表。
- [ ] 确认通信作者和所有作者信息与投稿系统完全一致（作者顺序提交后原则上不应再变更）。
- [ ] 选择出版模式：选 **Subscription（非开放获取）** 免 APC。
- [ ] 完成推荐审稿人或回避审稿人信息（如系统要求）。
- [ ] 上传 `declaration-of-competing-interests.docx` 与 `Highlights.tex`（作为单独可编辑文件）。

## 建议但非阻断项

- [ ] 给 GitHub 仓库创建 release，并用 Zenodo 或同类服务生成长期 DOI。
- [ ] 将实验环境、数据字段说明、运行命令和随机种子整理到仓库 README，以增强可复现性。
