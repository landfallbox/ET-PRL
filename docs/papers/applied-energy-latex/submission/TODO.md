# Applied Energy 投稿本地 TODO

更新时间：2026-07-11  
依据：Applied Energy Guide for Authors、当前投稿目录 `docs/papers/applied-energy-latex/submission/`、当前 LaTeX 构建结果。

## 当前状态

- [x] 主稿 `ET_PRL_Applied_Energy.tex` 可以用 `latexmk -xelatex` 成功编译。
- [x] 摘要、关键词、Highlights、参考文献 DOI、AI 使用声明已补充并通过基础检查。
- [x] 参考文献引用与 `bibitem` 数量一致，目前为 29 条引用、29 条参考文献。
- [ ] 主稿仍有 `TODO:` 占位，投稿前必须全部替换。

## 投稿前阻断项

- [ ] 替换标题页作者信息。
  - 文件位置：`ET_PRL_Applied_Energy.tex`
  - 当前占位：`TODO: First Author`、`TODO: Co-author`
  - 需要确认：作者姓名顺序、英文拼写、通信作者、所有作者是否已同意投稿版本。

- [ ] 补全通信作者联系方式。
  - 当前占位：`TODO: corresponding.author@example.com`
  - Applied Energy/Elsevier 要求：通信作者邮箱、完整邮政地址、投稿系统中的联系方式保持最新。

- [ ] 补全作者单位信息。
  - 当前占位：`TODO: Department, Institution`、`TODO: City`、`TODO: Country`
  - 需要确认：单位英文全称、城市、国家、如有多个单位需增加 `inst2` 等 affiliation。

- [ ] 填写 CRediT authorship contribution statement。
  - 当前占位：`TODO: Add author contribution roles before submission.`
  - 建议角色范围：Conceptualization、Methodology、Software、Validation、Formal analysis、Investigation、Data curation、Writing - original draft、Writing - review and editing、Supervision、Funding acquisition。

- [x] 完成 Declaration of competing interest。
  - 已在主稿中填写无利益冲突声明：`The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.`
  - 投稿系统中仍需在 Elsevier declarations tool 选择 `I have nothing to declare`，并按系统要求生成、上传 `.doc/.docx` 声明文件。

- [ ] 完成 Funding statement。
  - 当前占位：`TODO: Add funding information, grant numbers, or state that this research received no specific grant.`
  - 若无经费支持，Elsevier 推荐句式：`This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.`
  - 若有基金，需列出基金名称和 grant number，并说明资助方是否参与研究设计、数据分析、写作或投稿决定。

- [ ] 核实 Data availability 是否满足 Applied Energy 的 Research Data 要求。
  - 当前文本：代码和数据位于 `https://github.com/landfallbox/ET-PRL.git`。
  - Applied Energy 说明研究数据需存入相关数据仓库、在文中引用并链接；如不能公开，需说明原因。
  - 建议操作：将代码和数据归档到 Zenodo、Mendeley Data 或机构数据仓库，获得 DOI 后更新 Data availability；若 GitHub 已含完整可复现实验数据，也建议生成 release 并绑定 Zenodo DOI。

## 投稿文件包检查

- [ ] 准备可编辑主稿源文件。
  - 必传：`ET_PRL_Applied_Energy.tex`
  - 同包依赖：`elsarticle.cls`、`elsarticle-num.bst`、`elsarticle-num-names.bst`、`elsarticle-harv.bst`。
  - 不建议上传：`build/` 下的 `.aux`、`.log`、`.fls`、`.synctex.gz` 等构建中间文件。

- [ ] 上传单独 Highlights 文件。
  - 当前文件：`Highlights.tex`
  - 当前满足 3 到 5 条、每条不超过 85 字符的要求。

- [ ] 核对并上传所有图文件。
  - 当前投稿目录已有：`fig1_update_strategy_comparison.pdf`、`fig2_plain.pdf`、`fig3_dataset_temporal_characteristics.pdf`、`fig4_dataset_distribution_and_correlation.pdf`、`fig6_t_chws_macro_distribution.pdf`、`fig7_t_chws_delta_analysis.pdf`、`fig8_action_update_interval_distribution.pdf`、`fig9_trigger_load_alignment.pdf`、`fig10_ablation_pareto_scatter.pdf`。
  - 需要确认：正文是否引用所有图、图号顺序是否连续、文件名是否适合投稿系统、图片中文字在 5 x 13 cm 或单栏缩放后仍清晰。

- [ ] 确认图表版权和数据权限。
  - 若全部图表由本文代码和自有数据生成，可在投稿系统中按无第三方版权材料处理。
  - 若使用商业建筑运行数据，需要确认数据使用授权、匿名化处理和公开范围。

- [ ] 视情况准备 Graphical abstract。
  - Applied Energy 鼓励但非强制。
  - 若准备，需单独上传，建议尺寸至少 531 x 1328 px 或等比例更高，首选 TIFF、EPS、PDF 或 MS Office 文件。

- [ ] 视情况准备 cover letter。
  - Editorial Manager 可能要求或允许上传。
  - 建议简述论文主题、创新点、适配 Applied Energy 范围，以及未一稿多投声明。

## 稿件内容最终检查

- [ ] 全文语言和格式终检。
  - 检查英式/美式拼写是否统一。
  - 检查标题、摘要、关键词、图题、表题、变量符号和缩写是否一致。
  - 删除或替换所有 `TODO:`。

- [ ] 摘要终检。
  - Applied Energy 要求摘要不超过 250 words。
  - 当前此前检查通过，但在任何改写后需重新统计。

- [ ] 关键词终检。
  - Applied Energy 要求 1 到 7 个关键词。
  - 当前为 6 个关键词。

- [ ] 参考文献终检。
  - 已补全 29 条 DOI。
  - 投稿前再次检查：正文引用和参考文献列表一一对应，参考文献顺序按首次出现顺序排列。

- [ ] AI 使用声明终检。
  - 当前已放在参考文献之前，符合 Elsevier 要求。
  - 若图形或图形摘要使用了生成式 AI，还需在相关图注和 AI 声明中单独说明。

## 投稿系统内需要人工确认

- [ ] 确认 submission declaration。
  - 稿件未发表、未一稿多投、所有作者同意投稿、机构或项目负责人同意发表。

- [ ] 确认通信作者和所有作者信息与投稿系统完全一致。
  - 作者顺序提交后原则上不应再变更。

- [ ] 选择出版模式和费用责任。
  - 确认是否 open access，以及 APC 由谁承担。

- [ ] 完成推荐审稿人或回避审稿人信息。
  - 若系统要求，准备姓名、单位、邮箱、研究方向和推荐/回避理由。

## 建议但非阻断项

- [ ] 给 GitHub 仓库创建 release，并用 Zenodo 或同类服务生成长期 DOI。
- [ ] 将实验环境、数据字段说明、运行命令和随机种子整理到仓库 README，以增强可复现性。
- [ ] 检查 PDF 首页、图表、长表和参考文献 URL 是否存在明显排版溢出。

## 最后提交前命令

在投稿目录运行：

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN-WIP\docs\papers\applied-energy-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_Applied_Energy.tex
```

通过标准：命令退出码为 `0`，日志中无 `LaTeX Error`、`Emergency stop`、`Undefined control sequence`、undefined citation。