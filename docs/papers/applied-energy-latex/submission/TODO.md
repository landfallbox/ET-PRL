# Applied Energy 投稿本地 TODO

更新时间：2026-07-13  
依据：Applied Energy Guide for Authors、当前投稿目录 `docs/papers/applied-energy-latex/submission/`、当前 LaTeX 构建结果。

## 当前状态

- [x] 主稿 `ET_PRL_Applied_Energy.tex` 可以用 `latexmk -xelatex` 成功编译。
- [x] 摘要、关键词、Highlights、参考文献 DOI、AI 使用声明已补充并通过基础检查。
- [x] 参考文献引用与 `bibitem` 数量一致，目前为 29 条引用、29 条参考文献。
- [x] 主稿已无 `TODO:` 占位。

## 投稿前阻断项

- [x] 确认标题页作者信息。
  - 文件位置：`ET_PRL_Applied_Energy.tex`
  - 已按当前顺序写入作者：Xinwei Tang、Qiming Fu、Jianping Chen、You Lu、Yunzhe Wang、Ke Liu。
  - 已确认共同通信作者：Qiming Fu、Jianping Chen。
  - 已按当前确认信息写入作者单位和通信作者邮箱。
  - 已核对 Ke Liu 所属学院官方英文名：`School of Architecture and Urban Planning`。
  - 已确认作者英文拼写、作者顺序、作者单位、通信作者设置和投稿版本同意状态。

- [x] 补全通信作者联系方式。
  - 已填写共同通信作者邮箱：`fqm_1@126.com`、`alanjpchen@hotmail.com`。
  - 主稿 PDF 中仅保留共同通信作者邮箱；其他作者邮箱可在投稿系统作者信息中填写。
  - 投稿系统中的通信作者联系方式仍需人工保持一致。

- [x] 补全作者单位信息。
  - Xinwei Tang、Qiming Fu、You Lu、Yunzhe Wang：School of Electronic and Information Engineering；Jiangsu Province Key Laboratory of Intelligent Building Energy Efficiency。
  - Jianping Chen：School of Electronic and Information Engineering；Jiangsu Province Key Laboratory of Intelligent Building Energy Efficiency；Chongqing Industrial Big Data Innovation Center Co., Ltd.
  - Ke Liu：School of Architecture and Urban Planning。

- [x] 填写 CRediT authorship contribution statement。
  - 已按作者确认内容写入主稿。

- [x] 完成 Declaration of competing interest。
  - 主稿 PDF 中保留 Declaration of competing interest。
  - 已使用 Elsevier declarations tool 生成 Word 声明文件：`declaration-of-competing-interests.docx`。
  - 文件内容为作者声明无已知 competing financial interests 或 personal relationships。
  - 投稿时在 attach/upload files 步骤上传该 `.docx` 文件。

- [x] 完成 Funding statement。
  - 已按 Elsevier 推荐句式声明本研究未获得 public、commercial 或 not-for-profit sectors 的 specific grant。

- [x] 核实 Data availability 是否满足 Applied Energy 的 Research Data 要求。
  - 当前文本：代码和数据位于 `https://github.com/landfallbox/ET-PRL.git`。
  - Applied Energy 说明研究数据需存入相关数据仓库、在文中引用并链接；如不能公开，需说明原因。
  - 建议操作：将代码和数据归档到 Zenodo、Mendeley Data 或机构数据仓库，获得 DOI 后更新 Data availability；若 GitHub 已含完整可复现实验数据，也建议生成 release 并绑定 Zenodo DOI。

## 投稿文件包检查

- [x] 准备可编辑主稿源文件。
  - 必传：`ET_PRL_Applied_Energy.tex`
  - 同包依赖：`elsarticle.cls`、`elsarticle-num.bst`、`elsarticle-num-names.bst`、`elsarticle-harv.bst`。
  - 不建议上传：`build/` 下的 `.aux`、`.log`、`.fls`、`.synctex.gz` 等构建中间文件。

- [x] 准备单独 Highlights 文件。
  - 当前文件：`Highlights.tex`
  - 当前满足 3 到 5 条、每条不超过 85 字符的要求；投稿系统中需作为单独 editable file 上传。
  - 已按推荐做法从主稿 PDF 中移除 highlights 环境，避免重复显示。

- [x] 核对所有图文件已准备齐全。
  - 当前投稿目录已有：`fig1_update_strategy_comparison.pdf`、`fig2_plain.pdf`、`fig3_dataset_temporal_characteristics.pdf`、`fig4_dataset_distribution_and_correlation.pdf`、`fig6_t_chws_macro_distribution.pdf`、`fig7_t_chws_delta_analysis.pdf`、`fig8_action_update_interval_distribution.pdf`、`fig9_trigger_load_alignment.pdf`、`fig10_ablation_pareto_scatter.pdf`。
  - 已核对：主稿引用的所有图文件均存在；投稿系统中仍需上传所有图文件，并建议人工检查图片中文字在 5 x 13 cm 或单栏缩放后仍清晰。

- [x] 确认图表版权和数据权限。
  - 已确认：图表由本文代码和自有数据生成，可在投稿系统中按无第三方版权材料处理。
  - 已确认：商业建筑运行数据已获得论文使用与公开权限，并已确认匿名化处理和公开范围。

- [x] 视情况准备 Graphical abstract。
  - 已确认：本次投稿暂不准备 Graphical abstract。
  - Applied Energy 鼓励但非强制；若投稿系统后续要求，可基于方法流程图另行制作。

- [x] 视情况准备 cover letter。
  - 已确认：Applied Energy Guide for Authors 未将 cover letter 列为明确必传项，本次暂不单独准备。
  - 若 Editorial Manager 后续要求填写或上传，可简述论文主题、创新点、适配 Applied Energy 范围，以及未一稿多投声明。

## 稿件内容最终检查

- [x] 全文语言和格式终检。
  - 已统一美式拼写、复合术语连字符、图表标题关键术语、变量下标写法和 PPR/ACR/ARR/MAD 首次释义。
  - 已复扫标题、摘要、关键词、图题、表题、变量符号和缩写一致性。
  - 已确认主稿无 `TODO:`，并重新编译通过。

- [x] 摘要终检。
  - Applied Energy 要求摘要不超过 250 words。
  - 当前统计为 248 words；任何改写后需重新统计。

- [x] 关键词终检。
  - Applied Energy 要求 1 到 7 个关键词。
  - 当前为 6 个关键词。

- [x] 参考文献终检。
  - 已补全 29 条 DOI。
  - 已核对：正文唯一引用 29 条、`bibitem` 29 条，参考文献均含 DOI URL。

- [x] AI 使用声明终检。
  - 当前已放在参考文献之前，符合 Elsevier 要求。
  - 若图形或图形摘要使用了生成式 AI，还需在相关图注和 AI 声明中单独说明。

## 投稿系统内需要人工确认

> 以下事项均在 Elsevier Editorial Manager/投稿网页中完成，本地无需新增或修改文件；保留为线上提交阶段提醒。

- [ ] 确认 submission declaration。
  - 线上勾选/确认稿件未发表、未一稿多投、所有作者同意投稿、机构或项目负责人同意发表。

- [ ] 确认通信作者和所有作者信息与投稿系统完全一致。
  - 线上录入时按主稿标题页信息填写；作者顺序提交后原则上不应再变更。

- [ ] 选择出版模式和费用责任。
  - 线上选择是否 open access，以及 APC 由谁承担。

- [ ] 完成推荐审稿人或回避审稿人信息。
  - 若投稿系统要求，线上填写姓名、单位、邮箱、研究方向和推荐/回避理由。

## 建议但非阻断项

- [ ] 给 GitHub 仓库创建 release，并用 Zenodo 或同类服务生成长期 DOI。
- [ ] 将实验环境、数据字段说明、运行命令和随机种子整理到仓库 README，以增强可复现性。
- [x] 检查 PDF 首页、图表、长表和参考文献 URL 是否存在明显排版溢出。
  - 已检查：LaTeX 日志中无 `Overfull` 提示；PDF 共 50 页，抽查首页、主要图页、长表附近页面和参考文献 URL 页，未见文字越界、图表裁切或长 URL 明显跑出版心。
  - 现有提示为 `Underfull \hbox`、`Infinite glue shrinkage` 和 `hyperref` PDF string warning，不属于明显排版溢出阻断项。

## 最后提交前命令

在投稿目录运行：

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\applied-energy-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_Applied_Energy.tex
```

通过标准：命令退出码为 `0`，日志中无 `LaTeX Error`、`Emergency stop`、`Undefined control sequence`、undefined citation。