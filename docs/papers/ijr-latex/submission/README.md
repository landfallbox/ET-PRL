# International Journal of Refrigeration 投稿草稿

本目录包含 International Journal of Refrigeration（Elsevier，IJR）投稿草稿，基于 Applied Energy 投稿版本改造（两者同为 Elsevier `elsarticle` 模板，格式一致）。Applied Energy 原稿保留在 `docs/papers/applied-energy-latex/submission/`，未改动。

## 主要文件

- `ET_PRL_IJR.tex`：可编辑主稿源文件。
- `cover_letter_IJR.tex`：期刊定制封面信源文件（突出冷冻水/制冷系统侧契合）。
- `Highlights.tex`：Elsevier 要求的单独 Highlights 文件（3–5 条，每条 ≤ 85 字符）。
- `declaration-of-competing-interests.docx`：Elsevier declarations tool 生成的利益冲突声明，投稿时在 attach/upload 步骤上传。
- `elsarticle.cls` / `elsarticle-num.bst` / `elsarticle-num-names.bst` / `elsarticle-harv.bst`：模板依赖。
- `fig1`–`fig10` 的 PDF：正文引用的全部图文件。
- `TODO.md`：投稿前阻断项与作者待办。

## IJR 格式要点（与 Applied Energy 版的差异）

- 主稿类与参考文献风格沿用 `elsarticle`（`preprint,12pt,number`），编号参考文献，与 Applied Energy 版一致，无需改动。
- 摘要：沿用 Applied Energy 版（≤ 250 词，当前 248 词），IJR 同样要求 ≤ 250 词。
- 关键词：沿用 6 个（IJR 要求 1–7 个）。
- 通讯作者邮箱：已改为机构邮箱 `fqm@mail.usts.edu.cn`、`alan@mail.usts.edu.cn`（沿用 IJGE 版确认口径）。
- 末尾声明顺序（Elsevier）：CRediT 贡献 → 利益冲突声明 → 数据可用性 → 致谢 → 生成式 AI 声明 → 参考文献。
- 本目录刻意保持扁平，Elsevier Editorial Manager 对子文件夹较敏感。

## 构建

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\ijr-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_IJR.tex
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_IJR.tex
```

## 投稿系统

- 投稿入口：Elsevier Editorial Manager，`https://www.editorialmanager.com/ijref/`（或 ScienceDirect 期刊页 "Submit your article"）。
- 出版模式：混合刊，投稿时选择 **Subscription（非开放获取）** 通道即可免 APC。
- 本目录不存放投稿系统账号密码；如需记录，请另建 `CREDENTIALS.local.md` 并加入 `.gitignore`，勿提交、勿外传。

## 状态

主稿可编译。投稿前需完成 `TODO.md` 中的阻断项（分区版本确认、数据 DOI、作者终确认等）。
