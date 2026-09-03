# International Journal of Green Energy 投稿草稿

本目录包含 International Journal of Green Energy（T&F，期刊代码 LJGE）投稿草稿，基于 JOBE 稿件改造。JOBE 原稿保留在 `docs/papers/jobe-latex/submission/`，未改动。

## 主要文件

- `ET_PRL_IJGE.tex`：可编辑主稿源文件。
- `build/ET_PRL_IJGE.pdf`：编译后的主稿 PDF。
- `cover_letter_IJGE.tex`：期刊定制封面信源文件。
- `build/cover_letter_IJGE.pdf`：编译后的封面信 PDF。
- `TODO.md`：投稿前阻断项与作者待办。

## IJGE 格式要点（与 JOBE 版的差异）

- 摘要：**非结构化，约 100 词**（已压缩至约 115 词，投稿前可再精简）。
- **正文不写关键词**：IJGE 由编辑部在出版阶段自行分配关键词，主稿已移除 `\begin{keyword}` 块。
- 参考文献：**T&F CSE（Chicago author-date）** 风格。正文引用写作 `(Author Year)`，文献列表采用悬挂缩进 author-date 条目（`reflist` 环境），不再使用 `\bibitem` 编号。
- 末尾声明顺序：CRediT 贡献声明 → 致谢 → 利益冲突声明 → 数据可用性 → 生成式 AI 使用声明 → 参考文献。
- 主稿使用 `elsarticle` 类（IJGE 接受 LaTeX；其“官方模板”链接实际指向另一期刊，故复用现有 elsarticle 基础设施）。

## 构建

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\ijge-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_IJGE.tex
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_IJGE.tex
```

## 状态

尚未达到可投稿状态。IJGE 封面信要求列出**每位作者的机构邮箱**并说明与期刊范围的契合；当前封面信已列出 6 位作者邮箱与 CRediT 贡献，但部分邮箱仍为通用邮箱（见 TODO.md 阻断项）。

## 投稿系统

投稿系统账号与密码见本目录 `CREDENTIALS.local.md`（已加入 .gitignore，不会进入版本库）。