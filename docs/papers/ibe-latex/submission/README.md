# Indoor and Built Environment 投稿草稿

本目录包含 Indoor and Built Environment（Sage，期刊代码 IBE，ISSN 1420-326X）投稿草稿，基于 IJR 稿件改造。IJR 原稿保留在 `docs/papers/ijr-latex/submission/`，未改动。

## 主要文件

- `ET_PRL_IBE.tex`：可编辑主稿源文件。
- `build/ET_PRL_IBE.pdf`：编译后的主稿 PDF。
- `cover_letter_IBE.tex`：期刊定制封面信源文件。
- `build/cover_letter_IBE.pdf`：编译后的封面信 PDF。
- `README.md`：本文件。
- `TODO.md`：投稿前阻断项与作者待办。

## iBE 格式要点（与 IJR 版的差异）

- 摘要：**非结构化，约 200 词**（iBE 要求）。当前沿用 IJR 原版摘要（248 词，未做压缩），超出 iBE 约 200 词要求，投稿前需决定是否精简（见 TODO.md）。
- 关键词：**至少 5 个**，列在摘要后（沿用 6 个）。
- 参考文献：**Sage Vancouver 风格（编号制）**，与 IJR 版编号 `\bibitem` 结构一致，沿用现有编号格式；细节差异（作者/题名/卷期/页码/DOI 的精确排布）投稿前按 Sage Vancouver 规范核对（见 TODO.md）。
- 末尾声明：iBE 要求 `Statements and Declarations` 节，含子标题 Ethical considerations / Consent to participate / Consent for publication / Declaration of conflicting interest / Funding statement / Data availability（不适用项写 "Not applicable"）。本稿已按此结构重组，并保留 CRediT 贡献声明、Acknowledgements、生成式 AI 使用声明。
- 出版模式：混合刊（Subscription），投稿时选择非开放获取通道免 APC；单盲审稿。
- 主稿使用 `elsarticle` 类（iBE 接受 LaTeX；Sage 提供官方 LaTeX 模板，本稿复用现有 elsarticle 基础设施）。

## 构建

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\ibe-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_IBE.tex
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_IBE.tex
```

## 投稿系统

- 投稿入口：Sage Track（ScholarOne），`https://mc.manuscriptcentral.com/ibe`（或期刊页 "Submit Manuscript"）。
- 出版模式：混合刊，投稿时选择 **Subscription（非开放获取）** 通道即可免 APC。
- 本目录不存放投稿系统账号密码；如需记录，请另建 `CREDENTIALS.local.md` 并加入 `.gitignore`，勿提交、勿外传。

## 状态

主稿与封面信可编译。投稿前需完成 `TODO.md` 中的阻断项（分区确认、数据 DOI、参考文献 Sage Vancouver 细节核对、作者终确认等）。
