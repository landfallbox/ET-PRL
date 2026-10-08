# Journal of Building Engineering Submission Draft

This directory contains the JOBE-specific revision of the former Applied Energy submission. The original Applied Energy files remain unchanged in `docs/papers/et-prl/submissions/applied-energy-latex/submission/`.

## Main files

- `ET_PRL_JOBE.tex`: editable manuscript source.
- `build/ET_PRL_JOBE.pdf`: compiled manuscript PDF.
- `Highlights_JOBE.tex`: separate editable highlights file.
- `cover_letter_JOBE.tex`: editable cover letter source.
- `build/cover_letter_JOBE.pdf`: compiled cover letter.
- `declaration-of-competing-interests.docx`: Elsevier declaration generated for the same author group.

The manuscript uses Elsevier's `elsarticle` class because JOBE accepts LaTeX source and permits double-column formatting for LaTeX submissions. The current manuscript uses the preprint layout for review.

## Build

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\jobe-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_JOBE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build Highlights_JOBE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_JOBE.tex
```

Do not upload LaTeX intermediate files from `build/`. Upload the editable sources, manuscript PDF, declaration, and all separately requested figure files.