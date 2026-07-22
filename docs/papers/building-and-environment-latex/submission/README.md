# Building and Environment Submission Draft

This directory contains the Building and Environment transfer draft derived from the rejected Journal of Building Engineering submission. The JOBE files remain unchanged in `docs/papers/jobe-latex/submission/`.

## Main files

- `ET_PRL_BAE.tex`: editable manuscript source.
- `build/ET_PRL_BAE.pdf`: compiled manuscript PDF.
- `Highlights_BAE.tex`: separate editable highlights file.
- `cover_letter_BAE.tex`: journal-specific cover letter source.
- `build/cover_letter_BAE.pdf`: compiled cover letter.
- `TODO.md`: mandatory screening checks and remaining author actions.

The manuscript uses Elsevier's `elsarticle` class. Building and Environment accepts editable LaTeX source and permits double-column formatting for LaTeX submissions. The current source uses the preprint layout with double line spacing for review.

## Build

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\building-and-environment-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_BAE.tex
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build Highlights_BAE.tex
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_BAE.tex
```

## Status

This is not yet submission-ready. Building and Environment accepts only one corresponding author and requires institutional or professional email addresses from all authors. The manuscript currently retains the JOBE author metadata until the author group confirms the corresponding author and valid email addresses.