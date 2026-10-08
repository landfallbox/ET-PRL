# Applied Thermal Engineering Submission Package

Journal: Applied Thermal Engineering
Article type: Original Research Article

## Main files

- `ET_PRL_ATE.tex`: editable manuscript source.
- `build/ET_PRL_ATE.pdf`: compiled manuscript for visual review and submission preview.
- `Highlights_ATE.tex`: separate editable highlights source.
- `build/Highlights_ATE.pdf`: highlights preview.
- `cover_letter_ATE.tex`: editable cover letter source.
- `build/cover_letter_ATE.pdf`: cover letter preview.
- `declaration-of-competing-interests.docx`: declaration generated for the author group.

## Figures and graphical abstract

Upload each figure as a separate file:

- `Figure_1.pdf`
- `Figure_2.pdf`
- `Figure_3.pdf`
- `Figure_4.pdf`
- `Figure_5.pdf`
- `Figure_6.pdf`
- `Figure_7.pdf`
- `Figure_8.pdf`
- `Figure_9.pdf`

The graphical abstract is optional but prepared in editable and preview formats. The recommended upload candidate is `Graphical_Abstract_ATE_editable_final.pdf`; retain the PPTX and SVG files as editable sources if the submission system requests them.

## Build commands

Run from this directory:

```powershell
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_ATE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build Highlights_ATE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_ATE.tex
```

Do not upload LaTeX intermediate files from `build/`, including `.aux`, `.log`, `.fls`, `.out`, `.spl`, `.synctex.gz`, and `.xdv` files. The `.cls` and `.bst` files are included so the editable manuscript source can be compiled independently.

## ATE requirements checked

- Abstract: 250 words.
- Keywords: 6 terms.
- Highlights: 4 bullets inherited from the Energy and Buildings baseline.
- Graphical abstract: prepared at 2500 x 1000 pixels, which follows the required 531 x 1328 pixel proportional format.
- Figures: supplied as separate PDF files.
- CRediT, funding, data availability, competing-interest, and generative-AI statements: included in the manuscript.
- Submission portal: https://submit.elsevier.com/ATE

The official Guide for Authors was checked on 2026-08-02. See `TODO.md` for items that still require author confirmation or online action.
