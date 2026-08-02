# Applied Thermal Engineering Submission Checklist

Updated: 2026-08-02
Source: Applied Thermal Engineering Guide for Authors and the local submission package.

## Ready in the local package

- [x] Main editable manuscript: `ET_PRL_ATE.tex`.
- [x] Compiled manuscript: `build/ET_PRL_ATE.pdf`.
- [x] Separate editable Highlights file: `Highlights_ATE.tex`.
- [x] Separate cover letter source and compiled preview.
- [x] Competing-interest declaration in `.docx` format.
- [x] All nine manuscript figures prepared as separate PDF files.
- [x] Graphical abstract prepared as a separate optional upload.
- [x] Abstract is 250 words, within the 250-word limit.
- [x] Six English keywords, within the 1 to 7 keyword limit.
- [x] Four Highlights inherited from the Energy and Buildings baseline.
- [x] Main manuscript compiled successfully with XeLaTeX and no fatal LaTeX errors.
- [x] The manuscript states that the evaluation uses a calibrated EnergyPlus model driven by measured chiller operation data; it does not describe the result as a closed-loop field experiment.
- [x] The manuscript distinguishes policy executions from realized setpoint changes and states that actuator wear was not directly measured.
- [x] The manuscript includes CRediT, funding, competing-interest, data-availability, and generative-AI statements.

## Blocking items before online submission

- [ ] Confirm one and only one corresponding author for ATE.
  - The ATE draft currently designates Qiming Fu as the corresponding author.
  - Jianping Chen remains an author but is not marked as corresponding author in this draft.
  - Confirm the choice with all authors, then keep the manuscript, Editorial Manager record, cover letter, and email details consistent.

- [ ] Confirm the complete title-page information.
  - Check author spelling and order.
  - Check all affiliations, postal codes, email addresses, and the corresponding author's full postal address and telephone number.
  - Enter the same author order and affiliations in Editorial Manager.

- [ ] Deposit the research data in a relevant repository and obtain a persistent identifier.
  - ATE applies Research Data Option C: deposit the research data, cite and link it in the article, or explain why sharing is not possible.
  - The current manuscript links to GitHub only. A Zenodo, Mendeley Data, or institutional repository DOI is still recommended and may be required for full compliance.
  - After obtaining the DOI, update the Data availability statement and add a dataset reference if appropriate.

- [ ] Confirm that the Applied Energy rejection is final and the manuscript is not under consideration elsewhere.
  - Do not submit while any transfer, appeal, or concurrent review process remains active.

- [ ] Complete the Elsevier declarations tool and upload the resulting `.docx` declaration.
  - The local declaration file is ready, but the corresponding author should verify that its content still matches the final author list and disclosures.

- [ ] Confirm funding information.
  - The manuscript currently states support from the National Natural Science Foundation of China, No. 62372318.
  - Verify the grant number, wording, and whether other funding or institutional support must be disclosed.

- [ ] Confirm data and figure permissions.
  - Verify that the chiller operation data can be shared in the stated scope and that all figures and graphical-abstract elements are original or properly permitted.
  - Confirm that no third-party material or generative-AI-generated primary-data image is included without the required disclosure.

## Online upload order

1. Enter the final title, abstract, keywords, authors, affiliations, and one corresponding author.
2. Upload the editable manuscript source and the required LaTeX class/bibliography files.
3. Upload the compiled manuscript PDF if requested by the system.
4. Upload `Highlights_ATE.tex` as a separate editable Highlights file.
5. Upload each figure separately using logical figure names.
6. Upload the graphical abstract as an optional separate file.
7. Upload `declaration-of-competing-interests.docx`.
8. Upload the cover letter if the system provides a cover-letter file slot.
9. Complete declarations, data-linking information, suggested or opposed reviewers if requested, and the submission declaration.
10. Review the generated PDF before final submission.

## Final local validation

```powershell
Set-Location -LiteralPath 'd:\code\projects\Event-DQN\docs\papers\applied-thermal-engineering-latex\submission'
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir=build ET_PRL_ATE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build Highlights_ATE.tex
latexmk -interaction=nonstopmode -file-line-error -xelatex -outdir=build cover_letter_ATE.tex
```
