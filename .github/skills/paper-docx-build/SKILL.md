---
name: paper-docx-build
description: "Build or troubleshoot a portable Word DOCX paper from Markdown. Use when asked to convert Markdown to Word, export a paper as .docx, create a self-contained DOCX conversion workflow, or diagnose Pandoc/python-docx ET-PRL algorithm-table formatting."
argument-hint: "[input Markdown path] [optional output DOCX path]"
---

# Portable Paper DOCX Build

## When to Use

- Convert a paper Markdown source to a Word `.docx` file.
- Convert any compatible paper Markdown file into DOCX.
- Share a self-contained Markdown-to-DOCX workflow with another VS Code Copilot user.
- Diagnose missing Pandoc, missing document dependencies, or algorithm-table formatting failures.

## Workflow

1. Copy the complete `paper-docx-build` Skill directory into the recipient workspace at `.github/skills/paper-docx-build/`.
2. Install Pandoc and ensure it is available on `PATH`. `uv` is the recommended Python environment and dependency tool because it can install the required packages for one command.
3. From the recipient workspace root, use the recommended `uv` command for generic conversion:

   ```powershell
   uv run --with-requirements .github/skills/paper-docx-build/requirements.txt python `
     .github/skills/paper-docx-build/scripts/build_docx.py `
     --input <input-markdown-path> `
     --output <output-docx-path>
   ```

4. Conda, `venv`, or another Python environment tool may also be used. In that case, install [Python dependencies](./requirements.txt) with that tool first, then replace `uv run --with-requirements ... python` in the commands below with the environment's Python executable.
5. Use [Markdown Input Conventions](./references/markdown-format.md) for image paths, TeX math, and the optional ET-PRL algorithm placeholder.
6. Report the generated DOCX path and any unresolved dependency or formatting error.

## Build Behavior

`scripts/build_docx.py` is self-contained and performs the following conversion:

1. Uses Pandoc with `markdown+tex_math_dollars` to convert Markdown into a temporary DOCX.
2. Uses `python-docx` to apply full borders and selected table alignment rules.
3. When `--algorithm-title` is supplied, runs the bundled `scripts/build_algorithm_docx.py` and replaces the matching title plus fenced code block with the native Word ET-PRL algorithm table containing OMML equations.
4. Stores all intermediate files in a system temporary directory and writes only the requested output DOCX.

## ET-PRL Algorithm Mode

The bundled algorithm table reproduces the ET-PRL online control procedure, including its fixed pseudocode and formulas. Enable it only when the manuscript contains the corresponding ET-PRL algorithm placeholder:

```powershell
uv run --with-requirements .github/skills/paper-docx-build/requirements.txt python `
   .github/skills/paper-docx-build/scripts/build_docx.py `
   --input <input-markdown-path> `
   --output <output-docx-path> `
   --algorithm-title "Algorithm 1. Event-triggered predictive reinforcement learning with unsupervised dynamic event gating (ET-PRL) online control procedure"
```

Do not set `--algorithm-title` for other algorithms: generic conversion will retain their Pandoc-rendered Markdown content.

## Included Resources

- [DOCX builder](./scripts/build_docx.py): generic Pandoc conversion and optional table injection.
- [ET-PRL algorithm builder](./scripts/build_algorithm_docx.py): creates the OMML equation table.
- [Python dependencies](./requirements.txt): standalone `python-docx` and `lxml` dependency list.
- [Markdown Input Conventions](./references/markdown-format.md): portable input and placeholder requirements.

## Troubleshooting

- `pandoc not found on PATH`: install Pandoc, restart the terminal, and rerun the build.
- Missing `docx` module: use the recommended `uv run --with-requirements ...` command, or install [requirements.txt](./requirements.txt) into the active Conda, `venv`, or other Python environment.
- Target DOCX is locked by Word: close it or provide `--output` with a different filename.
- Algorithm title cannot be found: ensure the `--algorithm-title` text is contained in the Markdown title and is followed by one fenced code block before the next heading.
