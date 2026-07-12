# ET-PRL

This repository contains the source code for the ET-PRL experiments prepared for journal submission. It provides the event-triggered predictive reinforcement learning workflow, including LSTM load prediction, DQN-based control, online anomaly/event gating, baseline control comparisons, and ablation experiments.

## Repository Contents

- `src/`: experiment code, command-line entry points, model training/evaluation, control strategies, anomaly/event gate, and plotting utilities.
- `config/`: experiment configuration classes.
- `pyproject.toml` and `uv.lock`: Python environment and dependency lock files.
- `data/README.md`: expected dataset layout and data availability notes.

The repository intentionally excludes editor settings, agent instructions, manuscript drafts, LaTeX build outputs, local logs, trained checkpoints, and other development-only artifacts.

## Environment

Python 3.13 is required. The project uses `uv` for dependency management.

```powershell
uv sync
```

The project depends on `ml-toolkit`, which is referenced as a Git dependency in `pyproject.toml`.

## Data

The dataset required by the experiment workflows is included in this repository under `data/`.

See `data/README.md` for the expected file layout. For journal publication, consider archiving the final GitHub release with Zenodo or another research repository if a DOI is required.

## Outputs

Runtime outputs are written to `logs/<experiment>/<mode>/<timestamp>/` and are ignored by Git. Typical outputs include configuration snapshots, logs, metrics, training history, figures, and model checkpoints.

## Reproducibility Notes

- Configuration defaults are defined in `config/`.
- The random seed is set in the configuration classes where applicable.
- CUDA/cuDNN deterministic behavior can be controlled through configuration.
- Exact numeric results can vary across hardware, CUDA versions, and PyTorch versions.
