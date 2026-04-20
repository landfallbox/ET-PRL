# AGENTS.md

## Dev Commands

- Install: `uv sync`
- Run any CLI: `uv run <command>`
- Common commands:
  - DQN: `uv run event-dqn-train-dqn`, `event-dqn-test-dqn`, `event-dqn-optimize-dqn`
  - LSTM: `uv run event-dqn-train-lstm`, `event-dqn-test-lstm`, `event-dqn-optimize-lstm`, `event-dqn-predict-lstm`
  - Control: `uv run event-dqn-compare-control`, `event-dqn-test-event-driven`
  - Gate: `uv run event-dqn-prewarm-gate`, `event-dqn-optimize-gate`
  - Ablation: `uv run event-dqn-run-ablation-all`

## Architecture

- Entry points: `pyproject.toml` `[project.scripts]`
- CLI layer: `src/cli/`
- Business modules: `src/dqn/`, `src/cl_predict/`, `src/control_evaluation/`, `src/online_anomaly_detection/`
- Config: `config/*.py`
- Data in: `data/`
- Outputs to: `logs/<experiment>/<mode>/<timestamp>/`

## Key Constraints

- Python 3.13 required (see `.python-version`)
- Uses `uv` for all package management (not pip/poetry)
- `ml-toolkit` is a git dependency (`pyproject.toml:49`)
- Import via full package paths (`src.*`, `config.*`), not relative scripts

## Testing

- No formal test suite exists; validation is done via training/evaluation commands