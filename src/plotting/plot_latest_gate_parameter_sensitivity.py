"""Generate the publication-quality gate sensitivity figures from the latest results.

Usage:
    uv run python src/plotting/plot_latest_gate_parameter_sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting.gate_parameter_sensitivity_publication import generate_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the publication-quality gate parameter sensitivity figures.")
    parser.add_argument("--results-path", type=Path, default=None, help="Sensitivity results CSV or directory")
    parser.add_argument("--output-path", type=Path, default=None, help="Figure output path")
    args = parser.parse_args()

    generate_figure(results_path=args.results_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
