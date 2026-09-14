"""Compatibility entrypoint for split T_chws figure generation.

This script now delegates to two dedicated scripts:
- Figure 5-2-3 macro distribution
- Figure 5-2-4 delta analysis

Usage:
    uv run python src/plotting/plot_t_chws_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from et_prl.plotting.control.plot_t_chws_macro_distribution import (
    generate_macro_figure,
    load_control_results as load_macro_results,
)
from et_prl.plotting.control.plot_t_chws_delta_analysis import (
    generate_delta_figure,
    load_control_results as load_delta_results,
)


def main(results_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'logs' / 'control_compare' / '20260402_222140'
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'docs' / 'pics'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading control results from {results_dir}...')
    macro_results = load_macro_results(results_dir)
    delta_results = load_delta_results(results_dir)

    output_a = output_dir / 'fig6_t_chws_macro_distribution.svg'
    output_bc = output_dir / 'fig7_t_chws_delta_analysis.svg'

    generate_macro_figure(macro_results, output_a)
    generate_delta_figure(delta_results, output_bc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split T_chws figures (Figure 5-2-3 and 5-2-4).')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
