"""Generate one combined SVG containing two sub-figures and one shared legend.

The script is self-contained and does not require pre-generated child SVG files.
It directly draws:
- Subfigure (a): radar comparison
- Subfigure (b): 2D method lineage positioning
- One shared legend placed on the right side of the full canvas

Output:
- docs/pics/fig_combined_figure1_figure2.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _polar_to_xy(cx: float, cy: float, r: float, angle_rad: float) -> tuple[float, float]:
    return (cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad))


def _polygon(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _map_xy(
    x: float,
    y: float,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float = 0.0,
    x_max: float = 5.0,
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> tuple[float, float]:
    px = left + ((x - x_min) / (x_max - x_min)) * width
    py = top + height - ((y - y_min) / (y_max - y_min)) * height
    return px, py


def _build_radar_body(
    width: float,
    height: float,
    methods: list[tuple[str, str]],
    values_map: dict[str, list[float]],
) -> str:
    cx, cy = 430.0, 430.0
    radius = 240.0
    max_value = 3.0
    dynamic_method = "ETC (Dynamic Threshold)"

    criteria = [
        "Energy Efficiency",
        "Comfort Robustness",
        "Adaptability",
        "Computation Friendliness",
        "Actuator Friendliness",
        "Interpretability",
    ]

    angles = [-math.pi / 2 + i * 2 * math.pi / len(criteria) for i in range(len(criteria))]

    parts: list[str] = []
    for lvl in [1.0, 2.0, 3.0]:
        ring_r = radius * (lvl / max_value)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{ring_r:.2f}" fill="none" stroke="#c7cdd4" stroke-width="2.0"/>'
        )

    parts.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="none" stroke="#000000" stroke-width="2.4"/>'
    )

    for a in angles:
        x2, y2 = _polar_to_xy(cx, cy, radius, a)
        parts.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#d1d5db" stroke-width="1.2"/>'
        )

    for label, a in zip(criteria, angles):
        lx, ly = _polar_to_xy(cx, cy, radius + 30, a)
        anchor = "middle"
        if abs(math.cos(a)) > 0.45:
            anchor = "start" if math.cos(a) > 0 else "end"
        parts.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-family="Times New Roman" font-size="20pt" fill="#374151">{_svg_escape(label)}</text>'
        )

    for name, color in methods:
        vals = values_map[name]
        pts = []
        for v, a in zip(vals, angles):
            rr = radius * (v / max_value)
            pts.append(_polar_to_xy(cx, cy, rr, a))
        fill_opacity = "0.25" if name == dynamic_method else "0.06"
        dash = "" if name == dynamic_method else ' stroke-dasharray="10 7"'
        stroke_width = "4.0" if name == dynamic_method else "2.8"
        parts.append(
            f'<polygon points="{_polygon(pts)}" fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" stroke-width="{stroke_width}"{dash}/>'
        )

    return "\n".join(parts)


def _build_lineage_body(width: float, height: float, methods: list[tuple[str, str]]) -> str:
    margin_left, margin_right = 88.0, 170.0
    margin_top, margin_bottom = 62.0, 82.0

    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_min, x_max = 0.8, 4.8
    y_min, y_max = 0.8, 4.2
    fig2_font_size = "20pt"

    positions = {
        "RBC/PID": (1.0, 1.0),
        "MPC": (1.8, 2.8),
        "TTC": (2.2, 2.2),
        "ETC (Static Threshold)": (3.3, 1.8),
        "ETC (Dynamic Threshold)": (4.4, 3.8),
    }

    parts: list[str] = []

    mid_x = plot_x + plot_w / 2.0
    mid_y = plot_y + plot_h / 2.0
    parts.append(f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w/2:.2f}" height="{plot_h/2:.2f}" fill="#f8fafc"/>')
    parts.append(f'<rect x="{mid_x:.2f}" y="{plot_y:.2f}" width="{plot_w/2:.2f}" height="{plot_h/2:.2f}" fill="#f0fdf4"/>')
    parts.append(f'<rect x="{plot_x:.2f}" y="{mid_y:.2f}" width="{plot_w/2:.2f}" height="{plot_h/2:.2f}" fill="#fff7ed"/>')
    parts.append(f'<rect x="{mid_x:.2f}" y="{mid_y:.2f}" width="{plot_w/2:.2f}" height="{plot_h/2:.2f}" fill="#fef2f2"/>')

    # Explicit coordinate axes with arrowheads.
    x0 = plot_x
    y0 = plot_y + plot_h
    x1 = plot_x + plot_w
    y1 = plot_y
    parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" stroke="#374151" stroke-width="2.6"/>')
    parts.append(f'<polygon points="{x1:.2f},{y0:.2f} {x1-12:.2f},{y0-6:.2f} {x1-12:.2f},{y0+6:.2f}" fill="#374151"/>')
    parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="#374151" stroke-width="2.6"/>')
    parts.append(f'<polygon points="{x0:.2f},{y1:.2f} {x0-6:.2f},{y1+12:.2f} {x0+6:.2f},{y1+12:.2f}" fill="#374151"/>')

    x_label_y = y0 + 24.0
    parts.append(
        f'<text x="{plot_x + plot_w/2:.2f}" y="{x_label_y:.2f}" text-anchor="middle" '
        f'font-family="Times New Roman" font-size="{fig2_font_size}" fill="#111827">Action Sparsity</text>'
    )
    y_label_x = x0 - 22.0
    parts.append(
        f'<text x="{y_label_x}" y="{plot_y + plot_h/2:.2f}" transform="rotate(-90 {y_label_x} {plot_y + plot_h/2:.2f})" text-anchor="middle" '
        f'font-family="Times New Roman" font-size="{fig2_font_size}" fill="#111827">Overall Control Performance</text>'
    )

    ordered = ["RBC/PID", "MPC", "TTC", "ETC (Static Threshold)", "ETC (Dynamic Threshold)"]
    path_points = [
        _map_xy(
            positions[n][0],
            positions[n][1],
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        for n in ordered
    ]
    parts.append(
        f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in path_points)}" fill="none" stroke="#6b7280" stroke-width="2.2" stroke-dasharray="5 4"/>'
    )
    x_prev, y_prev = path_points[-2]
    x_end, y_end = path_points[-1]
    dx, dy = x_end - x_prev, y_end - y_prev
    norm = (dx * dx + dy * dy) ** 0.5
    if norm > 0:
        ux, uy = dx / norm, dy / norm
        bx, by = x_end - 10 * ux, y_end - 10 * uy
        lx, ly = bx - 4 * uy, by + 4 * ux
        rx, ry = bx + 4 * uy, by - 4 * ux
        parts.append(f'<polygon points="{x_end:.2f},{y_end:.2f} {lx:.2f},{ly:.2f} {rx:.2f},{ry:.2f}" fill="#6b7280"/>')

    for name, color in methods:
        px, py = _map_xy(
            positions[name][0],
            positions[name][1],
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="11.0" fill="{color}" stroke="#ffffff" stroke-width="2.2"/>')

        label_dx, label_dy = 12, -10
        if name == "ETC (Dynamic Threshold)":
            label_dx, label_dy = -12, -12
        elif name == "RBC/PID":
            label_dy = 18
        elif name == "ETC (Static Threshold)":
            label_dy = 18

        anchor = "start" if label_dx >= 0 else "end"
        parts.append(
            f'<text x="{px + label_dx:.2f}" y="{py + label_dy:.2f}" text-anchor="{anchor}" '
            f'font-family="Times New Roman" font-size="{fig2_font_size}" fill="#111827">{_svg_escape(name)}</text>'
        )

    return "\n".join(parts)


def combine_svgs(
    output_svg: Path,
    gap: float = 26.0,
    padding: float = 30.0,
    legend_width: float = 400.0,
    background: str = "#ffffff",
) -> None:
    methods: list[tuple[str, str]] = [
        ("RBC/PID", "#1f77b4"),
        ("MPC", "#ff7f0e"),
        ("TTC", "#2ca02c"),
        ("ETC (Static Threshold)", "#d62728"),
        ("ETC (Dynamic Threshold)", "#9467bd"),
    ]

    values_map = {
        "RBC/PID": [1.6, 1.8, 1.0, 2.8, 2.6, 2.9],
        "MPC": [2.3, 2.6, 2.1, 1.3, 1.7, 2.1],
        "TTC": [2.0, 2.9, 1.3, 1.1, 1.0, 1.7],
        "ETC (Static Threshold)": [2.5, 2.1, 1.7, 2.0, 2.3, 2.6],
        "ETC (Dynamic Threshold)": [2.9, 2.4, 2.9, 2.4, 2.8, 2.3],
    }

    # Draw size keeps original internal coordinates; layout size controls inter-panel spacing.
    lw_draw, lh = 1100.0, 840.0
    rw_draw, rh = 1120.0, 760.0
    lw_layout = 700.0
    rw_layout = 980.0
    lbody = _build_radar_body(lw_draw, lh, methods, values_map)
    rbody = _build_lineage_body(rw_draw, rh, methods)

    content_h = max(lh, rh)
    label_band_h = 56.0
    figure_shift_x = 8.0
    left_extra_whitespace = 56.0
    canvas_w = padding + left_extra_whitespace + lw_layout + gap + rw_layout + legend_width + padding + figure_shift_x
    canvas_h = padding + content_h + label_band_h + padding

    # Center each sub-figure vertically within the combined canvas.
    ly = padding + (content_h - lh) / 2.0
    ry = padding + (content_h - rh) / 2.0

    lx = padding + left_extra_whitespace + figure_shift_x
    rx = padding + left_extra_whitespace + lw_layout + gap + figure_shift_x
    # Anchor legend to the actual right edge of subplot-b plotting area,
    # and keep the gap slightly smaller than the gap between subplots.
    b_plot_right_global = rx + 940.0
    legend_gap_to_b = max(gap + 46.0, 64.0)

    legend_box_w = legend_width - 56.0
    marker_size = 18.0
    legend_item_step = 34.0
    legend_inner_pad = 16.0
    legend_box_h = legend_inner_pad * 2 + marker_size + (len(methods) - 1) * legend_item_step
    legend_box_x = b_plot_right_global + legend_gap_to_b
    legend_right_margin = 90.0
    max_legend_x = canvas_w - legend_right_margin - legend_box_w
    if legend_box_x > max_legend_x:
        legend_box_x = max_legend_x
    # Keep legend vertically centered, then shift slightly upward.
    legend_box_y = padding + (content_h - legend_box_h) / 2.0 - 26.0
    legend_x = legend_box_x + 16.0
    legend_first_center_y = legend_box_y + legend_inner_pad + marker_size / 2.0

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">'
    )
    lines.append(f'<rect width="100%" height="100%" fill="{background}"/>')

    # Subfigure A (left)
    lines.append(f'<g transform="translate({lx:.2f},{ly:.2f})">')
    lines.append(lbody)
    lines.append("</g>")

    # Subfigure B (right)
    lines.append(f'<g transform="translate({rx:.2f},{ry:.2f})">')
    lines.append(rbody)
    lines.append("</g>")

    # Shared legend on the top-right side of the combined figure.
    lines.append(
        f'<rect x="{legend_box_x:.2f}" y="{legend_box_y:.2f}" width="{legend_box_w:.2f}" height="{legend_box_h:.2f}" fill="#ffffff" stroke="#000000" stroke-opacity="0.5" stroke-width="1.6" rx="8" ry="8"/>'
    )
    for i, (name, color) in enumerate(methods):
        yy = legend_first_center_y + i * legend_item_step
        lines.append(
            f'<rect x="{legend_x:.2f}" y="{yy - marker_size / 2.0:.2f}" width="{marker_size:.0f}" height="{marker_size:.0f}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.6"/>'
        )
        lines.append(
            f'<text x="{legend_x + 28:.2f}" y="{yy + 2:.2f}" font-family="Times New Roman" font-size="20pt" fill="#1f2937">{_svg_escape(name)}</text>'
        )

    # Subfigure titles and labels centered below each subplot.
    a_center_x = lx + 430.0
    b_center_x = rx + rw_layout / 2.0
    title_y = padding + content_h + 16.0
    label_y = padding + content_h + 20.0
    lines.append(
        f'<text x="{a_center_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Times New Roman" font-size="20pt" fill="#111827">(a) Radar Chart of Method Performance</text>'
    )
    lines.append(
        f'<text x="{b_center_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Times New Roman" font-size="20pt" fill="#111827">(b) Evolutionary Lineage </text>'
    )

    lines.append("</svg>")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one large SVG with two sub-figures and a shared legend.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pics/fig_combined_figure1_figure2.svg"),
        help="Output combined SVG path.",
    )
    parser.add_argument("--gap", type=float, default=210.0, help="Gap between two sub-figures.")
    parser.add_argument("--padding", type=float, default=30.0, help="Outer padding of combined canvas.")
    parser.add_argument("--legend-width", type=float, default=400.0, help="Reserved width for shared legend area.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_svgs(
        output_svg=args.output,
        gap=args.gap,
        padding=args.padding,
        legend_width=args.legend_width,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
