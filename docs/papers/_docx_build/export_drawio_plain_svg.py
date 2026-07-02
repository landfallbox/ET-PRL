#!/usr/bin/env python3
"""Export a diagrams.net/draw.io diagram to a Word-friendly plain SVG.

The default path intentionally goes through PDF. Direct draw.io SVG exports can
contain HTML label layers as foreignObject nodes, which are fragile in Word and
third-party previewers. PDF import flattens those labels into vector geometry
before Inkscape writes the final plain SVG.

By default, draw.io exports a PDF whose page is fitted to the complete diagram
content, then Inkscape converts that page to a plain SVG. Use
--preserve-source-page only when the source .drawio page bounds should be kept.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import urllib.parse
import zlib
from pathlib import Path


BUILD_DIR = Path(__file__).resolve().parent
ROOT = BUILD_DIR.parents[2]
DEFAULT_INPUT = ROOT / "docs" / "pics" / "fig2.drawio"
DEFAULT_OUTPUT = ROOT / "docs" / "pics" / "fig2_plain.svg"
SVG_NS = "http://www.w3.org/2000/svg"

DRAWIO_ENV = "DRAWIO_EXE"
INKSCAPE_ENV = "INKSCAPE_EXE"

RISKY_TOKENS = {
    "foreignObject HTML layer": "<foreignObject",
    "embedded draw.io source XML": "<mxfile",
    "encoded draw.io source XML": "&lt;mxfile",
    "diagrams.net metadata": "app.diagrams.net",
    "XHTML namespace": "http://www.w3.org/1999/xhtml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a .drawio diagram to a plain SVG suitable for Word/pandoc workflows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input .drawio file. Defaults to docs/pics/fig2.drawio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output plain SVG path. Defaults to docs/pics/fig2_plain.svg.",
    )
    parser.add_argument(
        "--page-index",
        type=int,
        default=1,
        help="One-based draw.io page index to export.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help="Export border in pixels around the fitted diagram/page.",
    )
    parser.add_argument(
        "--crop-to-content",
        action="store_true",
        help="Deprecated alias for the default full-diagram page export.",
    )
    parser.add_argument(
        "--preserve-source-page",
        action="store_true",
        help="Use the page bounds stored in the source .drawio file without fitting the page to all content.",
    )
    parser.add_argument(
        "--source-canvas-svg",
        type=Path,
        default=None,
        help="Optional SVG whose canvas should be preserved instead of the temporary .drawio reference export.",
    )
    parser.add_argument(
        "--no-source-canvas",
        action="store_true",
        help="Do not preserve the source/reference SVG canvas around the cleaned diagram.",
    )
    parser.add_argument(
        "--drawio",
        type=Path,
        default=None,
        help=f"Path to draw.io/diagrams.net executable. Overrides {DRAWIO_ENV}.",
    )
    parser.add_argument(
        "--inkscape",
        type=Path,
        default=None,
        help=f"Path to Inkscape executable. Overrides {INKSCAPE_ENV}.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw draw.io PDF next to the output for troubleshooting.",
    )
    parser.add_argument(
        "--direct-svg",
        action="store_true",
        help="Use direct draw.io SVG export before Inkscape cleanup. This is mainly for diagnostics.",
    )
    return parser.parse_args()


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _first_existing(paths: list[str | Path | None]) -> Path | None:
    for value in paths:
        found = _existing_path(value)
        if found is not None:
            return found
    return None


def _which_path(names: list[str]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _find_drawio(explicit: Path | None) -> Path | None:
    return _first_existing(
        [
            explicit,
            os.environ.get(DRAWIO_ENV),
            _which_path(["drawio", "draw.io", "diagrams.net"]),
            Path("C:/Program Files/draw.io/draw.io.exe"),
            Path("C:/Program Files/diagrams.net/diagrams.net.exe"),
            Path("C:/Users") / os.environ.get("USERNAME", "") / "AppData/Local/Programs/draw.io/draw.io.exe",
        ]
    )


def _find_inkscape(explicit: Path | None) -> Path | None:
    return _first_existing(
        [
            explicit,
            os.environ.get(INKSCAPE_ENV),
            _which_path(["inkscape"]),
            Path("C:/Program Files/Inkscape/bin/inkscape.exe"),
            Path("C:/Program Files/Inkscape/inkscape.exe"),
        ]
    )


def _install_hint() -> str:
    return (
        "Install the required CLI tools, then rerun this command.\n"
        "PowerShell examples:\n"
        "  winget install --id JGraph.Draw -e --accept-package-agreements --accept-source-agreements\n"
        "  winget install --id Inkscape.Inkscape -e --accept-package-agreements --accept-source-agreements\n"
        f"You can also set {DRAWIO_ENV} and {INKSCAPE_ENV} to explicit executable paths."
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _parse_svg_length(value: str | None) -> tuple[float, str]:
    if value is None:
        raise RuntimeError("SVG length is missing.")
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(.*)\s*", value)
    if match is None:
        raise RuntimeError(f"Cannot parse SVG length: {value}")
    return float(match.group(1)), match.group(2)


def _svg_canvas_size(path: Path) -> tuple[float, float]:
    root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]

    width, _ = _parse_svg_length(root.get("width"))
    height, _ = _parse_svg_length(root.get("height"))
    return width, height


def _decode_diagram_model(diagram: ET.Element) -> ET.Element:
    children = list(diagram)
    if children and children[0].tag.endswith("mxGraphModel"):
        return children[0]

    data = (diagram.text or "").strip()
    if data.startswith("<mxGraphModel"):
        return ET.fromstring(data)

    compressed = base64.b64decode(data)
    encoded_xml = zlib.decompress(compressed, -15).decode("utf-8")
    model_xml = urllib.parse.unquote(encoded_xml)
    return ET.fromstring(model_xml)


def _content_bounds(model: ET.Element) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    xe: list[float] = []
    ye: list[float] = []

    for cell in model.findall(".//mxCell"):
        geometry = cell.find("mxGeometry")
        if geometry is None:
            continue

        if geometry.get("width") is not None or geometry.get("height") is not None:
            x = float(geometry.get("x", "0"))
            y = float(geometry.get("y", "0"))
            width = float(geometry.get("width", "0"))
            height = float(geometry.get("height", "0"))
            xs.append(x)
            ys.append(y)
            xe.append(x + width)
            ye.append(y + height)

        for point in geometry.findall(".//mxPoint"):
            if point.get("x") is None or point.get("y") is None:
                continue
            x = float(point.get("x", "0"))
            y = float(point.get("y", "0"))
            xs.append(x)
            ys.append(y)
            xe.append(x)
            ye.append(y)

    if not xs:
        raise RuntimeError("No drawable geometry found in the selected draw.io page.")
    return min(xs), min(ys), max(xe), max(ye)


def _drawio_page_grid_canvas_size(input_path: Path, *, page_index: int) -> tuple[float, float]:
    root = ET.fromstring(input_path.read_text(encoding="utf-8"))
    diagrams = root.findall("diagram")
    selected_index = page_index - 1
    if selected_index < 0 or selected_index >= len(diagrams):
        raise RuntimeError(f"Page index {page_index} is out of range for {input_path}.")

    model = _decode_diagram_model(diagrams[selected_index])
    min_x, min_y, max_x, max_y = _content_bounds(model)
    content_width = max_x - min_x
    content_height = max_y - min_y
    page_width = float(model.get("pageWidth", "0"))
    page_height = float(model.get("pageHeight", "0"))

    if page_width <= 0 or page_height <= 0:
        return math.ceil(content_width), math.ceil(content_height)

    canvas_width = math.ceil(content_width / page_width) * page_width
    canvas_height = math.ceil(content_height / page_height) * page_height
    return canvas_width, canvas_height


def _write_canvas_reference_svg(path: Path, *, width: float, height: float) -> None:
    path.write_text(
        f'<svg xmlns="{SVG_NS}" width="{_format_number(width)}" height="{_format_number(height)}" '
        f'viewBox="0 0 {_format_number(width)} {_format_number(height)}" />',
        encoding="utf-8",
    )


def _source_canvas_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        return path if path.exists() else None
    return None


def _preserve_source_canvas(output_path: Path, source_canvas_path: Path, reference_svg_path: Path) -> None:
    source_width, source_height = _svg_canvas_size(source_canvas_path)
    reference_width, reference_height = _svg_canvas_size(reference_svg_path)

    if source_width <= reference_width and source_height <= reference_height:
        return

    tree = ET.parse(output_path)
    root = tree.getroot()
    output_width, width_unit = _parse_svg_length(root.get("width"))
    output_height, height_unit = _parse_svg_length(root.get("height"))
    target_width = output_width * source_width / reference_width
    target_height = output_height * source_height / reference_height
    dx = (target_width - output_width) / 2
    dy = (target_height - output_height) / 2

    ET.register_namespace("", SVG_NS)
    root.set("width", f"{_format_number(target_width)}{width_unit}")
    root.set("height", f"{_format_number(target_height)}{height_unit}")
    root.set("viewBox", f"0 0 {_format_number(target_width)} {_format_number(target_height)}")

    children = list(root)
    for child in children:
        root.remove(child)

    background = ET.Element(f"{{{SVG_NS}}}rect")
    background.set("width", _format_number(target_width))
    background.set("height", _format_number(target_height))
    background.set("fill", "#ffffff")
    root.append(background)

    wrapper = ET.Element(f"{{{SVG_NS}}}g")
    wrapper.set("transform", f"translate({_format_number(dx)} {_format_number(dy)})")
    for child in children:
        wrapper.append(child)
    root.append(wrapper)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(
        "Preserved source SVG canvas: "
        f"{source_canvas_path.name} {source_width:g} x {source_height:g} "
        f"from fitted {reference_width:g} x {reference_height:g}"
    )


def _export_raw_pdf(
    drawio: Path,
    input_path: Path,
    raw_pdf: Path,
    *,
    page_index: int,
    border: int,
    crop_to_content: bool,
) -> None:
    command = [
        str(drawio),
        "--export",
        "--format",
        "pdf",
        "--page-index",
        str(page_index),
        "--border",
        str(border),
        "--output",
        str(raw_pdf),
        str(input_path),
    ]
    if crop_to_content:
        command.insert(4, "--crop")
    _run(command)


def _export_raw_svg(
    drawio: Path,
    input_path: Path,
    raw_svg: Path,
    *,
    page_index: int,
    border: int,
    crop_to_content: bool,
) -> None:
    command = [
        str(drawio),
        "--export",
        "--format",
        "svg",
        "--page-index",
        str(page_index),
        "--border",
        str(border),
        "--output",
        str(raw_svg),
        str(input_path),
    ]
    if crop_to_content:
        command.insert(4, "--crop")
    _run(command)


def _export_plain_svg(inkscape: Path, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(inkscape),
            str(input_path),
            "--pdf-poppler",
            "--export-type=svg",
            "--export-area-page",
            "--export-plain-svg",
            "--export-text-to-path",
            f"--export-filename={output_path}",
        ]
    )


def _validate_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeError(f"Output is not valid XML: {path}: {exc}") from exc

    if not root.tag.endswith("svg"):
        raise RuntimeError(f"Output root element is not SVG: {root.tag}")

    found = [name for name, token in RISKY_TOKENS.items() if token in text]
    if found:
        details = ", ".join(found)
        raise RuntimeError(f"Output SVG still contains risky structures: {details}")


def main() -> int:
    args = _parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    drawio = _find_drawio(args.drawio)
    inkscape = _find_inkscape(args.inkscape)
    missing = []
    if drawio is None:
        missing.append("draw.io/diagrams.net CLI")
    if inkscape is None:
        missing.append("Inkscape CLI")
    if missing:
        print(f"ERROR: Missing required tool(s): {', '.join(missing)}")
        print(_install_hint())
        return 1

    print(f"Using draw.io: {drawio}")
    print(f"Using Inkscape: {inkscape}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    crop_to_content = args.crop_to_content or not args.preserve_source_page
    source_canvas = None if args.no_source_canvas else _source_canvas_path(args.source_canvas_svg)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_svg = Path(tmpdir) / "drawio_reference.svg"
            if not args.no_source_canvas and crop_to_content:
                if source_canvas is None:
                    canvas_width, canvas_height = _drawio_page_grid_canvas_size(input_path, page_index=args.page_index)
                    source_canvas = Path(tmpdir) / "drawio_page_grid_canvas.svg"
                    _write_canvas_reference_svg(source_canvas, width=canvas_width, height=canvas_height)
                    print(f"Using draw.io page-grid canvas: {canvas_width:g} x {canvas_height:g}")

                _export_raw_svg(
                    drawio,
                    input_path,
                    reference_svg,
                    page_index=args.page_index,
                    border=args.border,
                    crop_to_content=True,
                )
                if source_canvas is None:
                    source_canvas = reference_svg

            if args.direct_svg:
                raw_input = Path(tmpdir) / "drawio_raw.svg"
                if crop_to_content:
                    print("Exporting draw.io diagram directly to full-diagram fitted SVG...")
                else:
                    print("Exporting draw.io diagram directly to source-page SVG...")
                _export_raw_svg(
                    drawio,
                    input_path,
                    raw_input,
                    page_index=args.page_index,
                    border=args.border,
                    crop_to_content=crop_to_content,
                )
            else:
                raw_input = Path(tmpdir) / "drawio_raw.pdf"
                if crop_to_content:
                    print("Exporting draw.io diagram to full-diagram fitted PDF...")
                else:
                    print("Exporting draw.io diagram to source-page PDF...")
                _export_raw_pdf(
                    drawio,
                    input_path,
                    raw_input,
                    page_index=args.page_index,
                    border=args.border,
                    crop_to_content=crop_to_content,
                )

            if args.keep_raw:
                debug_raw = output_path.with_name(f"{output_path.stem}.raw{raw_input.suffix}")
                shutil.copyfile(raw_input, debug_raw)
                print(f"Raw export kept at: {debug_raw}")

            print("Converting to Inkscape plain SVG with text converted to paths...")
            _export_plain_svg(inkscape, raw_input, output_path)

            if source_canvas is not None and crop_to_content:
                _preserve_source_canvas(output_path, source_canvas, reference_svg)

        print("Validating plain SVG...")
        _validate_svg(output_path)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: External command failed with exit code {exc.returncode}")
        return exc.returncode or 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())