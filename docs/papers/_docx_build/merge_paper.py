#!/usr/bin/env python3
"""
将 build_algorithm_docx.py 生成的算法表格注入 pandoc 转换后的论文 docx。

流程:
    1. 运行 build_algorithm_docx.py → generated/algorithm1_for_merge.docx
  2. pandoc 将 小论文.md → 临时 docx
    3. 为 pandoc 生成的普通论文表格补全所有框线
    4. 在临时 docx 中找到算法标题段落 ("Algorithm 1. Event-Triggered...")
    5. 删除算法标题至其后第一个 fenced code block 结束之间的旧文本
    6. 将其替换为算法 docx 中的完整表格
    7. 保存到 小论文.docx
"""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


BUILD_DIR = Path(__file__).resolve().parent
PAPER_DIR = BUILD_DIR.parent

ALGO_BUILD_SCRIPT = BUILD_DIR / "build_algorithm_docx.py"
ALGO_DOCX = PAPER_DIR / "algorithm1.docx"
GENERATED_DIR = BUILD_DIR / "generated"
MERGE_ALGO_DOCX = GENERATED_DIR / "algorithm1_for_merge.docx"
PAPER_MD = PAPER_DIR / "小论文.md"
OUTPUT_DOCX = PAPER_DIR / "小论文.docx"

ALGO_TITLE_MARKER = "Algorithm 1. Event-Triggered Predictive Reinforcement Learning"
NORMAL_TABLE_BORDER_COLOR = "000000"
NORMAL_TABLE_BORDER_SIZE = "4"

CODE_BLOCK_STYLE_KEYWORDS = (
    "source code",
    "code block",
    "preformatted",
    "verbatim",
)


def _is_heading_paragraph(para) -> bool:
    """判断段落是否是 pandoc 生成的 Word 标题。"""
    return para.style.name.startswith("Heading")


def _is_code_block_paragraph(para) -> bool:
    """判断段落是否来自 Markdown fenced code block。

    pandoc 转换为 docx 后通常不会保留字面量 ``` 标记，而是把 fenced code
    block 转换为 ``Source Code`` 等段落样式。
    """
    style_name = (para.style.name or "").strip().lower()
    return any(keyword in style_name for keyword in CODE_BLOCK_STYLE_KEYWORDS)


def _find_pandoc() -> str:
    """确认 pandoc 在 PATH 上。"""
    import shutil

    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError(
            "pandoc not found on PATH. Please install pandoc: "
            "https://pandoc.org/installing.html"
        )
    return pandoc


def _get_or_add_child(parent, tag: str):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        parent.append(child)
    return child


def _set_single_border(border_node, *, color: str, size: str) -> None:
    border_node.set(qn("w:val"), "single")
    border_node.set(qn("w:sz"), size)
    border_node.set(qn("w:space"), "0")
    border_node.set(qn("w:color"), color)


def _set_full_grid_borders(table, *, color: str = NORMAL_TABLE_BORDER_COLOR, size: str = NORMAL_TABLE_BORDER_SIZE) -> None:
    tbl_pr = table._tbl.tblPr
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        table._tbl.insert(0, tbl_pr)

    table_borders = _get_or_add_child(tbl_pr, "w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border_node = _get_or_add_child(table_borders, f"w:{edge}")
        _set_single_border(border_node, color=color, size=size)

    for table_row in table.rows:
        for table_cell in table_row.cells:
            cell_properties = table_cell._tc.get_or_add_tcPr()
            cell_borders = _get_or_add_child(cell_properties, "w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                border_node = _get_or_add_child(cell_borders, f"w:{edge}")
                _set_single_border(border_node, color=color, size=size)


def _set_regular_table_borders(doc: Document) -> int:
    for table in doc.tables:
        _set_full_grid_borders(table)
    return len(doc.tables)


def main() -> int:
    # --- Step 1: 构建算法 docx ---
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Building algorithm docx...")
    subprocess.run(
        [sys.executable, str(ALGO_BUILD_SCRIPT), "--output", str(MERGE_ALGO_DOCX)],
        check=True,
        cwd=str(BUILD_DIR),
    )

    # --- Step 2: pandoc 将 md 转为临时 docx ---
    pandoc = _find_pandoc()
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_docx = Path(tmpdir) / "paper_temp.docx"

        print(f"[2/4] Converting {PAPER_MD.name} to docx via pandoc...")
        subprocess.run(
            [
                pandoc,
                "-f",
                "markdown+tex_math_dollars",
                "-t",
                "docx",
                "-o",
                str(temp_docx),
                str(PAPER_MD),
            ],
            check=True,
            cwd=str(PAPER_DIR),
        )

        # --- Step 3: 普通表格补全框线 ---
        print(f"[3/4] Applying full borders to regular tables...")
        doc = Document(temp_docx)
        bordered_table_count = _set_regular_table_borders(doc)
        print(f"Regular tables updated: {bordered_table_count}")

        # --- Step 4: 注入算法表格 ---
        print(f"[4/4] Injecting algorithm table...")
        algo_doc = Document(MERGE_ALGO_DOCX)

        if not algo_doc.tables:
            print("ERROR: Algorithm docx contains no tables.")
            return 1
        algo_table = algo_doc.tables[0]

        # 定位算法标题段落
        algo_title_para = None
        algo_title_idx = -1
        for i, para in enumerate(doc.paragraphs):
            if ALGO_TITLE_MARKER in para.text:
                algo_title_para = para
                algo_title_idx = i
                break

        if algo_title_para is None:
            print(
                f"ERROR: Could not find algorithm title paragraph "
                f"(marker: '{ALGO_TITLE_MARKER}')."
            )
            return 1

        # 查找算法标题之后的第一个 fenced code block。
        # 注意：pandoc 转为 docx 后不会保留字面量 ``` 标记，而是使用
        # Source Code 等段落样式表示代码块。
        code_block_start_idx = -1
        blocking_heading_idx = -1
        for i in range(algo_title_idx + 1, len(doc.paragraphs)):
            para = doc.paragraphs[i]
            if _is_code_block_paragraph(para):
                code_block_start_idx = i
                break
            if _is_heading_paragraph(para):
                blocking_heading_idx = i
                break

        if code_block_start_idx == -1:
            if blocking_heading_idx != -1:
                heading_text = doc.paragraphs[blocking_heading_idx].text
                print(
                    "ERROR: Reached the next heading before finding the "
                    f"Algorithm 1 fenced code block: '{heading_text}'."
                )
            else:
                print(
                    "ERROR: Could not find a fenced code block after the "
                    "Algorithm 1 title. Put Input, Output, and pseudocode "
                    "inside one ```text block."
                )
            return 1

        # 删除范围截止到第一个连续代码块区域结束。
        algo_end_idx = code_block_start_idx + 1
        while (
            algo_end_idx < len(doc.paragraphs)
            and _is_code_block_paragraph(doc.paragraphs[algo_end_idx])
        ):
            algo_end_idx += 1

        # 算法标题的前一个元素 → 将表格插入到它之后
        prev_element = algo_title_para._element.getprevious()
        if prev_element is None:
            print("ERROR: Algorithm title is the first element in the document body.")
            return 1

        # 收集待删除的段落元素（算法标题 + 标题后第一个代码块结束前的内容）
        elements_to_remove = [
            doc.paragraphs[i]._element
            for i in range(algo_title_idx, algo_end_idx)
        ]

        # 深拷贝算法表格并插入
        tbl_copy = copy.deepcopy(algo_table._tbl)
        prev_element.addnext(tbl_copy)

        # 删除旧的算法文本段落
        for elem in elements_to_remove:
            elem.getparent().remove(elem)

        # 保存最终文档
        doc.save(OUTPUT_DOCX)

    print(f"Done! Created {OUTPUT_DOCX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
