"""템플릿 기반 오프라인 OCR — 5회 이상 처리된 문서 형식은 API 없이 로컬 처리.

OpenCV 격자 감지 → 셀 크롭 → RapidOCR → 엔티티 사전 퍼지 매칭.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .pattern_db import _conn

# ── DB 스키마 ──

TEMPLATE_DDL = """
CREATE TABLE IF NOT EXISTS document_templates (
    template_hash   TEXT PRIMARY KEY,
    doc_type        TEXT NOT NULL,
    grid_rows       INTEGER,
    grid_cols       INTEGER,
    header_json     TEXT NOT NULL,
    column_types    TEXT NOT NULL,
    cell_regions    TEXT,
    confidence      REAL DEFAULT 0.0,
    sample_count    INTEGER DEFAULT 1
);
"""

_initialized = False

def _ensure_table():
    global _initialized
    if _initialized:
        return
    with _conn() as con:
        con.executescript(TEMPLATE_DDL)
    _initialized = True


# ── 템플릿 해시 계산 ──

def _compute_template_hash(grid_rows: int, grid_cols: int, cell_ratios: list) -> str:
    """격자 구조의 해상도 무관 해시."""
    data = f"{grid_rows}x{grid_cols}|" + ",".join(f"{r:.2f}" for r in cell_ratios)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _cells_to_ratios(cells: list[dict], img_w: int, img_h: int) -> list[float]:
    """셀 좌표를 이미지 크기 대비 비율로 정규화."""
    ratios = []
    for cell in cells:
        ratios.append(cell["x"] / max(img_w, 1))
        ratios.append(cell["y"] / max(img_h, 1))
        ratios.append(cell["w"] / max(img_w, 1))
        ratios.append(cell["h"] / max(img_h, 1))
    return ratios


# ── 템플릿 조회 ──

def can_use_template(image_path: str) -> tuple[bool, dict | None]:
    """이미지의 격자 구조가 학습된 템플릿과 매칭되는지 확인."""
    _ensure_table()
    try:
        from .table_detector import detect_table_cells, get_grid_dimensions
        from .ocr_extractor import _imread_unicode

        img = _imread_unicode(image_path)
        if img is None:
            return False, None

        cells = detect_table_cells(img)
        if not cells or len(cells) < 4:
            return False, None

        grid_rows, grid_cols = get_grid_dimensions(cells)
        if grid_rows < 2 or grid_cols < 2:
            return False, None

        ratios = _cells_to_ratios(cells[:grid_cols], img.shape[1], img.shape[0])
        tmpl_hash = _compute_template_hash(grid_rows, grid_cols, ratios)

        with _conn() as con:
            row = con.execute(
                "SELECT * FROM document_templates WHERE template_hash=? AND sample_count>=5",
                (tmpl_hash,),
            ).fetchone()

        if row:
            return True, {
                "template_hash": tmpl_hash,
                "doc_type": row["doc_type"],
                "grid_rows": row["grid_rows"],
                "grid_cols": row["grid_cols"],
                "headers": json.loads(row["header_json"]),
                "column_types": json.loads(row["column_types"]),
                "cells": cells,
                "img": img,
            }
    except Exception:
        pass
    return False, None


# ── 템플릿 기반 OCR ──

def extract_with_template(
    image_path: str,
    template: dict,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[list[pd.DataFrame], dict] | None:
    """템플릿 + RapidOCR + 엔티티 사전으로 API 없이 표를 추출한다."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        from .entity_dict import correct_text_column
    except ImportError:
        return None

    ocr = RapidOCR()
    img = template["img"]
    cells = template["cells"]
    headers = template["headers"]
    col_types = template["column_types"]
    grid_rows = template["grid_rows"]
    grid_cols = template["grid_cols"]

    if progress_cb:
        progress_cb(10)

    # 셀 크롭 → OCR
    rows_data = []
    for r in range(1, grid_rows):  # 0번 행은 헤더
        row_cells = [c for c in cells if c["row"] == r]
        row_cells.sort(key=lambda c: c["col"])
        row_values = []
        for cell in row_cells:
            x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
            # 여백 제거
            pad = 2
            crop = img[max(0, y+pad):min(img.shape[0], y+h-pad),
                       max(0, x+pad):min(img.shape[1], x+w-pad)]
            if crop.size == 0:
                row_values.append("")
                continue

            result, _ = ocr(crop)
            if result:
                text = " ".join(r[1] for r in result).strip()
                row_values.append(text)
            else:
                row_values.append("")

        # 열 수 맞추기
        while len(row_values) < len(headers):
            row_values.append("")
        rows_data.append(row_values[:len(headers)])

        if progress_cb:
            progress_cb(10 + int(80 * r / max(grid_rows - 1, 1)))

    df = pd.DataFrame(rows_data, columns=headers)

    # 엔티티 사전 교정
    for ci, (col, ctype) in enumerate(zip(df.columns, col_types)):
        if ctype == "company":
            df[col] = correct_text_column(df[col], "company")
        elif ctype == "department":
            df[col] = correct_text_column(df[col], "department")
        elif ctype == "account":
            df[col] = correct_text_column(df[col], "account")

    if progress_cb:
        progress_cb(100)

    return ([df], {})


# ── 템플릿 학습 ──

def learn_template(
    image_path: str,
    tables: list[pd.DataFrame],
    doc_type: str,
) -> None:
    """성공적인 API 결과로 템플릿을 학습/업데이트한다."""
    _ensure_table()
    try:
        from .table_detector import detect_table_cells, get_grid_dimensions
        from .ocr_extractor import _imread_unicode

        img = _imread_unicode(image_path)
        if img is None:
            return

        cells = detect_table_cells(img)
        if not cells or len(cells) < 4:
            return

        grid_rows, grid_cols = get_grid_dimensions(cells)
        if grid_rows < 2 or grid_cols < 2:
            return

        ratios = _cells_to_ratios(cells[:grid_cols], img.shape[1], img.shape[0])
        tmpl_hash = _compute_template_hash(grid_rows, grid_cols, ratios)

        df = tables[0] if tables else pd.DataFrame()
        headers = list(df.columns)
        col_types = _classify_columns(df, headers)

        cell_regions = json.dumps([
            {"row": c["row"], "col": c["col"], "x": c["x"], "y": c["y"],
             "w": c["w"], "h": c["h"]}
            for c in cells
        ])

        with _conn() as con:
            con.execute(
                """INSERT INTO document_templates
                   (template_hash, doc_type, grid_rows, grid_cols,
                    header_json, column_types, cell_regions, sample_count)
                   VALUES (?,?,?,?,?,?,?,1)
                   ON CONFLICT(template_hash)
                   DO UPDATE SET sample_count=sample_count+1""",
                (tmpl_hash, doc_type, grid_rows, grid_cols,
                 json.dumps(headers, ensure_ascii=False),
                 json.dumps(col_types, ensure_ascii=False),
                 cell_regions),
            )
    except Exception:
        pass


def _classify_columns(df: pd.DataFrame, headers: list[str]) -> list[str]:
    """열 유형을 분류한다."""
    types = []
    for i, h in enumerate(headers):
        h_lower = str(h).strip()
        if "회사" in h_lower or "기관" in h_lower:
            types.append("company")
        elif "부서" in h_lower or "팀" in h_lower:
            types.append("department")
        elif "이름" in h_lower or "성명" in h_lower:
            types.append("name")
        elif "계정" in h_lower:
            types.append("account")
        elif "일시" in h_lower or "날짜" in h_lower:
            types.append("datetime")
        elif h_lower.lstrip("-+").isdigit() or "금액" in h_lower or "합계" in h_lower:
            types.append("number")
        else:
            # 데이터 기반 추론
            if i < len(df.columns) and len(df) > 0:
                sample = df.iloc[:5, i].astype(str)
                nums = sample.str.replace(",", "").str.strip()
                if nums.str.match(r"^-?\d+\.?\d*$").sum() >= 3:
                    types.append("number")
                else:
                    types.append("text")
            else:
                types.append("text")
    return types
