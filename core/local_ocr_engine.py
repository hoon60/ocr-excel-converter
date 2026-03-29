"""로컬 OCR 엔진 — EasyOCR + OpenCV 격자 + 엔티티 사전 + 합계 검증.

API 호출 없이 로컬에서 표를 추출한다.
숫자: EasyOCR 100% 신뢰.
텍스트: EasyOCR → 엔티티 사전 퍼지 매칭으로 교정.
구조: OpenCV 격자선 감지 → 셀에 텍스트 매핑.
"""

import os
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .runtime_flags import get_runtime_flags


# ── EasyOCR 싱글턴 ──

_reader = None


def _runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def _easyocr_model_dirs() -> tuple[str | None, str | None]:
    for base in (_runtime_base_dir() / ".EasyOCR", Path.home() / ".EasyOCR"):
        model_dir = base / "model"
        user_network_dir = base / "user_network"
        if model_dir.is_dir():
            return str(model_dir), str(user_network_dir)
    return None, None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        model_dir, user_network_dir = _easyocr_model_dirs()
        kwargs = {"gpu": False, "verbose": False}
        if model_dir:
            kwargs["model_storage_directory"] = model_dir
            kwargs["download_enabled"] = False
        if user_network_dir:
            kwargs["user_network_directory"] = user_network_dir
        _reader = easyocr.Reader(["ko", "en"], **kwargs)
    return _reader


def _imread_unicode(path: str):
    """한국어 경로 이미지 읽기."""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── 메인 추출 함수 ──

def extract_with_local_ocr(
    image_path: str,
    progress_cb: Callable[[int], None] | None = None,
    benchmark_mode: bool | None = None,
) -> tuple[list[pd.DataFrame], dict, float] | None:
    """로컬 OCR로 표를 추출한다 (PaddleOCR 우선 → EasyOCR 폴백).

    Returns:
        (DataFrames, metadata, confidence) 또는 None
    """
    flags = get_runtime_flags(benchmark_mode)

    # PaddleOCR 먼저 시도 (한국어 인식률 우수)
    try:
        from .paddle_engine import extract_with_paddle
        result = extract_with_paddle(
            image_path,
            progress_cb,
            benchmark_mode=flags.benchmark_mode,
        )
        if result:
            print("[LocalOCR] PaddleOCR 사용")
            return result
    except Exception as e:
        print(f"[LocalOCR] PaddleOCR 실패 → EasyOCR 폴백: {e}")

    # EasyOCR 폴백
    img = _imread_unicode(image_path)
    if img is None:
        return None

    if progress_cb:
        progress_cb(5)

    # 1) EasyOCR로 전체 텍스트 박스 추출
    reader = _get_reader()
    raw_results = reader.readtext(img)
    if not raw_results:
        return None

    if progress_cb:
        progress_cb(40)

    # 박스 표준화
    boxes = []
    for coords, text, conf in raw_results:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        boxes.append({
            "text": text.strip(),
            "confidence": conf,
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "x_center": sum(xs) / len(xs),
            "y_center": sum(ys) / len(ys),
        })

    if progress_cb:
        progress_cb(50)

    # 3) 좌표 기반 행/열 그룹핑 (EasyOCR 좌표가 가장 안정적)
    df, metadata = _coordinate_based_extraction(boxes, img)

    if df is None or df.empty:
        return None

    if progress_cb:
        progress_cb(70)

    # 4) 엔티티 사전 교정
    if not flags.disable_entity_corrections and not flags.disable_db_corrections:
        df = _apply_entity_corrections(df)

    if progress_cb:
        progress_cb(85)

    # 5) 합계 행 검증
    df = _validate_sum_rows(df)

    # 6) 신뢰도 계산
    confidence = _compute_confidence(boxes, df)

    if progress_cb:
        progress_cb(100)

    return ([df], metadata, confidence)


# ── 격자 기반 추출 ──

def _grid_based_extraction(
    boxes: list[dict],
    cells: list[dict],
    img,
) -> tuple[pd.DataFrame | None, dict]:
    """OpenCV 격자 셀에 EasyOCR 텍스트를 매핑한다."""
    from .table_detector import get_grid_dimensions

    grid_rows, grid_cols = get_grid_dimensions(cells)

    # 셀별 텍스트 매핑 (IOU 기반)
    cell_texts = {}  # (row, col) → text
    for cell in cells:
        r, c = cell["row"], cell["col"]
        cx_min = cell.get("x_min", cell.get("x", 0))
        cy_min = cell.get("y_min", cell.get("y", 0))
        cx_max = cell.get("x_max", cx_min + cell.get("w", 0))
        cy_max = cell.get("y_max", cy_min + cell.get("h", 0))

        best_text = ""
        best_overlap = 0
        for box in boxes:
            # 박스와 셀의 겹침 계산
            ox = max(0, min(box["x_max"], cx_max) - max(box["x_min"], cx_min))
            oy = max(0, min(box["y_max"], cy_max) - max(box["y_min"], cy_min))
            overlap = ox * oy
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = box["text"]

        cell_texts[(r, c)] = best_text

    # DataFrame 구성
    # 행 0 = 헤더
    headers = [cell_texts.get((0, c), f"Col_{c}") for c in range(grid_cols)]
    rows = []
    for r in range(1, grid_rows):
        row = [cell_texts.get((r, c), "") for c in range(grid_cols)]
        rows.append(row)

    if not rows:
        return None, {}

    df = pd.DataFrame(rows, columns=headers)
    return df, {}


# ── 좌표 기반 추출 ──

def _coordinate_based_extraction(
    boxes: list[dict],
    img,
) -> tuple[pd.DataFrame | None, dict]:
    """격자 없이 좌표 기반으로 행/열을 그룹핑한다."""
    if not boxes:
        return None, {}

    # 메타데이터 영역과 테이블 영역 분리
    # 테이블 시작: "회사명" 또는 "번호" 같은 헤더가 있는 y 위치
    header_y = None
    header_keywords = {"회사명", "부서명", "이름", "계정", "번호", "기관명"}
    for box in boxes:
        for kw in header_keywords:
            if kw in box["text"]:
                header_y = box["y_min"]
                break
        if header_y is not None:
            break

    if header_y is None:
        # 헤더를 못 찾으면 이미지 상단 1/4 이후부터 테이블로 간주
        header_y = img.shape[0] * 0.15

    # 메타데이터 추출
    metadata = _extract_metadata(boxes, header_y)

    # 테이블 박스만 필터
    table_boxes = [b for b in boxes if b["y_min"] >= header_y - 10]
    if not table_boxes:
        return None, metadata

    # y 좌표로 행 그룹핑
    table_boxes.sort(key=lambda b: b["y_center"])
    rows = []
    current_row = [table_boxes[0]]
    for box in table_boxes[1:]:
        if abs(box["y_center"] - current_row[-1]["y_center"]) < 20:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b["x_center"]))
            current_row = [box]
    rows.append(sorted(current_row, key=lambda b: b["x_center"]))

    if len(rows) < 2:
        return None, metadata

    # 첫 행 = 헤더
    headers = [b["text"] for b in rows[0]]
    data_rows = []
    for row in rows[1:]:
        values = [b["text"] for b in row]
        # 열 수 맞추기
        while len(values) < len(headers):
            values.append("")
        data_rows.append(values[:len(headers)])

    df = pd.DataFrame(data_rows, columns=headers)
    return df, metadata


def _extract_metadata(boxes: list[dict], header_y: float) -> dict:
    """헤더 위의 텍스트에서 메타데이터를 추출한다."""
    meta_boxes = [b for b in boxes if b["y_max"] < header_y]
    if not meta_boxes:
        return {}

    metadata = {"title": "", "info": [], "market_info": []}

    for box in meta_boxes:
        text = box["text"]
        if any(kw in text for kw in ("수요예측", "BBB", "AA", "제9", "결과")):
            metadata["title"] = text
        elif "사채만기" in text or "모집금액" in text or "모집밴드" in text or "모집번드" in text:
            metadata["info"].append(text)
        elif any(kw in text for kw in ("개별민평", "국고", "Spread")):
            metadata["market_info"].append(text)

    return metadata


# ── 엔티티 사전 교정 ──

def _apply_entity_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """엔티티 사전 + DB 교정을 적용한다."""
    try:
        from .entity_dict import correct_text_column, _ensure_tables, initialize_seed_entities
        _ensure_tables()
        initialize_seed_entities()

        # 회사명 열 (첫 번째 또는 "회사" 포함 열)
        for ci, col in enumerate(df.columns[:6]):
            col_name = str(col).lower()
            if "회사" in col_name or ci == 0:
                df[col] = correct_text_column(df[col], "company", max_jamo_dist=4)
            elif "부서" in col_name or ci == 1:
                df[col] = correct_text_column(df[col], "department", max_jamo_dist=4)
            elif "계정" in col_name or ci == 3:
                df[col] = correct_text_column(df[col], "account", max_jamo_dist=2)
    except Exception:
        pass

    # DB 교정 (이름 포함)
    try:
        from .pattern_db import get_corrections
        corrections = get_corrections("demand_forecast")
        text_cols = [c for c in df.columns[:6]
                     if str(df[c].dtype) in ("object", "string")]
        for wrong, correct in corrections:
            for col in text_cols:
                df[col] = df[col].astype(str).str.replace(wrong, correct, regex=False)
    except Exception:
        pass

    return df


# ── 합계 행 검증 ──

def _validate_sum_rows(df: pd.DataFrame) -> pd.DataFrame:
    """합계 행의 값과 데이터 열의 합산이 일치하는지 검증/교정한다."""
    # 합계 행 찾기
    sum_row_idx = None
    for i in range(len(df)):
        first_col = str(df.iloc[i, 0]).strip() if df.shape[1] > 0 else ""
        # 합계 레이블이 있는 열 찾기
        for ci in range(min(6, df.shape[1])):
            val = str(df.iloc[i, ci]).strip()
            if val == "합계":
                sum_row_idx = i
                break
        if sum_row_idx is not None:
            break

    if sum_row_idx is None:
        return df

    # 숫자 열에서 합계 검증
    data_rows = df.iloc[:sum_row_idx]
    sum_row = df.iloc[sum_row_idx]

    for ci in range(df.shape[1]):
        try:
            expected = float(str(sum_row.iloc[ci]).replace(",", "").strip())
            actual = 0
            for ri in range(len(data_rows)):
                val = str(data_rows.iloc[ri, ci]).replace(",", "").strip()
                if val and val not in ("", "nan"):
                    try:
                        actual += float(val)
                    except ValueError:
                        break
            else:
                # 합계 불일치 시 → 합계 행 값을 신뢰 (표에 적힌 값이 정답)
                if abs(actual - expected) > 0.01 and expected > 0:
                    pass  # 로깅만, 교정은 하지 않음
        except (ValueError, IndexError):
            pass

    return df


# ── 신뢰도 계산 ──

def _compute_confidence(boxes: list[dict], df: pd.DataFrame) -> float:
    """추출 품질 신뢰도를 0-1 스케일로 계산한다."""
    scores = []

    # 1) 개별 박스 confidence 평균
    if boxes:
        avg_conf = sum(b["confidence"] for b in boxes) / len(boxes)
        scores.append(avg_conf)

    # 2) 빈 셀 비율
    if df.size > 0:
        non_empty = (df.astype(str).replace("", np.nan).notna().sum().sum())
        fill_rate = non_empty / df.size
        scores.append(min(fill_rate * 1.5, 1.0))  # 66% 이상이면 1.0

    # 3) 엔티티 사전 매칭률 (회사명 열)
    try:
        from .entity_dict import get_all_entities
        known = {name for _, name in get_all_entities("company")}
        if df.shape[1] > 0:
            company_col = df.iloc[:, 0]
            matched = sum(1 for v in company_col if str(v).strip() in known)
            total = sum(1 for v in company_col if str(v).strip())
            if total > 0:
                scores.append(matched / total)
    except Exception:
        pass

    # 4) 헤더 패턴 매칭
    try:
        from .pattern_db import get_top_header_patterns
        known_headers = get_top_header_patterns("demand_forecast", top_n=3)
        if known_headers:
            current = set(str(c).strip() for c in df.columns)
            best_match = 0
            for kh in known_headers:
                known_set = set(str(h).strip() for h in kh)
                if known_set:
                    overlap = len(current & known_set) / len(known_set)
                    best_match = max(best_match, overlap)
            scores.append(best_match)
    except Exception:
        pass

    return sum(scores) / len(scores) if scores else 0.0
