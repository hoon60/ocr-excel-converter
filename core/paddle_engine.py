"""PaddleOCR 한국어 엔진 — 로컬 텍스트 인식 + 좌표 기반 표 구조 분석.

EasyOCR보다 한국어 인식률이 높다 (92-96% vs 80-90%).
CPU 전용, PyInstaller 호환.
"""

import os
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .runtime_flags import get_runtime_flags


# ── PaddleOCR 싱글턴 ──

_ocr = None


def _runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def _bundled_paddle_model_kwargs() -> dict:
    for base in (
        _runtime_base_dir() / ".paddleocr" / "whl",
        Path.home() / ".paddleocr" / "whl",
    ):
        det_dir = base / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer"
        rec_dir = base / "rec" / "korean" / "korean_PP-OCRv4_rec_infer"
        cls_dir = base / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"
        if det_dir.is_dir() and rec_dir.is_dir() and cls_dir.is_dir():
            return {
                "lang": "korean",
                "use_angle_cls": True,
                "show_log": False,
                "det_model_dir": str(det_dir),
                "rec_model_dir": str(rec_dir),
                "cls_model_dir": str(cls_dir),
            }

    for base in (
        _runtime_base_dir() / ".paddlex" / "official_models",
        Path.home() / ".paddlex" / "official_models",
    ):
        det_dir = base / "PP-OCRv5_server_det"
        rec_dir = base / "korean_PP-OCRv5_mobile_rec"
        if det_dir.is_dir() and rec_dir.is_dir():
            return {
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_detection_model_dir": str(det_dir),
                "text_recognition_model_name": "korean_PP-OCRv5_mobile_rec",
                "text_recognition_model_dir": str(rec_dir),
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
    return {}


def _get_ocr():
    global _ocr
    if _ocr is None:
        # Paddle import changes DLL resolution on Windows, so preload torch first.
        import torch  # noqa: F401
        from paddleocr import PaddleOCR
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        model_kwargs = _bundled_paddle_model_kwargs()
        candidates = []
        if model_kwargs:
            candidates.extend(
                [
                    model_kwargs,
                    {**model_kwargs, "lang": "korean"},
                ]
            )
        candidates.extend(
            [
                {"lang": "korean"},
                {"lang": "korean", "use_gpu": False},
                {"lang": "korean", "use_gpu": False, "show_log": False},
            ]
        )

        last_error = None
        for kwargs in candidates:
            try:
                _ocr = PaddleOCR(**kwargs)
                break
            except Exception as exc:
                last_error = exc
        if _ocr is None:
            raise last_error or RuntimeError("PaddleOCR 초기화 실패")
    return _ocr


def _imread_unicode(path: str):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── 메인 추출 함수 ──

def extract_with_paddle(
    image_path: str,
    progress_cb: Callable[[int], None] | None = None,
    benchmark_mode: bool | None = None,
) -> tuple[list[pd.DataFrame], dict, float] | None:
    """PaddleOCR로 이미지에서 표를 추출한다.

    Returns:
        (DataFrames, metadata, confidence) 또는 None
    """
    flags = get_runtime_flags(benchmark_mode)

    img = _imread_unicode(image_path)
    if img is None:
        return None

    if progress_cb:
        progress_cb(5)

    # PaddleOCR 실행
    ocr = _get_ocr()
    try:
        raw = ocr.ocr(image_path, cls=True)
    except TypeError:
        raw = ocr.ocr(image_path)
    if not raw or not raw[0]:
        return None

    if progress_cb:
        progress_cb(40)

    # 박스 표준화 (v4 포맷: [[coords, (text, conf)], ...])
    boxes = []
    for item in raw[0]:
        coords, (text, conf) = item
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

    if not boxes:
        return None

    if progress_cb:
        progress_cb(50)

    # ── 격자 + 박스 매핑 (가장 정확) → 폴백: 좌표 기반 ──
    df, metadata = None, {}
    try:
        from .table_detector import detect_table_cells, get_grid_dimensions
        cells = detect_table_cells(img)
        if cells:
            grid_rows, grid_cols = get_grid_dimensions(cells)
            if grid_rows >= 3 and grid_cols >= 7:
                df, metadata = _grid_based_extraction(boxes, cells, grid_rows, grid_cols, img)
                if df is not None and not df.empty and df.shape[1] >= 7:
                    # 헤더 복원
                    df = _restore_spread_headers(df, cells, img, boxes)
                    print(f"[PaddleOCR] 격자+박스 매핑 ({grid_rows}x{grid_cols}, {df.shape[1]}열)")
                else:
                    df, metadata = None, {}
    except Exception as e:
        print(f"[PaddleOCR] 격자 매핑 실패: {e}")
        df, metadata = None, {}

    # 폴백: 좌표 기반 추출
    if df is None or df.empty or df.shape[1] < 5:
        df, metadata = _coordinate_based_extraction(boxes, img)

    if df is None or df.empty:
        return None

    if progress_cb:
        progress_cb(70)

    # 숫자 열 OCR 문자 교정 (O→0, l→1 등)
    df = _fix_numeric_ocr(df)

    # 엔티티 사전 교정
    if not flags.disable_entity_corrections and not flags.disable_db_corrections:
        df = _apply_entity_corrections(df)

    if progress_cb:
        progress_cb(85)

    # 신뢰도 계산
    confidence = _compute_confidence(boxes, df)

    if progress_cb:
        progress_cb(100)

    return ([df], metadata, confidence)


# ── 격자 기반 추출 (OpenCV 셀 + PaddleOCR 텍스트 매핑) ──

def _grid_based_extraction(
    boxes: list[dict],
    cells: list[dict],
    grid_rows: int,
    grid_cols: int,
    img,
) -> tuple[pd.DataFrame | None, dict]:
    """OpenCV 격자 셀에 PaddleOCR 텍스트를 매핑한다."""

    # 메타데이터 영역 (격자 위)
    grid_y_min = min(c["y_min"] for c in cells)
    metadata = _extract_metadata(boxes, grid_y_min)

    # 셀별 텍스트 매핑: 각 격자 셀 안에 중심점이 들어가는 박스를 찾음
    cell_texts = {}  # (row, col) → [texts]
    for box in boxes:
        bx, by = box["x_center"], box["y_center"]
        best_cell = None
        best_area = float("inf")
        for cell in cells:
            cx_min = cell["x_min"]
            cy_min = cell["y_min"]
            cx_max = cell["x_max"]
            cy_max = cell["y_max"]
            if cx_min <= bx <= cx_max and cy_min <= by <= cy_max:
                area = (cx_max - cx_min) * (cy_max - cy_min)
                if area < best_area:
                    best_area = area
                    best_cell = (cell["row"], cell["col"])
        if best_cell:
            cell_texts.setdefault(best_cell, []).append(box["text"])

    # 같은 셀 안에 여러 텍스트가 있으면 결합
    merged_cells = {}
    for key, texts in cell_texts.items():
        merged_cells[key] = " ".join(texts)

    # 헤더 행 결정: 격자가 메타데이터까지 포함할 수 있으므로 전체 검색
    header_row = None
    for r in range(grid_rows):
        row_text = " ".join(merged_cells.get((r, c), "") for c in range(grid_cols))
        if any(kw in row_text for kw in ("회사명", "부서명", "이름", "계정")):
            header_row = r
            break
    if header_row is None:
        return None, metadata  # 헤더를 못 찾으면 격자 추출 포기

    # 헤더 구성
    headers = []
    for c in range(grid_cols):
        h = merged_cells.get((header_row, c), f"Col_{c}")
        headers.append(h.strip() if h else f"Col_{c}")

    # 데이터 행 구성
    data_rows = []
    for r in range(header_row + 1, grid_rows):
        row = [merged_cells.get((r, c), "") for c in range(grid_cols)]
        data_rows.append(row)

    if not data_rows:
        return None, metadata

    df = pd.DataFrame(data_rows, columns=headers)
    return df, metadata


# ── 좌표 기반 행/열 그룹핑 ──

def _coordinate_based_extraction(
    boxes: list[dict], img
) -> tuple[pd.DataFrame | None, dict]:
    if not boxes:
        return None, {}

    # 테이블 헤더 위치 찾기
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
        header_y = img.shape[0] * 0.15

    # 메타데이터 추출
    metadata = _extract_metadata(boxes, header_y)

    # 테이블 박스만 필터
    table_boxes = [b for b in boxes if b["y_min"] >= header_y - 10]
    if not table_boxes:
        return None, metadata

    # y 좌표로 행 그룹핑 (이미지 높이의 2% 이내면 같은 행)
    row_threshold = max(15, img.shape[0] * 0.02)
    table_boxes.sort(key=lambda b: b["y_center"])
    rows = []
    current_row = [table_boxes[0]]
    for box in table_boxes[1:]:
        if abs(box["y_center"] - current_row[-1]["y_center"]) < row_threshold:
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
        while len(values) < len(headers):
            values.append("")
        data_rows.append(values[:len(headers)])

    df = pd.DataFrame(data_rows, columns=headers)
    return df, metadata


def _extract_metadata(boxes: list[dict], header_y: float) -> dict:
    meta_boxes = [b for b in boxes if b["y_max"] < header_y]
    metadata = {"title": "", "info": [], "market_info": []}
    for box in meta_boxes:
        text = box["text"]
        if any(kw in text for kw in ("수요예측", "BBB", "AA", "결과")):
            metadata["title"] = text
        elif any(kw in text for kw in ("사채만기", "모집금액", "모집밴드", "모집번드")):
            metadata["info"].append(text)
        elif any(kw in text for kw in ("개별민평", "국고", "Spread")):
            metadata["market_info"].append(text)
    return metadata


def _find_header_row_in_cells(
    cell_texts: dict[tuple[int, int], str],
    grid_rows: int,
    grid_cols: int,
) -> int:
    """셀 텍스트에서 '회사명' 등 헤더 키워드가 있는 행을 찾는다."""
    header_keywords = {"회사명", "부서명", "이름", "계정", "회사", "기관명"}
    for r in range(min(5, grid_rows)):
        row_text = " ".join(cell_texts.get((r, c), "") for c in range(grid_cols))
        if any(kw in row_text for kw in header_keywords):
            return r
    return 0


# ── 스프레드 헤더 복원 ──

def _restore_spread_headers(
    df: pd.DataFrame,
    cells: list[dict],
    img: np.ndarray,
    boxes: list[dict],
) -> pd.DataFrame:
    """Col_N placeholder 헤더를 실제 스프레드 값으로 복원한다.

    전략:
    1) 헤더 셀을 crop + 반전 → PaddleOCR 재인식
    2) 실패 시: "기준일 스프레드" 행의 값을 헤더로 사용
    3) 최종 폴백: 인식된 인접 헤더 사이를 등차수열로 보간
    """
    import re

    placeholder_cols = []
    for i, col in enumerate(df.columns):
        if re.match(r"^Col_\d+$", str(col).strip()):
            placeholder_cols.append(i)

    if not placeholder_cols:
        return df

    # ── 전략 1: 헤더 셀 crop + 반전 OCR ──
    from .table_detector import get_grid_dimensions
    grid_rows, grid_cols = get_grid_dimensions(cells)

    # 헤더 행의 셀 목록
    header_cells = [c for c in cells if c["row"] == 0]
    header_cells.sort(key=lambda c: c["col"])

    new_headers = list(df.columns)
    ocr = None

    for pi in placeholder_cols:
        # 격자 열 인덱스 매핑 (df 열 인덱스 → 격자 열)
        # df 열과 격자 열이 1:1 매핑이라고 가정
        if pi >= len(header_cells):
            continue

        hc = header_cells[pi]
        x1, y1, x2, y2 = hc["x_min"], hc["y_min"], hc["x_max"], hc["y_max"]

        # 여백 추가
        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 어두운 배경 감지 → 반전
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        mean_val = np.mean(gray_crop)
        if mean_val < 128:
            gray_crop = cv2.bitwise_not(gray_crop)
            crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

        # PaddleOCR로 재인식
        try:
            if ocr is None:
                ocr = _get_ocr()
            result = ocr.ocr(crop, cls=False)
            if result and result[0]:
                text = " ".join(item[1][0] for item in result[0]).strip()
                # 숫자 + 부호만 추출
                match = re.search(r"[+-]?\d+\.?\d*", text)
                if match:
                    new_headers[pi] = match.group(0)
                    continue
        except Exception:
            pass

    # ── 전략 2: "기준일 스프레드" 행의 값을 헤더로 사용 ──
    still_placeholder = [i for i in placeholder_cols if re.match(r"^Col_\d+$", str(new_headers[i]))]
    if still_placeholder:
        for ri in range(len(df)):
            first_col = str(df.iloc[ri, 0]).strip() if df.shape[1] > 0 else ""
            # "기준일 스프레드" 또는 "기준일 SP" 행
            is_spread_ref = False
            for ci in range(min(6, df.shape[1])):
                val = str(df.iloc[ri, ci]).strip()
                if "기준일" in val and ("스프레드" in val or "SP" in val):
                    is_spread_ref = True
                    break
            if is_spread_ref:
                for pi in still_placeholder:
                    v = str(df.iloc[ri, pi]).strip().replace(",", "")
                    try:
                        fv = float(v)
                        new_headers[pi] = v
                    except (ValueError, TypeError):
                        pass
                break

    # ── 전략 3: 등차수열 보간 ──
    still_placeholder = [i for i in placeholder_cols if re.match(r"^Col_\d+$", str(new_headers[i]))]
    if still_placeholder:
        # 인식된 숫자 헤더 수집
        known = {}  # col_idx → numeric value
        for i, h in enumerate(new_headers):
            s = str(h).strip().lstrip("+-")
            if s.replace(".", "").isdigit() and i >= 5:
                try:
                    known[i] = float(str(h).strip())
                except (ValueError, TypeError):
                    pass

        if len(known) >= 2:
            sorted_known = sorted(known.items())
            # 인접한 두 known 헤더 사이를 선형 보간
            for j in range(len(sorted_known) - 1):
                idx1, val1 = sorted_known[j]
                idx2, val2 = sorted_known[j + 1]
                gap = idx2 - idx1
                if gap <= 1:
                    continue
                step = (val2 - val1) / gap
                for k in range(1, gap):
                    target_idx = idx1 + k
                    if target_idx in still_placeholder:
                        interp_val = val1 + step * k
                        if interp_val == int(interp_val):
                            new_headers[target_idx] = str(int(interp_val))
                        else:
                            new_headers[target_idx] = f"{interp_val:.1f}"

    df.columns = new_headers
    restored = sum(1 for i in placeholder_cols if not re.match(r"^Col_\d+$", str(new_headers[i])))
    if restored > 0:
        print(f"[PaddleOCR] {restored}/{len(placeholder_cols)}개 스프레드 헤더 복원")

    return df


# ── 숫자 열 OCR 교정 ──

def _fix_numeric_ocr(df: pd.DataFrame) -> pd.DataFrame:
    """숫자 열의 OCR 문자 혼동을 교정한다 (O→0, l→1 등)."""
    _CHAR_MAP = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}

    for col in df.columns:
        col_name = str(col).strip()
        is_numeric = col_name.lstrip("+-").isdigit() or "금액" in col_name or "합계" in col_name
        if not is_numeric:
            continue
        for idx, val in df[col].items():
            s = str(val).strip()
            if not s:
                continue
            fixed = s
            for wrong, correct in _CHAR_MAP.items():
                fixed = fixed.replace(wrong, correct)
            if fixed != s:
                df.at[idx, col] = fixed
    return df


# ── 엔티티 사전 교정 ──

def _apply_entity_corrections(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from .entity_dict import correct_text_column, _ensure_tables, initialize_seed_entities
        _ensure_tables()
        initialize_seed_entities()
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


# ── 신뢰도 계산 ──

def _compute_confidence(boxes: list[dict], df: pd.DataFrame) -> float:
    scores = []

    if boxes:
        avg_conf = sum(b["confidence"] for b in boxes) / len(boxes)
        scores.append(avg_conf)

    if df.size > 0:
        non_empty = (df.astype(str).replace("", np.nan).notna().sum().sum())
        fill_rate = non_empty / df.size
        scores.append(min(fill_rate * 1.5, 1.0))

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

    # 열 수 기반 구조 신뢰도
    n_cols = df.shape[1]
    if n_cols < 5:
        scores.append(0.1)   # 열이 너무 적음 → 구조 파악 실패
    elif n_cols < 8:
        scores.append(0.4)   # 수요예측표는 최소 8열 이상
    elif n_cols > 15:
        scores.append(0.5)   # 넓은 표: 좌표 그룹핑 불안정

    return sum(scores) / len(scores) if scores else 0.0
