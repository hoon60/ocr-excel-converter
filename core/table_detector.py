"""OpenCV 기반 테이블 격자선 감지 + OCR 텍스트 → 셀 매핑.

이미지에서 수평/수직선을 감지하여 셀 경계를 추출한다.
격자선이 있는 표라면 OCR 좌표 기반 클러스터링보다 훨씬 정확한 열/행 분리가 가능하다.
"""

import cv2
import numpy as np
import pandas as pd


def detect_table_cells(
    img: np.ndarray,
    min_line_ratio: float = 0.05,
) -> list[dict] | None:
    """이미지에서 테이블 격자선을 감지하고 셀 좌표를 반환한다.

    Args:
        img: BGR 이미지
        min_line_ratio: 최소 선 길이 비율 (이미지 폭/높이 대비)

    Returns:
        셀 리스트 [{"row": r, "col": c, "x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}]
        격자선을 찾지 못하면 None
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 2단계 이진화: 고정 임계값 → 실패 시 적응형
    h_positions, v_positions = None, None
    for binary in _binarize_variants(gray):
        hp, vp = _detect_lines(binary, w, h, min_line_ratio)
        if hp is not None and vp is not None:
            h_positions, v_positions = hp, vp
            break

    if h_positions is None or v_positions is None:
        return None

    # 셀 좌표 생성
    cells = []
    for r in range(len(h_positions) - 1):
        for c in range(len(v_positions) - 1):
            cells.append({
                "row": r,
                "col": c,
                "x_min": v_positions[c],
                "y_min": h_positions[r],
                "x_max": v_positions[c + 1],
                "y_max": h_positions[r + 1],
            })

    num_rows = len(h_positions) - 1
    num_cols = len(v_positions) - 1

    if num_rows < 2 or num_cols < 2:
        return None

    return cells


def _binarize_variants(gray: np.ndarray):
    """여러 이진화 전략을 순차 생성한다."""
    # 1차: 고정 임계값 (깨끗한 이미지)
    _, b1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    yield b1
    # 2차: 적응형 임계값 (얇은 선, 스크린샷)
    b2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 5)
    yield b2
    # 3차: 낮은 고정 임계값
    _, b3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    yield b3


def _detect_lines(
    binary: np.ndarray, w: int, h: int, min_line_ratio: float
) -> tuple[list[int] | None, list[int] | None]:
    """이진화된 이미지에서 수평/수직선을 감지한다."""
    # 수평선 감지
    h_len = max(int(w * min_line_ratio), 20)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 수직선 감지 (더 짧은 최소 길이 허용)
    v_len = max(int(h * 0.02), 10)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    h_positions = _extract_line_positions(h_lines, axis="horizontal")
    v_positions = _extract_line_positions(v_lines, axis="vertical")

    if len(h_positions) < 2 or len(v_positions) < 2:
        return None, None

    h_positions = _merge_close_positions(h_positions, threshold=max(h * 0.005, 3))
    v_positions = _merge_close_positions(v_positions, threshold=max(w * 0.004, 3))

    if len(h_positions) < 2 or len(v_positions) < 2:
        return None, None

    return h_positions, v_positions


def detect_column_boundaries(
    img: np.ndarray,
) -> list[int] | None:
    """이미지에서 열 경계(수직선) 위치만 감지한다.

    전체 격자선이 없어도 열 경계만 있으면 동작한다.
    Vision OCR 스프레드 열 보정에 사용.

    Returns:
        수직선 x좌표 리스트 (오름차순), 실패 시 None
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    for binary in _binarize_variants(gray):
        v_len = max(int(h * 0.02), 10)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        v_positions = _extract_line_positions(v_lines, axis="vertical")

        if len(v_positions) >= 5:
            v_merged = _merge_close_positions(v_positions, threshold=max(w * 0.004, 3))
            if len(v_merged) >= 5:
                return v_merged

    return None


def _extract_line_positions(line_img: np.ndarray, axis: str) -> list[int]:
    """감지된 선 이미지에서 선의 위치(좌표)를 추출한다."""
    if axis == "horizontal":
        projection = np.sum(line_img, axis=1)  # 행별 합
    else:
        projection = np.sum(line_img, axis=0)  # 열별 합

    # 임계값 이상인 위치 = 선이 있는 위치
    threshold = np.max(projection) * 0.3
    positions = np.where(projection > threshold)[0]

    if len(positions) == 0:
        return []

    return positions.tolist()


def _merge_close_positions(positions: list[int], threshold: float) -> list[int]:
    """가까운 위치들을 하나로 병합한다."""
    if not positions:
        return []

    positions = sorted(positions)
    merged = [positions[0]]

    for p in positions[1:]:
        if p - merged[-1] > threshold:
            merged.append(p)
        else:
            # 중간값으로 업데이트
            merged[-1] = (merged[-1] + p) // 2

    return merged


def assign_ocr_to_cells(
    ocr_boxes: list[dict],
    cells: list[dict],
) -> pd.DataFrame:
    """OCR 텍스트 박스를 감지된 셀에 매핑한다.

    각 OCR 박스의 중심점이 어느 셀에 속하는지 판별하고,
    같은 셀에 여러 텍스트가 있으면 결합한다.

    Returns:
        행/열이 정확히 분리된 DataFrame
    """
    if not cells or not ocr_boxes:
        return pd.DataFrame()

    num_rows = max(c["row"] for c in cells) + 1
    num_cols = max(c["col"] for c in cells) + 1

    # 셀 매핑 테이블 초기화
    grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]

    for box in ocr_boxes:
        cx, cy = box["x_center"], box["y_center"]
        best_cell = None
        best_dist = float("inf")

        for cell in cells:
            # 중심점이 셀 내부에 있는지 확인
            if (cell["x_min"] <= cx <= cell["x_max"] and
                    cell["y_min"] <= cy <= cell["y_max"]):
                best_cell = cell
                break

            # 내부가 아니면 가장 가까운 셀 선택
            dx = max(cell["x_min"] - cx, 0, cx - cell["x_max"])
            dy = max(cell["y_min"] - cy, 0, cy - cell["y_max"])
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        if best_cell is not None:
            r, c = best_cell["row"], best_cell["col"]
            if grid[r][c]:
                grid[r][c] += " " + box["text"]
            else:
                grid[r][c] = box["text"]

    # 첫 행을 헤더로 사용
    if num_rows >= 2:
        header = [grid[0][c] if grid[0][c] else f"Col_{c}" for c in range(num_cols)]
        data = grid[1:]
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(grid)

    # 완전히 빈 행/열 제거
    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")

    return df


def scan_occupied_cells(
    img: np.ndarray,
    cells: list[dict],
    dark_threshold: int = 100,
    min_dark_ratio: float = 0.008,
) -> dict[tuple[int, int], bool]:
    """각 셀에 텍스트(어두운 픽셀)가 있는지 스캔한다.

    Returns:
        {(row, col): True} — 텍스트가 있는 셀만 포함
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    occupied: dict[tuple[int, int], bool] = {}

    for cell in cells:
        r, c = cell["row"], cell["col"]
        # 경계선 안쪽만 검사 (선 자체를 피함)
        margin_x = max(int((cell["x_max"] - cell["x_min"]) * 0.1), 2)
        margin_y = max(int((cell["y_max"] - cell["y_min"]) * 0.1), 2)
        x1 = cell["x_min"] + margin_x
        y1 = cell["y_min"] + margin_y
        x2 = cell["x_max"] - margin_x
        y2 = cell["y_max"] - margin_y

        if x2 <= x1 or y2 <= y1:
            continue

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 배경색 추정 (모서리 픽셀의 중간값)
        corners = [roi[0, 0], roi[0, -1], roi[-1, 0], roi[-1, -1]]
        bg_val = int(np.median(corners))

        # 배경과 크게 다른 픽셀 = 텍스트
        diff = np.abs(roi.astype(np.int16) - bg_val)
        dark_pixels = np.sum(diff > 40)
        dark_ratio = dark_pixels / roi.size

        if dark_ratio > min_dark_ratio:
            occupied[(r, c)] = True

    return occupied


def get_grid_dimensions(cells: list[dict]) -> tuple[int, int]:
    """셀 리스트에서 (행 수, 열 수)를 반환한다."""
    if not cells:
        return 0, 0
    return max(c["row"] for c in cells) + 1, max(c["col"] for c in cells) + 1


def extract_header_texts_from_cells(
    img: np.ndarray,
    cells: list[dict],
    header_row: int = 0,
) -> dict[int, str]:
    """격자 셀의 헤더 행에서 각 셀을 crop하여 개별 OCR한다.

    어두운 배경(수요예측표 헤더 #002060 등) 셀은 반전 처리 후 인식.

    Returns:
        {col_index: recognized_text} — 인식 성공한 셀만 포함
    """
    import re

    header_cells = [c for c in cells if c["row"] == header_row]
    if not header_cells:
        return {}

    header_cells.sort(key=lambda c: c["col"])
    results = {}

    try:
        from .paddle_engine import _get_ocr
        ocr = _get_ocr()
    except Exception:
        return {}

    for cell in header_cells:
        col = cell["col"]
        x1 = max(0, cell["x_min"] - 1)
        y1 = max(0, cell["y_min"] - 1)
        x2 = min(img.shape[1], cell["x_max"] + 1)
        y2 = min(img.shape[0], cell["y_max"] + 1)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 어두운 배경 감지 → 반전
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        if np.mean(gray) < 128:
            gray = cv2.bitwise_not(gray)
            crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        try:
            raw = ocr.ocr(crop, cls=False)
            if raw and raw[0]:
                text = " ".join(item[1][0] for item in raw[0]).strip()
                # 숫자 or +/- 숫자 패턴 추출
                match = re.search(r"[+-]?\d+\.?\d*", text)
                if match:
                    results[col] = match.group(0)
                elif text:
                    results[col] = text
        except Exception:
            pass

    return results


def detect_header_colors(
    img: np.ndarray,
    cells: list[dict],
) -> str:
    """헤더 영역의 배경색을 감지하여 문서 유형을 추정한다.

    Returns:
        "demand_forecast", "financial_statement", 또는 "default"
    """
    if not cells:
        return "default"

    # 첫 번째 행의 셀들
    header_cells = [c for c in cells if c["row"] == 0]
    if not header_cells:
        return "default"

    # 헤더 영역에서 색상 샘플링
    colors = []
    for cell in header_cells[:5]:
        x1, y1 = cell["x_min"] + 2, cell["y_min"] + 2
        x2, y2 = cell["x_max"] - 2, cell["y_max"] - 2
        if x2 <= x1 or y2 <= y1:
            continue
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        avg_color = roi.mean(axis=(0, 1))  # BGR
        colors.append(avg_color)

    if not colors:
        return "default"

    avg = np.mean(colors, axis=0)  # BGR
    b, g, r = int(avg[0]), int(avg[1]), int(avg[2])

    # 진한 남색 (#002060) → 수요예측표
    if b > 80 and r < 30 and g < 50:
        return "demand_forecast"

    # 연한 파랑 (#D9E2F3) → 재무제표
    if b > 200 and g > 200 and r > 180:
        return "financial_statement"

    # 회색/흰색 → 기본
    return "default"
