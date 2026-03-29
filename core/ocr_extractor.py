"""스캔 PDF / 이미지 파일에서 OCR로 표를 추출한다.

RapidOCR로 텍스트 인식 후, 좌표 기반으로 행/열을 재구성한다.
"""

import os
import tempfile
from typing import Callable

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

# 프로젝트 루트 (core/ 상위)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
_KOREAN_REC_MODEL = os.path.join(_MODELS_DIR, "korean_rec.onnx")
_KOREAN_DICT = os.path.join(_MODELS_DIR, "korean_dict.txt")


def _create_ocr_engine() -> RapidOCR:
    """한국어 모델이 있으면 한국어 OCR, 없으면 기본(중국어) OCR을 반환한다."""
    if os.path.isfile(_KOREAN_REC_MODEL) and os.path.isfile(_KOREAN_DICT):
        return RapidOCR(
            rec_model_path=_KOREAN_REC_MODEL,
            rec_keys_path=_KOREAN_DICT,
        )
    return RapidOCR()


def _upscale_if_small(img: np.ndarray, min_height: int = 1500) -> np.ndarray:
    """작은 이미지는 2배 업스케일하여 OCR 정확도를 높인다."""
    h, w = img.shape[:2]
    if h < min_height:
        scale = max(2, min_height // h + 1)
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return img


def _is_screenshot(img: np.ndarray) -> bool:
    """이미지가 스크린샷인지 스캔 문서인지 판별한다.

    스크린샷: 색상이 선명하고 노이즈가 적음 (분산이 큼, 노이즈가 낮음)
    스캔 문서: 전체적으로 회색톤, 노이즈 있음
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # 라플라시안으로 선명도 측정 — 높으면 스크린샷
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 500


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """OCR 정확도 향상을 위한 이미지 전처리.

    스크린샷: 최소한의 전처리 (원본 정보 보존)
    스캔 문서: CLAHE + 이진화 (노이즈 제거)
    """
    # 1) 업스케일
    img = _upscale_if_small(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if _is_screenshot(img):
        # 스크린샷: 경미한 샤프닝만 적용 (이진화 하지 않음)
        sharpen_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # 스캔 문서: 기존 전처리
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


def _pdf_page_to_image(doc: fitz.Document, page_idx: int, dpi: int = 300) -> np.ndarray:
    """PDF 한 페이지를 numpy 이미지로 변환한다."""
    page = doc[page_idx]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _group_into_rows(boxes: list[dict], y_tolerance: int = 15) -> list[list[dict]]:
    """OCR 결과를 y좌표 기준으로 행별로 그룹핑한다."""
    if not boxes:
        return []

    # y 중심점 기준 정렬
    sorted_boxes = sorted(boxes, key=lambda b: b["y_center"])

    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_boxes[0]]
    current_y = sorted_boxes[0]["y_center"]

    for box in sorted_boxes[1:]:
        if abs(box["y_center"] - current_y) <= y_tolerance:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b["x_center"]))
            current_row = [box]
            current_y = box["y_center"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x_center"]))

    return rows


def _consensus_column_positions(rows: list[list[dict]], num_cols: int) -> list[float]:
    """모든 행의 x좌표를 히스토그램으로 집계하여 열 위치를 결정한다.

    단일 앵커 행 대신 전체 행의 합의(consensus)를 사용하여
    열 분리 정확도를 높인다.
    """
    all_x = sorted(box["x_center"] for row in rows for box in row)
    if len(all_x) < num_cols:
        return _cluster_positions(all_x, num_cols)

    # 히스토그램 기반 피크 감지
    x_min, x_max = min(all_x), max(all_x)
    x_range = x_max - x_min
    if x_range == 0:
        return _cluster_positions(all_x, num_cols)

    # 빈 너비: 평균 텍스트 너비의 절반 정도
    avg_text_w = sum(b["x_max"] - b["x_min"] for row in rows for b in row) / len(all_x)
    bin_width = max(avg_text_w * 0.3, 5)
    num_bins = max(int(x_range / bin_width), num_cols * 2)

    hist, bin_edges = [], []
    step = x_range / num_bins
    for i in range(num_bins):
        lo = x_min + i * step
        hi = lo + step
        count = sum(1 for x in all_x if lo <= x < hi)
        hist.append(count)
        bin_edges.append(lo + step / 2)

    # 히스토그램 피크 찾기 (로컬 최대값)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0:
            peaks.append((hist[i], bin_edges[i]))

    # 피크가 num_cols보다 많으면 상위 num_cols개만
    peaks.sort(reverse=True)
    if len(peaks) >= num_cols:
        col_positions = sorted(p[1] for p in peaks[:num_cols])
    else:
        # 피크 부족 시 gap 기반 폴백
        col_positions = _cluster_positions(all_x, num_cols)

    return col_positions


def _rows_to_dataframe(rows: list[list[dict]], num_cols: int | None = None) -> pd.DataFrame:
    """행별 그룹핑된 데이터를 DataFrame으로 변환한다.

    모든 행의 x좌표 합의(consensus)로 열 위치를 결정한다.
    """
    if not rows:
        return pd.DataFrame()

    if num_cols is None:
        num_cols = max(len(row) for row in rows)

    if num_cols == 0:
        return pd.DataFrame()

    # 합의 기반 열 위치 결정
    col_positions = _consensus_column_positions(rows, num_cols)

    # 열 경계 계산
    col_boundaries = []
    for i in range(len(col_positions) - 1):
        col_boundaries.append((col_positions[i] + col_positions[i + 1]) / 2)

    # 각 셀을 가장 가까운 열에 배치
    result = []
    for row in rows:
        row_data = [""] * num_cols
        for box in row:
            x = box["x_center"]
            # 경계 기반 열 결정
            col_idx = 0
            for boundary in col_boundaries:
                if x > boundary:
                    col_idx += 1
                else:
                    break

            col_idx = min(col_idx, num_cols - 1)

            if row_data[col_idx]:
                row_data[col_idx] += " " + box["text"]
            else:
                row_data[col_idx] = box["text"]
        result.append(row_data)

    # 첫 행을 헤더로 사용
    if len(result) >= 2:
        header = result[0]
        header = [h if h else f"Col_{i}" for i, h in enumerate(header)]
        df = pd.DataFrame(result[1:], columns=header)
    else:
        df = pd.DataFrame(result)

    # 빈 행/열 제거
    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")

    return df


def _cluster_positions(positions: list[float], k: int) -> list[float]:
    """1D 위치 목록을 k개 클러스터로 그룹핑하고 각 중심을 반환한다.

    Gap 기반 분할: 가장 큰 간격 k-1개를 기준으로 클러스터를 나눈다.
    """
    positions = sorted(set(positions))

    if len(positions) <= k:
        result = list(positions)
        while len(result) < k:
            result.append(result[-1] + 100 if result else 0)
        return result

    # 인접 간격 계산 후 가장 큰 gap k-1개를 경계로 사용
    gaps = [(positions[i + 1] - positions[i], i) for i in range(len(positions) - 1)]
    gaps.sort(reverse=True)

    # 상위 k-1개 gap의 인덱스 → 분할 지점
    split_indices = sorted(g[1] for g in gaps[: k - 1])

    # 클러스터 중심 계산
    centers = []
    prev = 0
    for si in split_indices:
        cluster = positions[prev : si + 1]
        centers.append(sum(cluster) / len(cluster))
        prev = si + 1
    # 마지막 클러스터
    cluster = positions[prev:]
    centers.append(sum(cluster) / len(cluster))

    return sorted(centers)


def _parse_ocr_result(result: list) -> list[dict]:
    """RapidOCR 결과를 파싱하여 표준 박스 리스트로 변환한다."""
    boxes = []
    for item in result:
        coords, text, confidence = item
        if confidence < 0.3:
            continue
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        boxes.append({
            "text": text.strip(),
            "x_center": sum(xs) / 4,
            "y_center": sum(ys) / 4,
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "confidence": confidence,
        })
    return boxes


def _extract_tables_from_image(img: np.ndarray, ocr_engine: RapidOCR) -> list[pd.DataFrame]:
    """단일 이미지에서 표를 추출한다.

    1차: OpenCV 격자선 감지 → 셀 경계 기반 매핑 (정확도 높음)
    2차: 좌표 기반 행/열 클러스터링 (격자선 없을 때 폴백)
    """
    from .table_detector import assign_ocr_to_cells, detect_table_cells

    # ── OCR 실행 (원본 이미지에서) ──
    processed = preprocess_image(img)
    if len(processed.shape) == 2:
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_bgr = processed

    result, _ = ocr_engine(processed_bgr)
    if not result:
        return []

    boxes = _parse_ocr_result(result)
    if not boxes:
        return []

    # ── 1차: 격자선 감지 시도 ──
    cells = detect_table_cells(img)

    if cells:
        df = assign_ocr_to_cells(boxes, cells)
        if not df.empty and len(df.columns) >= 3:
            return [df]

    # ── 2차: 좌표 기반 폴백 ──
    heights = [b["y_max"] - b["y_min"] for b in boxes]
    avg_height = sum(heights) / len(heights) if heights else 20
    y_tolerance = avg_height * 0.6

    rows = _group_into_rows(boxes, y_tolerance=int(y_tolerance))
    if not rows:
        return []

    df = _rows_to_dataframe(rows)
    if df.empty:
        return []

    return [df]


def extract_tables_from_scan_pdf(
    pdf_path: str,
    progress_cb: Callable[[int], None] | None = None,
) -> list[pd.DataFrame]:
    """스캔 PDF에서 OCR로 표를 추출한다."""
    ocr_engine = _create_ocr_engine()
    results: list[pd.DataFrame] = []

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    for i in range(total_pages):
        img = _pdf_page_to_image(doc, i)
        tables = _extract_tables_from_image(img, ocr_engine)
        results.extend(tables)

        if progress_cb:
            progress_cb(int((i + 1) / total_pages * 100))

    doc.close()
    return results


def _imread_unicode(path: str) -> np.ndarray | None:
    """한글 경로를 지원하는 이미지 읽기."""
    # cv2.imread는 한글 경로를 지원하지 않으므로 numpy로 우회
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def extract_tables_from_image_file(
    image_path: str,
    progress_cb: Callable[[int], None] | None = None,
) -> list[pd.DataFrame]:
    """이미지 파일(PNG/JPG/BMP/TIFF)에서 표를 추출한다."""
    ocr_engine = _create_ocr_engine()
    img = _imread_unicode(image_path)

    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    if progress_cb:
        progress_cb(10)

    results = _extract_tables_from_image(img, ocr_engine)

    if progress_cb:
        progress_cb(100)

    return results
