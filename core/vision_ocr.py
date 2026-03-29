"""Vision AI 기반 OCR — Gemini / Groq로 이미지에서 표를 직접 추출한다.

폴백 체인: Gemini 2.5 Flash → Groq Llama 4 Scout → None (RapidOCR로 폴백)
"""

import base64
import json
import os
import re
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .runtime_flags import get_runtime_flags

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ── API 키 로드 ──

def _load_config() -> dict:
    """config.json에서 API 키들을 로드한다."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    return _load_config().get("groq_api_key", "")


def _load_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    return _load_config().get("gemini_api_key", "")


# ── 공통 유틸 ──

VISION_PROMPT = """\
이 이미지에서 표(table) 데이터를 정확하고 빠짐없이 추출해주세요.

## 핵심 규칙
1. **모든 열을 빠짐없이 추출하세요.** 특히:
   - 왼쪽 텍스트 열 (회사명, 부서명, 이름, 계정 등)
   - 가운데 날짜/금액 열 (최종참여일시, 총참여금액)
   - **오른쪽 숫자 열들** (스프레드별 금액: -100, -66, -60, -56, -50, ... 등)
   - 이 숫자 열들이 매우 중요합니다. 절대 생략하지 마세요!
2. 한국어 텍스트(회사명, 부서명, 이름)를 정확하게 읽으세요.
3. 숫자는 콤마 없이 순수 숫자로 유지하세요.
4. 빈 셀은 빈 문자열("")로 표시하세요.
5. 표 위의 제목/발행정보/시장정보도 metadata 필드에 포함하세요.
6. 표 아래의 합계, 비중(%), 누적합계, 누적비중(%) 행도 포함하세요.
7. 전일기준 금리, 전일기준 SP 행도 포함하세요.
8. **넓은 표**: 스프레드 열이 20개 이상인 경우 오른쪽 끝 열까지 빠짐없이 추출하세요.
9. **기준일 스프레드 행**: 헤더 아래에 소수점 숫자(62.6, 63.6 등)가 나열된 행이 있으면 반드시 포함하세요.

## 출력 형식
JSON만 반환 (마크다운 코드블록 없이):
{
  "metadata": {
    "title": "한솔테크닉스(BBB+) 제93-2회 수요예측 결과(2026/03/16)",
    "info": [
      ["발행정보", ""],
      ["사채만기", "2년"],
      ["모집금액", "150억원"],
      ["모집밴드", "개별민평 -30 ~ +30"]
    ],
    "market_info": [
      ["시장정보", ""],
      ["개별민평", "4.658"],
      ["국고", "3.190"],
      ["Spread", "146.8"]
    ]
  },
  "headers": ["회사명", "부서명", "이름", "계정", "최종참여일시", "총참여금액", "-100", "-66", "-60", ...],
  "rows": [
    ["에이펙스자산운용", "주식운용부문", "송혜인", "집합", "2026-03-16 15:52:13", "20", "20", "", "", ...],
    ...
  ]
}

metadata는 표 위에 있는 정보입니다. 없으면 빈 객체 {}로 반환하세요.
"""

VISION_PROMPT_SPLIT = """\
이 이미지는 큰 표의 {side} 부분입니다.
이미지에 보이는 모든 열과 행을 정확하게 추출하세요.

## 규칙
1. 모든 열을 빠짐없이 추출하세요.
2. 한국어 텍스트를 정확하게 읽으세요.
3. 숫자는 콤마 없이 순수 숫자로 유지하세요.
4. 빈 셀은 빈 문자열("")로 표시하세요.
5. 첫 번째 열(회사명 등 식별 열)은 반드시 포함하세요.

## 출력 형식
JSON만 반환:
{{"headers": [...], "rows": [[...], ...]}}
"""


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def _parse_vision_response(result_text: str) -> dict | None:
    """Vision AI 응답에서 JSON을 추출한다."""
    # 코드블록 내 JSON
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result_text)
    raw = json_match.group(1).strip() if json_match else result_text.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        # JSON이 잘린 경우 복구 시도
        raw = _repair_truncated_json(raw)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def _repair_truncated_json(raw: str) -> str:
    """잘린 JSON을 복구한다 (닫히지 않은 괄호 닫기)."""
    # 마지막 완전한 행까지 자르기
    last_bracket = raw.rfind("]")
    if last_bracket > 0:
        raw = raw[:last_bracket + 1]
    # 닫히지 않은 괄호 닫기
    opens = raw.count("[") - raw.count("]")
    raw += "]" * max(opens, 0)
    opens = raw.count("{") - raw.count("}")
    raw += "}" * max(opens, 0)
    return raw


def _json_to_dataframes(data: dict) -> tuple[list[pd.DataFrame], dict]:
    """파싱된 JSON을 DataFrame과 metadata로 변환한다."""
    metadata = data.get("metadata", {})
    headers = data.get("headers", [])
    rows = data.get("rows", [])

    if not rows:
        return [], metadata

    num_cols = max(len(headers), max(len(r) for r in rows) if rows else 0)

    if not headers:
        headers = [f"Col_{i}" for i in range(num_cols)]
    while len(headers) < num_cols:
        headers.append(f"Col_{len(headers)}")

    normalized_rows = []
    for row in rows:
        while len(row) < num_cols:
            row.append("")
        normalized_rows.append(row[:num_cols])

    df = pd.DataFrame(normalized_rows, columns=headers)
    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")

    return [df] if not df.empty else [], metadata


def _should_split_image(image_path: str, max_width: int = 1800) -> bool:
    """이미지가 넓어서 분할이 필요한지 판단한다."""
    from .ocr_extractor import _imread_unicode
    img = _imread_unicode(image_path)
    if img is None:
        return False
    return img.shape[1] > max_width


def _split_image(image_path: str, overlap_ratio: float = 0.15) -> list[str]:
    """이미지를 좌/우로 분할한다. overlap 포함."""
    import tempfile
    from .ocr_extractor import _imread_unicode

    img = _imread_unicode(image_path)
    if img is None:
        return [image_path]

    h, w = img.shape[:2]
    overlap = int(w * overlap_ratio)
    mid = w // 2

    left = img[:, :mid + overlap]
    right = img[:, mid - overlap:]

    paths = []
    for i, part in enumerate([left, right]):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        cv2.imencode(".png", part)[1].tofile(tmp.name)
        paths.append(tmp.name)

    return paths


def _merge_split_results(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """좌/우 분할 결과를 병합한다. 첫 번째 열(식별키) 기준."""
    if left_df.empty:
        return right_df
    if right_df.empty:
        return left_df

    # 오른쪽 결과에서 왼쪽과 겹치는 열 제거
    left_cols = set(left_df.columns)
    right_unique_cols = [c for c in right_df.columns if c not in left_cols]

    if not right_unique_cols:
        return left_df

    # 첫 번째 열(회사명 등)을 키로 사용하여 행 매칭
    key_col = left_df.columns[0]
    if key_col in right_df.columns:
        merged = pd.merge(
            left_df, right_df[[key_col] + right_unique_cols],
            on=key_col, how="left"
        )
        return merged.fillna("")
    else:
        # 키 매칭 불가 시 인덱스 기반 병합
        right_part = right_df[right_unique_cols].reset_index(drop=True)
        left_reset = left_df.reset_index(drop=True)
        min_len = min(len(left_reset), len(right_part))
        return pd.concat(
            [left_reset.iloc[:min_len], right_part.iloc[:min_len]], axis=1
        ).fillna("")


# ── Gemini Vision ──

def extract_with_gemini(
    image_path: str,
    api_key: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
    prompt_override: str | None = None,
) -> tuple[list[pd.DataFrame], dict] | None:
    """Gemini 2.5 Flash로 이미지에서 표를 추출한다."""
    key = api_key or _load_gemini_key()
    if not key or not GEMINI_AVAILABLE:
        return None

    if progress_cb:
        progress_cb(10)

    client = genai.Client(api_key=key)
    prompt = prompt_override or VISION_PROMPT

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        mime = _detect_mime_type(image_path)

        if progress_cb:
            progress_cb(20)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                genai_types.Part.from_bytes(data=image_data, mime_type=mime),
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=32768,
            ),
        )

        if progress_cb:
            progress_cb(80)

        result_text = response.text or ""
        data = _parse_vision_response(result_text)
        if not data:
            return None

        dfs, metadata = _json_to_dataframes(data)

        if progress_cb:
            progress_cb(100)

        return (dfs, metadata) if dfs else None

    except Exception as e:
        print(f"[Gemini] 오류: {e}")
        return None


# ── Groq Vision ──

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def extract_with_groq(
    image_path: str,
    api_key: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
    prompt_override: str | None = None,
) -> tuple[list[pd.DataFrame], dict] | None:
    """Groq Llama 4 Scout로 이미지에서 표를 추출한다."""
    key = api_key or _load_groq_key()
    if not key or not GROQ_AVAILABLE:
        return None

    if progress_cb:
        progress_cb(10)

    b64 = _image_to_base64(image_path)
    mime = _detect_mime_type(image_path)

    if progress_cb:
        progress_cb(20)

    client = Groq(api_key=key)
    prompt = prompt_override or VISION_PROMPT

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=8192,
        )

        if progress_cb:
            progress_cb(80)

        result_text = response.choices[0].message.content or ""
        data = _parse_vision_response(result_text)
        if not data:
            return None

        dfs, metadata = _json_to_dataframes(data)

        if progress_cb:
            progress_cb(100)

        return (dfs, metadata) if dfs else None

    except Exception as e:
        print(f"[Groq] 오류: {e}")
        return None


# ── 스프레드 열 보정 (OpenCV 격자선 기반) ──

def correct_spread_placement(
    image_path: str,
    dfs: list[pd.DataFrame],
    metadata: dict,
) -> list[pd.DataFrame]:
    """OpenCV 셀 스캔으로 스프레드 열 배치를 best-effort 보정한다.

    각 셀의 픽셀을 분석하여 어느 스프레드 열에 값이 있는지 감지하고,
    Gemini 결과의 잘못된 열 배치를 수정한다.
    격자 감지 실패 또는 열 수 불일치 시 원본을 그대로 반환한다.
    """
    from .ocr_extractor import _imread_unicode
    from .table_detector import detect_table_cells, get_grid_dimensions, scan_occupied_cells

    if not dfs or dfs[0].empty:
        return dfs

    df = dfs[0]
    img = _imread_unicode(image_path)
    if img is None:
        return dfs

    cells = detect_table_cells(img)
    if not cells:
        return dfs

    grid_rows, grid_cols = get_grid_dimensions(cells)

    spread_start = _find_spread_start_col(df)
    if spread_start is None or spread_start >= len(df.columns):
        return dfs

    num_spread_cols = len(df.columns) - spread_start
    if num_spread_cols < 3 or num_spread_cols > 22:
        return dfs

    # 격자에서 좁은 열(스프레드) 시작점 찾기
    col_widths = {}
    for c in cells:
        if c["row"] == 0:
            col_widths[c["col"]] = c["x_max"] - c["x_min"]
    if not col_widths:
        return dfs

    widths = [w for _, w in sorted(col_widths.items())]
    median_w = sorted(widths)[len(widths) // 2]

    grid_spread_start_col = 0
    for i in range(len(widths) - 1, -1, -1):
        if widths[i] > median_w * 1.3:
            grid_spread_start_col = i + 1
            break

    grid_spread_count = grid_cols - grid_spread_start_col
    # 격자 열 수와 DataFrame 열 수가 크게 다르면 보정 포기
    if abs(grid_spread_count - num_spread_cols) > 3:
        return dfs

    # 셀 점유 스캔
    occupied = scan_occupied_cells(img, cells)

    # 헤더 행 찾기
    header_grid_row = 0
    max_occ = 0
    for r in range(min(5, grid_rows)):
        count = sum(1 for c in range(grid_spread_start_col, grid_cols) if (r, c) in occupied)
        if count > max_occ:
            max_occ = count
            header_grid_row = r

    # DataFrame 데이터 행 범위
    df_data_start, df_data_end = _find_df_data_range(df, spread_start)

    # 격자 데이터 행 추출
    grid_data_rows = []
    for gr in range(header_grid_row + 1, grid_rows):
        has_text = any((gr, c) in occupied for c in range(min(grid_spread_start_col, 3)))
        occ_spread = [c - grid_spread_start_col
                      for c in range(grid_spread_start_col, grid_cols)
                      if (gr, c) in occupied]
        occ_spread = [c for c in occ_spread if c < num_spread_cols]
        if has_text or occ_spread:
            grid_data_rows.append((gr, occ_spread))

    # 보정
    corrected = 0
    for df_idx, (_, occ_cols) in zip(range(df_data_start, df_data_end), grid_data_rows):
        if not occ_cols:
            continue

        spread_vals = []
        for si in range(num_spread_cols):
            val = df.iloc[df_idx, spread_start + si]
            if val is not None and str(val).strip() and str(val) not in ("", "nan"):
                try:
                    float(str(val).replace(",", ""))
                    spread_vals.append((si, val))
                except (ValueError, TypeError):
                    pass

        if not spread_vals and len(occ_cols) == 1:
            try:
                total_val = df.iloc[df_idx, spread_start - 1]
                if total_val is not None and str(total_val).strip() and str(total_val) != "nan":
                    df.iloc[df_idx, spread_start + occ_cols[0]] = total_val
                    corrected += 1
            except (IndexError, ValueError):
                pass
            continue

        if len(spread_vals) == 1 and len(occ_cols) == 1:
            old_col = spread_vals[0][0]
            new_col = occ_cols[0]
            if old_col != new_col:
                df.iloc[df_idx, spread_start + old_col] = ""
                df.iloc[df_idx, spread_start + new_col] = spread_vals[0][1]
                corrected += 1
        elif len(spread_vals) == 1 and len(occ_cols) > 1:
            old_col = spread_vals[0][0]
            closest = min(occ_cols, key=lambda c: abs(c - old_col))
            if closest != old_col:
                df.iloc[df_idx, spread_start + old_col] = ""
                df.iloc[df_idx, spread_start + closest] = spread_vals[0][1]
                corrected += 1

    if corrected > 0:
        print(f"[GridCorrect] {corrected}개 셀 스프레드 열 보정")

    df = df.fillna("")
    return [df]


def _find_spread_start_col(df: pd.DataFrame) -> int | None:
    """DataFrame에서 스프레드 열의 시작 인덱스를 찾는다."""
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        # 4.078% (-50) 같은 형식
        if "(" in col_str and ")" in col_str:
            inner = col_str[col_str.rfind("(") + 1:col_str.rfind(")")].strip()
            cleaned = inner.lstrip("+-").replace(".", "")
            if cleaned.isdigit():
                return i
        # 순수 숫자 열 (-100, -50, 5, 22 등)
        cleaned = col_str.lstrip("+-").replace(".", "").replace("%", "")
        if cleaned.isdigit():
            return i
    for i, col in enumerate(df.columns):
        if "참여금액" in str(col) or "총참여" in str(col):
            return i + 1
    return None


def _find_grid_spread_start(
    col_boundaries: list[int], num_spread_cols: int
) -> int | None:
    """격자 열 경계에서 스프레드 열의 시작 인덱스를 찾는다.

    전략: 오른쪽에서부터 균일한 폭의 열이 끝나는 지점을 찾는다.
    스프레드 열은 모두 같은 폭 (±15%), 텍스트 열은 그보다 넓다.
    """
    if len(col_boundaries) < 3:
        return None

    widths = [col_boundaries[i + 1] - col_boundaries[i]
              for i in range(len(col_boundaries) - 1)]
    if not widths:
        return None

    # 오른쪽 절반 열의 중간 폭 (스프레드 열의 대표 폭)
    right_half = widths[len(widths) // 2:]
    median_width = sorted(right_half)[len(right_half) // 2]

    # 오른쪽에서 왼쪽으로 스캔하며 폭이 크게 달라지는 지점 찾기
    spread_start_idx = 0
    for i in range(len(widths) - 1, -1, -1):
        if widths[i] > median_width * 1.3:
            spread_start_idx = i + 1
            break

    # 감지된 스프레드 열 수
    detected_spread = len(widths) - spread_start_idx

    # num_spread_cols와 차이가 크면 폴백
    if abs(detected_spread - num_spread_cols) > 3:
        # 폴백: 뒤에서 num_spread_cols번째
        spread_start_idx = max(0, len(widths) - num_spread_cols)

    return spread_start_idx


def _find_df_data_range(df: pd.DataFrame, spread_start: int) -> tuple[int, int]:
    """DataFrame에서 데이터 행의 시작/끝 인덱스를 찾는다."""
    data_start = 0
    data_end = len(df)

    for i in range(len(df)):
        first_col = str(df.iloc[i, 0]).strip() if df.iloc[i, 0] is not None else ""
        fourth_col = ""
        if len(df.columns) > 3:
            fourth_col = str(df.iloc[i, 3]).strip() if df.iloc[i, 3] is not None else ""

        if "전일기준" in first_col:
            data_start = max(data_start, i + 1)
            continue

        for label in ("합계", "비중(%)", "누적합계", "누적비중(%)"):
            if first_col == label or fourth_col == label:
                data_end = min(data_end, i)
                return data_start, data_end

    return data_start, data_end


# ── 통합 엔트리포인트 ──

def extract_with_vision(
    image_path: str,
    api_key: str | None = None,
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
    engine: str = "auto",
    prompt_override: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
    consensus_runs: int = 1,
    benchmark_mode: bool | None = None,
) -> tuple[list[pd.DataFrame], dict] | None:
    """Vision AI로 이미지에서 표를 추출한다.

    폴백 체인:
      ① 캐시 → HIT → 즉시 반환
      ② EasyOCR 로컬 추출 → 신뢰도 ≥ 0.85 → 반환
      ③ Groq API → 텍스트 열 교차검증 → 반환
      ④ Gemini API → 최후 수단

    Args:
        engine: "auto", "gemini", "groq", "local"
        prompt_override: 동적 생성된 문서 타입 특화 프롬프트
        api_key: 레거시 호환 (groq_api_key로 사용)
        consensus_runs: Groq 합의 투표 실행 횟수 (1이면 비활성)
    """
    groq_key = groq_api_key or api_key or _load_groq_key()
    flags = get_runtime_flags(benchmark_mode)

    # ── ① 결과 캐시 조회 ──
    file_hash = None
    phash = None
    if not flags.disable_cache:
        try:
            from .result_cache import (
                compute_perceptual_hash, lookup_cache, lookup_similar,
                store_result, cache_to_dataframes,
            )
            from .pattern_db import compute_file_hash
            file_hash = compute_file_hash(image_path)
            cached = lookup_cache(file_hash)
            if cached:
                print(f"[Cache] 정확 매칭 HIT (verified={cached['verified']})")
                dfs, meta = cache_to_dataframes(cached)
                if dfs:
                    if progress_cb:
                        progress_cb(100)
                    return (dfs, meta)

            phash = compute_perceptual_hash(image_path)
            similar = lookup_similar(phash)
            if similar:
                print(f"[Cache] 유사 매칭 HIT (verified)")
                dfs, meta = cache_to_dataframes(similar)
                if dfs:
                    if progress_cb:
                        progress_cb(100)
                    return (dfs, meta)
        except Exception as e:
            print(f"[Cache] 조회 실패 (무시): {e}")

    # ── ② 로컬 OCR (PaddleOCR → EasyOCR 폴백) ──
    local_result = None
    local_confidence = 0.0
    if engine in ("auto", "local"):
        try:
            from .local_ocr_engine import extract_with_local_ocr
            lr = extract_with_local_ocr(
                image_path,
                progress_cb=lambda p: progress_cb(int(p * 0.4)) if progress_cb else None,
                benchmark_mode=flags.benchmark_mode,
            )
            if lr:
                local_dfs, local_meta, local_confidence = lr
                local_result = (local_dfs, local_meta)
                print(f"[LocalOCR] 추출 완료 (confidence={local_confidence:.2f})")

                if local_confidence >= 0.80:
                    print(f"[LocalOCR] 신뢰도 충분 → 로컬 결과 사용 (API 0회)")
                    result = local_result
                    _cache_result(file_hash, phash, result)
                    return result
                else:
                    print(f"[LocalOCR] 신뢰도 부족 ({local_confidence:.2f}) → API 폴백")
        except Exception as e:
            print(f"[EasyOCR] 로컬 추출 실패: {e}")

    if engine == "local":
        # 로컬 전용 모드
        return local_result

    # ── ③ Groq API (텍스트 열 교차검증) ──
    result = None
    need_split = _should_split_image(image_path)

    if engine in ("auto", "groq"):
        if consensus_runs > 1 and not need_split:
            try:
                from .consensus import consensus_extract
                result = consensus_extract(
                    image_path, groq_key,
                    num_runs=consensus_runs,
                    progress_cb=lambda p: progress_cb(40 + int(p * 0.3)) if progress_cb else None,
                    prompt_override=prompt_override,
                )
            except Exception as e:
                print(f"[Consensus] 합의 투표 실패: {e}")

        if not result and need_split:
            parts = _split_image(image_path)
            if len(parts) == 2:
                left_result = extract_with_groq(parts[0], groq_key)
                right_result = extract_with_groq(parts[1], groq_key)
                for p in parts:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                if left_result and right_result:
                    merged = _merge_split_results(left_result[0][0], right_result[0][0])
                    meta = left_result[1] or right_result[1]
                    result = ([merged], meta)

        if not result:
            result = extract_with_groq(
                image_path, groq_key,
                progress_cb=lambda p: progress_cb(40 + int(p * 0.3)) if progress_cb else None,
                prompt_override=prompt_override,
            )

        # 교차검증: 로컬 신뢰도가 충분할 때만
        if (
            result and local_result and local_confidence >= 0.75
            and not flags.disable_entity_corrections
            and not flags.disable_db_corrections
        ):
            result = _cross_validate(local_result, result)

    # ── ④ Gemini API (최후 수단, 한도 체크) ──
    if not result and engine in ("auto", "gemini"):
        if _check_gemini_quota():
            result = extract_with_gemini(image_path, gemini_api_key,
                                         progress_cb=lambda p: progress_cb(70 + int(p * 0.2)) if progress_cb else None,
                                         prompt_override=prompt_override)
            if result:
                _increment_gemini_usage()
            if (
                result and local_result and local_confidence >= 0.75
                and not flags.disable_entity_corrections
                and not flags.disable_db_corrections
            ):
                result = _cross_validate(local_result, result)
        else:
            print("[Gemini] 일일 한도 초과 → 건너뜀")

    if not result:
        # EasyOCR 결과라도 반환
        if local_result:
            result = local_result
        else:
            return None

    # ── 후처리 ──
    # OpenCV 격자선 기반 스프레드 열 보정
    try:
        dfs, metadata = result
        corrected_dfs = correct_spread_placement(image_path, dfs, metadata)
        result = (corrected_dfs, metadata)
    except Exception as e:
        print(f"[GridCorrect] 보정 중 오류 (원본 유지): {e}")

    # 엔티티 사전 퍼지 매칭 + DB 교정
    try:
        dfs, metadata = result
        if not flags.disable_entity_corrections and not flags.disable_db_corrections:
            dfs = _apply_entity_corrections(dfs)
            result = (dfs, metadata)
    except Exception:
        pass

    # 결과 캐시 저장
    _cache_result(file_hash, phash, result)

    return result


def _cross_validate(
    local_result: tuple,
    api_result: tuple,
) -> tuple:
    """로컬 OCR(기본) + API(보조) 교차검증.

    원칙:
    - 로컬 OCR + 엔티티 사전 교정 결과를 BASE로 사용
    - API는 로컬이 빈 셀이거나 엔티티 미매칭일 때만 보완
    - 양쪽이 일치하면 → 확정 (최고 신뢰)
    - 양쪽이 다르면 → 엔티티 사전 매칭 여부로 판정
    - 숫자 열: 일치 시 확정, 불일치 시 로컬 우선
    """
    try:
        local_dfs, local_meta = local_result
        api_dfs, api_meta = api_result

        if not local_dfs or not api_dfs:
            return api_result if api_dfs else local_result

        local_df = local_dfs[0]
        api_df = api_dfs[0]

        # 로컬 결과를 BASE로 사용
        merged = api_df.copy()  # API의 구조(헤더/행수)를 사용 (구조 파악은 API가 우수)

        # 엔티티 사전 로드 (매칭 판정용)
        known_entities = set()
        try:
            from .entity_dict import get_all_entities
            for etype in ("company", "department", "account"):
                known_entities.update(name for _, name in get_all_entities(etype))
        except Exception:
            pass

        n_rows = min(len(merged), len(local_df))
        n_cols = min(merged.shape[1], local_df.shape[1])

        for ri in range(n_rows):
            for ci in range(n_cols):
                api_val = str(merged.iloc[ri, ci]).strip()
                local_val = str(local_df.iloc[ri, ci]).strip()

                # 일치 → 확정
                if api_val == local_val:
                    continue

                col_name = str(merged.columns[ci]).strip()
                is_numeric = col_name.lstrip("+-").isdigit() or "금액" in col_name or "합계" in col_name

                if is_numeric:
                    # 숫자 열: 로컬 우선 (숫자 인식 정확)
                    if local_val and local_val.replace(",", "").replace("-", "").replace(".", "").isdigit():
                        merged.iloc[ri, ci] = local_val
                elif ci < 4:
                    # 텍스트 열 (회사명/부서명/이름/계정): 엔티티 매칭 기반 판정
                    local_in_dict = local_val in known_entities
                    api_in_dict = api_val in known_entities

                    if local_in_dict and not api_in_dict:
                        merged.iloc[ri, ci] = local_val  # 로컬이 사전에 있음
                    elif not local_in_dict and api_in_dict:
                        pass  # API가 사전에 있음 → API 유지
                    elif local_in_dict and api_in_dict:
                        pass  # 둘 다 사전에 있음 → API 유지 (구조 파악 우수)
                    else:
                        # 둘 다 사전에 없음 → API 유지 (텍스트 인식은 API가 우수)
                        pass

        # 메타데이터는 API 것 사용 (구조 파악 우수)
        return ([merged], api_meta or local_meta)
    except Exception:
        return api_result


def _check_gemini_quota() -> bool:
    """오늘 남은 Gemini 한도를 확인한다."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
        )
        import json
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        daily_limit = cfg.get("gemini_daily_limit", 20)
        reserve = cfg.get("gemini_reserve", 5)
        usable = daily_limit - reserve

        from .pattern_db import _conn
        from datetime import date
        today = date.today().isoformat()
        with _conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS gemini_usage (date TEXT PRIMARY KEY, call_count INTEGER DEFAULT 0)"
            )
            row = con.execute("SELECT call_count FROM gemini_usage WHERE date=?", (today,)).fetchone()
            used = row["call_count"] if row else 0
        return used < usable
    except Exception:
        return True  # 에러 시 허용


def _increment_gemini_usage():
    """오늘의 Gemini 사용 횟수를 1 증가."""
    try:
        from .pattern_db import _conn
        from datetime import date
        today = date.today().isoformat()
        with _conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS gemini_usage (date TEXT PRIMARY KEY, call_count INTEGER DEFAULT 0)"
            )
            con.execute(
                "INSERT INTO gemini_usage (date, call_count) VALUES (?, 1) "
                "ON CONFLICT(date) DO UPDATE SET call_count=call_count+1",
                (today,),
            )
    except Exception:
        pass


def _cache_result(file_hash, phash, result):
    """결과를 캐시에 저장한다."""
    if get_runtime_flags().disable_cache:
        return
    try:
        if file_hash and result:
            from .result_cache import store_result
            dfs, metadata = result
            cache_data = {
                "metadata": metadata,
                "headers": list(dfs[0].columns) if dfs else [],
                "rows": dfs[0].astype(str).values.tolist() if dfs else [],
            }
            store_result(file_hash, phash or "", "demand_forecast", cache_data)
    except Exception:
        pass


def _apply_entity_corrections(dfs: list) -> list:
    """엔티티 사전 퍼지 매칭 + pattern_db 교정을 적용한다."""
    # DB 교정 — 후처리에서는 이름 교정도 포함 (정확 문자열 치환이므로 안전)
    # (프롬프트에서만 이름 교정 제외 — AI가 다른 이름에 오적용 방지)
    try:
        from .pattern_db import get_corrections
        corrections = get_corrections("demand_forecast")  # 이름 포함
    except Exception:
        corrections = []

    # 엔티티 사전 초기화
    entity_available = False
    try:
        from .entity_dict import (
            correct_text_column, initialize_seed_entities, _ensure_tables,
        )
        _ensure_tables()
        initialize_seed_entities()
        entity_available = True
    except Exception:
        pass

    corrected = []
    for df in dfs:
        df = df.copy()
        text_cols = [c for c in df.columns[:6]
                     if str(df[c].dtype) in ("object", "string")]

        # 1) DB 교정 (정확 문자열 치환)
        for wrong, correct in (corrections or []):
            for col in text_cols:
                df[col] = df[col].astype(str).str.replace(wrong, correct, regex=False)

        # 2) 엔티티 사전 퍼지 매칭
        if entity_available:
            for ci, col in enumerate(text_cols):
                col_name = str(col).lower()
                if "회사" in col_name or ci == 0:
                    df[col] = correct_text_column(df[col], "company", max_jamo_dist=4)
                elif "부서" in col_name or ci == 1:
                    df[col] = correct_text_column(df[col], "department", max_jamo_dist=4)
                elif "계정" in col_name or ci == 3:
                    df[col] = correct_text_column(df[col], "account", max_jamo_dist=2)

        corrected.append(df)
    return corrected


# 레거시 호환
def _load_api_key() -> str:
    return _load_groq_key()
