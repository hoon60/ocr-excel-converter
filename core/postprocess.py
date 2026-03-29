"""숫자 검증 및 후처리 레이어."""

import re

import pandas as pd

# 원화 금액 패턴: ₩1,234 또는 1234원 (₩ 또는 원 기호가 반드시 있어야 함)
RE_CURRENCY = re.compile(r"^₩\s*[\d,]+$|^[\d,]+\s*원$")

# 백분율: 23.5% 또는 -1.2%
RE_PERCENT = re.compile(r"^([+-]?\d+\.?\d*)\s*%$")

# 날짜: 2024-01-15, 2024.01.15, 2024/01/15
RE_DATE = re.compile(
    r"^(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})$"
)

# 일반 숫자: 1,234.56 또는 -0.5
RE_NUMBER = re.compile(r"^[+-]?[\d,]+\.?\d*$")


def detect_cell_type(value: str) -> str:
    """셀 값의 타입을 판별한다.

    Returns:
        "currency", "percent", "date", "number", "text" 중 하나.
    """
    v = value.strip()
    if not v:
        return "text"
    if RE_CURRENCY.match(v):
        return "currency"
    if RE_PERCENT.match(v):
        return "percent"
    if RE_DATE.match(v):
        return "date"
    if RE_NUMBER.match(v):
        return "number"
    return "text"


def parse_numeric(value: str) -> float | None:
    """문자열을 숫자로 변환. 실패 시 None."""
    v = value.strip()
    # 통화 기호/단위 제거
    v = re.sub(r"[₩원\s]", "", v)
    # % 제거
    v = v.rstrip("%").strip()
    # 콤마 제거
    v = v.replace(",", "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def validate_and_annotate(
    tables: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[list[tuple[int, int]]]]:
    """각 표의 숫자 셀을 검증하고, 오류 셀 좌표를 반환한다.

    숫자처럼 보이지만 파싱에 실패한 셀은 오류로 표시한다.

    Returns:
        (정제된 테이블 리스트, 각 테이블별 오류 셀 좌표 리스트)
    """
    all_error_cells: list[list[tuple[int, int]]] = []

    for df in tables:
        error_cells: list[tuple[int, int]] = []

        for col_idx, col in enumerate(df.columns):
            for row_idx in range(len(df)):
                val = str(df.iloc[row_idx, col_idx]).strip()
                if not val:
                    continue

                cell_type = detect_cell_type(val)

                if cell_type in ("currency", "percent", "number"):
                    parsed = parse_numeric(val)
                    if parsed is None:
                        # 숫자로 보이지만 파싱 실패 → 오류 셀
                        error_cells.append((row_idx, col_idx))

        all_error_cells.append(error_cells)

    return tables, all_error_cells


# ── 수요예측표 전용: 열 합계 교차검증 ──

def _find_total_col(df: pd.DataFrame) -> int | None:
    """'총참여금액' 열 인덱스를 찾는다."""
    for i, col in enumerate(df.columns):
        if "총참여" in str(col) or "참여금액" in str(col):
            return i
    return None


def _find_sum_row(df: pd.DataFrame) -> int | None:
    """'합계' 레이블이 있는 행 인덱스를 찾는다."""
    for i in range(len(df)):
        for ci in range(min(6, df.shape[1])):
            if str(df.iloc[i, ci]).strip() == "합계":
                return i
    return None


def _find_spread_cols(df: pd.DataFrame) -> list[int]:
    """스프레드 열 인덱스 목록을 반환한다 (-100, -50, +30 등 숫자 헤더 또는 Col_N)."""
    spread_cols = []
    for i, col in enumerate(df.columns):
        s = str(col).strip()
        # Col_N placeholder
        if re.match(r"^Col_\d+$", s):
            spread_cols.append(i)
            continue
        # 순수 숫자 (+/- 부호 포함)
        cleaned = s.lstrip("+-").replace(".", "").replace("%", "")
        if cleaned.isdigit() and i >= 5:
            spread_cols.append(i)
            continue
        # "(+30)" 같은 형식
        if "(" in s and ")" in s:
            inner = s[s.rfind("(") + 1:s.rfind(")")].strip().lstrip("+-")
            if inner.replace(".", "").isdigit():
                spread_cols.append(i)
    return spread_cols


def _safe_float(val) -> float | None:
    """안전하게 float 변환. 실패 시 None."""
    s = str(val).strip().replace(",", "")
    if not s or s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def validate_demand_forecast_sums(
    tables: list[pd.DataFrame],
    doc_type: str,
) -> tuple[list[pd.DataFrame], list[list[tuple[int, int]]]]:
    """수요예측표의 열 합계와 합계 행 교차검증 + 자동 보정.

    Returns:
        (보정된 tables, 각 테이블별 오류 셀 좌표 리스트)
    """
    all_errors: list[list[tuple[int, int]]] = [[] for _ in tables]

    if doc_type != "demand_forecast":
        return tables, all_errors

    for t_idx, df in enumerate(tables):
        if df.empty or df.shape[1] < 7:
            continue

        total_col = _find_total_col(df)
        sum_row = _find_sum_row(df)
        spread_cols = _find_spread_cols(df)

        if sum_row is None or not spread_cols:
            continue

        data_end = sum_row  # 합계 행 전까지가 데이터

        # ── 행 검증: 각 행의 스프레드 합 == 총참여금액 ──
        if total_col is not None:
            for ri in range(data_end):
                total_val = _safe_float(df.iloc[ri, total_col])
                if total_val is None or total_val == 0:
                    continue

                spread_sum = 0.0
                spread_entries = []  # (col_idx, value)
                for ci in spread_cols:
                    v = _safe_float(df.iloc[ri, ci])
                    if v is not None and v != 0:
                        spread_sum += v
                        spread_entries.append((ci, v))

                if abs(spread_sum - total_val) > 0.01 and spread_entries:
                    # 단일 값인데 합계와 다르면 → 해당 행의 값이 잘못된 열에 있을 수 있음
                    if len(spread_entries) == 1:
                        old_ci, val = spread_entries[0]
                        if abs(val - total_val) < 0.01:
                            pass  # 값 자체는 맞지만 열이 틀릴 수 있음 → 격자 보정에 맡김
                        else:
                            all_errors[t_idx].append((ri, old_ci))
                    else:
                        # 복수 값인데 합계 불일치 → 오류 표시
                        for ci, _ in spread_entries:
                            all_errors[t_idx].append((ri, ci))

        # ── 열 검증: 각 스프레드 열의 데이터합 == 합계 행 값 ──
        for ci in spread_cols:
            expected = _safe_float(df.iloc[sum_row, ci])
            if expected is None:
                continue

            col_sum = 0.0
            for ri in range(data_end):
                v = _safe_float(df.iloc[ri, ci])
                if v is not None:
                    col_sum += v

            if abs(col_sum - expected) > 0.01:
                all_errors[t_idx].append((sum_row, ci))

    return tables, all_errors
