"""동적 프롬프트 빌더 — DB 패턴 기반으로 Gemini 프롬프트를 문서 타입별로 강화한다.

처리 건수가 쌓일수록 헤더 힌트, 교정 목록, 메타데이터 키 힌트가 구체화된다.
"""

from .pattern_db import (
    get_column_stats,
    get_corrections,
    get_metadata_patterns,
    get_top_header_patterns,
)

# ── 프롬프트 길이 예산 (chars) ──
_MAX_CORRECTION_CHARS = 600
_MAX_HEADER_COLS = 30
_MAX_CORRECTIONS = 20

# ── 공통 출력 포맷 ──
_OUTPUT_FORMAT = """\
## 출력 형식
JSON만 반환 (마크다운 코드블록 없이):
{
  "metadata": {
    "title": "...",
    "info": [["사채만기","2년"], ...],
    "market_info": [["개별민평","4.658"], ...]
  },
  "headers": [...],
  "rows": [[...], ...]
}
metadata가 없으면 빈 객체 {}로 반환.
"""

# ── 기본 규칙 ──
_BASE_RULES = """\
## 핵심 규칙
1. **모든 열을 빠짐없이 추출하세요.** 헤더에 있는 열은 데이터가 없어도 포함하세요.
2. 한국어 텍스트(회사명, 부서명, 이름)를 정확하게 읽으세요.
   - 부서명에 공백이 있으면 반드시 유지하세요 (예: "채권운용본부 투자전략팀" — 공백 포함).
   - 한글 자모 구분 주의: 하/아, 완/환, 원/윤, 헌/현, 정/장 — 픽셀을 주의 깊게 확인하세요.
3. 숫자는 콤마 없이 순수 숫자로 유지하세요.
4. 빈 셀은 빈 문자열("")로 표시하세요.
5. 표 위의 제목/발행정보/시장정보도 metadata 필드에 포함하세요.
6. 합계, 비중(%), 누적합계, 누적비중(%) 행도 포함하세요.
7. 전일기준 금리, 전일기준 SP 행도 포함하세요.
"""


def build_prompt(
    doc_type: str | None = None,
    image_width_px: int | None = None,
) -> str:
    """
    문서 타입별로 강화된 Gemini 프롬프트를 생성한다.

    Args:
        doc_type: 사전 분류된 문서 타입. None이면 범용 프롬프트.
        image_width_px: 넓은 이미지 경고 문구 삽입용.

    Returns:
        Gemini에 전달할 완성된 프롬프트 문자열.
    """
    if doc_type == "demand_forecast":
        return _build_demand_forecast_prompt()
    elif doc_type == "financial_statement":
        return _build_financial_statement_prompt()
    else:
        return _build_generic_prompt(image_width_px)


# ── 수요예측표 전용 프롬프트 ──

def _build_demand_forecast_prompt() -> str:
    parts = ["이 이미지는 **채권 수요예측표**입니다. 아래 규칙에 따라 추출하세요.\n"]

    parts.append(_BASE_RULES)

    # 학습된 헤더 힌트
    known_headers = get_top_header_patterns("demand_forecast", top_n=1)
    if known_headers:
        hint = _inject_header_hint(known_headers[0])
        parts.append(hint)
    else:
        # 시드 힌트
        parts.append("""\
## 열 순서 힌트 (매우 중요!)
이 표의 열은 **반드시** 다음 순서입니다:
회사명, 부서명, 이름, 계정, 최종참여일시, 총참여금액, [스프레드 열들]

**스프레드 열 규칙:**
- 헤더 행의 짙은 파란색(남색) 배경 위에 흰색 숫자로 표시됩니다.
- 음수(-100, -66, -60, -50 등)부터 양수(+10, +20, +30 등)까지 순서로 나열됩니다.
- 스프레드 열이 10~25개까지 있을 수 있습니다. **하나도 빠뜨리지 마세요.**
- 헤더 행 바로 아래에 "기준일 금리" 또는 "기준일 스프레드" 행이 있을 수 있습니다 (소수점 숫자: 3.506%, 62.6 등).
  이 행도 반드시 rows에 포함하세요.
- 각 데이터 행에서 숫자가 채워진 셀은 보통 1~2개뿐이고, 나머지는 빈 셀("")입니다.
- 합계 행의 열 합산은 총참여금액 합계와 같아야 합니다.

**핵심**: 각 회사의 총참여금액 = 해당 행의 스프레드 열 숫자 합계여야 합니다.
""")

    # 학습된 교정 힌트 — 개인 이름(name)은 다른 이름 오인식 유발 가능성이 높아 제외
    corrections = get_corrections("demand_forecast", exclude_types=["name"])
    if corrections:
        hint = _inject_correction_hints(corrections)
        if hint:
            parts.append(hint)
    else:
        # 시드 교정
        parts.append("""\
## OCR 교정 주의사항
- '흥국증권', '흥국자산운용'의 '흥'을 '홍'으로 잘못 읽지 마세요.
- 날짜 "2026-03-1615:59:51" → "2026-03-16 15:59:51" (공백 주의)
- 합계/비중/누적합계/누적비중(%) 행의 레이블은 '최종참여일시' 열 위치에 씁니다.
""")

    # 학습된 메타데이터 힌트
    meta_patterns = get_metadata_patterns("demand_forecast")
    hint = _inject_metadata_hint(meta_patterns, "demand_forecast")
    if hint:
        parts.append(hint)
    else:
        parts.append("""\
## 메타데이터 구조
metadata.info에는: 사채만기, 모집금액, 모집밴드
metadata.market_info에는: 개별민평, 국고, Spread
전일기준 금리/SP 행은 rows에 포함하세요.
""")

    # 스프레드 수치 범위 힌트 (학습된 경우)
    col_stats = get_column_stats("demand_forecast")
    if col_stats:
        spread_cols = [k for k in col_stats if k.lstrip("+-").isdigit()]
        if spread_cols:
            sp_min = min(int(k) for k in spread_cols)
            sp_max = max(int(k) for k in spread_cols)
            parts.append(f"\n스프레드 열 범위 힌트: {sp_min}bp ~ +{sp_max}bp 사이의 정수 값입니다.\n")

    parts.append(_OUTPUT_FORMAT)
    return "\n".join(parts)


# ── 재무제표 전용 프롬프트 ──

def _build_financial_statement_prompt() -> str:
    parts = ["이 이미지는 **재무제표(재무상태표/손익계산서)**입니다. 아래 규칙에 따라 추출하세요.\n"]
    parts.append(_BASE_RULES)

    known_headers = get_top_header_patterns("financial_statement", top_n=1)
    if known_headers:
        hint = _inject_header_hint(known_headers[0])
        parts.append(hint)
    else:
        parts.append("""\
## 열 구조 힌트
- 첫 번째 열: 계정과목명 (구 분, 항목)
- 나머지 열: 기간별 수치 (제N기, YYYY년 N월말 등)
- [유동자산], [비유동자산] 같은 대분류 행은 별도 행으로 포함하세요.
- ㆍ 또는 · 로 시작하는 항목은 소분류 상세 행입니다.
""")

    corrections = get_corrections("financial_statement", exclude_types=["name"])
    if corrections:
        parts.append(_inject_correction_hints(corrections))
    else:
        parts.append("""\
## OCR 교정 주의사항
- 숫자 0과 O(영문), 1과 l(소문자L) 혼동 주의
- △ 기호 앞의 숫자는 음수입니다
""")

    meta_patterns = get_metadata_patterns("financial_statement")
    hint = _inject_metadata_hint(meta_patterns, "financial_statement")
    if hint:
        parts.append(hint)

    parts.append(_OUTPUT_FORMAT)
    return "\n".join(parts)


# ── 범용 프롬프트 (미분류 문서) ──

def _build_generic_prompt(image_width_px: int | None = None) -> str:
    parts = ["이 이미지에서 표(table) 데이터를 정확하고 빠짐없이 추출해주세요.\n"]
    parts.append(_BASE_RULES)

    if image_width_px and image_width_px > 2000:
        parts.append(f"\n이 이미지는 폭이 {image_width_px}px로 넓습니다. 오른쪽 열이 잘리지 않도록 주의하세요.\n")

    parts.append(_OUTPUT_FORMAT)
    return "\n".join(parts)


# ── 힌트 생성 유틸 ──

def _inject_header_hint(headers: list[str]) -> str:
    truncated = headers[:_MAX_HEADER_COLS]
    col_str = ", ".join(str(h) for h in truncated)
    suffix = f" ... (외 {len(headers)-_MAX_HEADER_COLS}개)" if len(headers) > _MAX_HEADER_COLS else ""
    return f"\n## 열 순서 힌트 (과거 문서 기반)\n다음 순서로 열을 추출하세요: {col_str}{suffix}\n"


def _inject_correction_hints(corrections: list[tuple[str, str]]) -> str:
    if not corrections:
        return ""

    lines = ["## OCR 교정 목록 (학습된 패턴)"]
    total_chars = len(lines[0])

    for wrong, correct in corrections[:_MAX_CORRECTIONS]:
        line = f'- "{wrong}" → "{correct}"'
        if total_chars + len(line) > _MAX_CORRECTION_CHARS:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def _inject_metadata_hint(meta_patterns: dict, doc_type: str) -> str:
    info_labels = meta_patterns.get("info_labels", [])
    market_labels = meta_patterns.get("market_info_labels", [])

    if not info_labels and not market_labels:
        return ""

    lines = ["## 메타데이터 키 힌트 (학습된 패턴)"]
    if info_labels:
        lines.append(f"발행정보 항목: {', '.join(str(l) for l in info_labels[:10])}")
    if market_labels:
        lines.append(f"시장정보 항목: {', '.join(str(l) for l in market_labels[:10])}")

    return "\n".join(lines) + "\n"


def estimate_doc_type_from_image(
    width_px: int,
    height_px: int,
) -> str | None:
    """이미지 비율로 문서 타입을 추정한다. (aspect > 2.5 = 수요예측표)"""
    if height_px > 0 and width_px / height_px > 2.5:
        return "demand_forecast"
    return None
