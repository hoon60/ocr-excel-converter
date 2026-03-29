"""Groq AI를 활용한 원본 표 스타일 분석 모듈.

OCR 추출 데이터와 함께, AI가 원본 이미지의 표 스타일을 분석하여
Excel에 재현할 스타일 정보를 반환한다:
- 행별 타입 (헤더/대분류/세부항목/소계/기간구분)
- 누락된 항목명 추정
- 셀 병합 정보
"""

import json
import os
import re
from typing import Any

import pandas as pd

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def _load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f).get("groq_api_key", "")
        except (json.JSONDecodeError, OSError):
            pass
    return ""


STYLE_PROMPT = """\
당신은 한국어 재무제표/표 문서 전문가입니다.
OCR로 추출된 표 데이터를 보고, 각 행의 역할과 누락된 항목명을 추정해주세요.

## 입력 형식
각 행이 | 로 구분된 표 데이터입니다.

## 출력 규칙
각 행에 대해 JSON 객체를 반환하세요:
- "row": 행 번호 (0부터)
- "type": 행 타입 ("title" / "header" / "period" / "category" / "detail" / "subtotal" / "note" / "normal")
  - title: 표 제목 (1. 요약재무정보 등)
  - header: 열 헤더 (구분, 제57기 등)
  - period: 기간 표시 (2025년 12월말, 2025년 1월~12월 등)
  - category: 대분류 ([유동자산], [비유동부채] 등)
  - detail: 세부항목 (ㆍ현금및현금성자산, ·단기금융상품 등)
  - subtotal: 소계/합계 (자산총계, 부채총계, 매출액, 영업이익 등)
  - note: 주석
  - normal: 기타
- "label": 첫 열이 비어있으면 문맥상 추정되는 항목명 (확실한 경우만, 아니면 null)
- "merge_with_above": 이 행이 위 행과 병합된 기간 행이면 true

반드시 JSON 배열로만 반환하세요. 다른 텍스트 없이.
"""


def analyze_table_style(
    df: pd.DataFrame,
    api_key: str | None = None,
) -> list[dict[str, Any]] | None:
    """AI로 표의 행별 스타일 정보를 분석한다.

    Returns:
        행별 스타일 정보 리스트. API 실패 시 None.
    """
    key = api_key or _load_api_key()
    if not key or not GROQ_AVAILABLE:
        return None

    # 표 데이터를 텍스트로 변환
    lines = []
    for i in range(len(df)):
        row_vals = [str(df.iloc[i, c]) if pd.notna(df.iloc[i, c]) else "" for c in range(len(df.columns))]
        lines.append(f"행{i}: {' | '.join(row_vals)}")

    table_text = "\n".join(lines)

    prompt = f"다음 OCR 추출 표를 분석해주세요.\n\n{table_text}"

    client = Groq(api_key=key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": STYLE_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=4096,
        )
        result_text = response.choices[0].message.content or ""

        # JSON 추출
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result_text)
        raw = json_match.group(1).strip() if json_match else result_text.strip()
        data = json.loads(raw)

        if isinstance(data, list):
            return data

    except (json.JSONDecodeError, Exception):
        pass

    return None
