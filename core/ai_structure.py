"""Groq AI 기반 테이블 구조 재구성.

OCR 결과의 구조적 문제(열 병합, 영역 혼재)를 AI가 분석하여 교정한다.
- 문서 유형 자동 감지 (재무제표/수요예측표/일반 표)
- 열 분리 교정 (부서명+이름 합쳐진 경우 분리)
- 영역 구분 (제목/메타데이터/헤더/데이터/합계)
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


STRUCTURE_PROMPT = """\
당신은 한국어 문서 표 구조 분석 전문가입니다.
OCR로 추출된 표 데이터를 받아, 문서 유형을 판별하고 구조를 교정합니다.

## 분석 항목
1. **문서 유형**: "financial_statement"(재무제표), "demand_forecast"(수요예측표), "general"(일반 표)
2. **열 헤더**: 올바른 열 이름 목록
3. **데이터 시작 행**: 실제 데이터가 시작되는 행 번호 (0-indexed)
4. **합계 행**: 합계/비중/누적 등 요약 행 번호들

## 출력 형식 (JSON만 반환)
{
  "doc_type": "demand_forecast",
  "columns": ["회사명", "부서명", "이름", "계정", "최종참여일시", "총참여금액", "-100", "-66", ...],
  "data_start_row": 7,
  "summary_rows": [35, 36, 37, 38],
  "title": "한솔테크닉스(BBB+) 제93-2회 수요예측 결과",
  "metadata": {"사채만기": "2년", "모집금액": "150억원"}
}
"""


def analyze_structure(
    df: pd.DataFrame,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """AI로 표의 전체 구조를 분석한다.

    Returns:
        구조 분석 결과 딕셔너리. API 실패 시 None.
    """
    key = api_key or _load_api_key()
    if not key or not GROQ_AVAILABLE:
        return None

    # 표 데이터를 텍스트로 변환 (처음 15행 + 마지막 5행)
    lines = []
    total_rows = len(df)
    show_rows = list(range(min(15, total_rows)))
    if total_rows > 15:
        show_rows += list(range(max(15, total_rows - 5), total_rows))

    for i in show_rows:
        vals = [str(df.iloc[i, c]) if pd.notna(df.iloc[i, c]) else "" for c in range(len(df.columns))]
        lines.append(f"행{i}: {' | '.join(vals)}")

    header_line = f"헤더: {' | '.join(str(c) for c in df.columns)}"
    table_text = f"{header_line}\n총 {total_rows}행, {len(df.columns)}열\n\n" + "\n".join(lines)

    prompt = f"다음 OCR 추출 표를 분석해주세요.\n\n{table_text}"

    client = Groq(api_key=key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": STRUCTURE_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2048,
        )
        result_text = response.choices[0].message.content or ""

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result_text)
        raw = json_match.group(1).strip() if json_match else result_text.strip()
        data = json.loads(raw)

        if isinstance(data, dict) and "doc_type" in data:
            return data

    except (json.JSONDecodeError, Exception):
        pass

    return None
