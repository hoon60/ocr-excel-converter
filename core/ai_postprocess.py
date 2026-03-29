"""Groq API를 활용한 OCR 결과 AI 보정 모듈.

OCR이 추출한 표 데이터를 Llama 모델로 교정한다:
- 숫자 오인식 보정 (O→0, l→1, S→5 등)
- 깨진 한글 복원
- 금액/날짜/백분율 포맷 정규화
"""

import json
import os
import re
from typing import Callable

import pandas as pd

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"


def _load_api_key() -> str:
    """config.json 또는 환경변수에서 API 키를 로드한다."""
    # 1) 환경변수 우선
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # 2) config.json 폴백
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("groq_api_key", "")
        except (json.JSONDecodeError, OSError):
            pass
    return ""


GROQ_API_KEY = _load_api_key()

SYSTEM_PROMPT = """\
당신은 한국 채권시장 문서 OCR 후처리 전문가입니다. 스캔된 한국어 금융 문서에서 OCR로 추출한 표 데이터를 교정합니다.

## 교정 규칙

### 1. 한국 금융기관명 교정 (최우선)
OCR이 자주 틀리는 한국 증권사/운용사 이름을 올바르게 교정:
- 주요 증권사: KB증권, NH투자증권, 삼성증권, 미래에셋증권, 한국투자증권, 메리츠증권, 신한투자증권, 대신증권, 하나증권, 키움증권, SK증권, DB증권, 한화투자증권, 교보증권, 유안타증권, 유진투자증권, 현대차증권, BNK투자증권, IBK투자증권, 이베스트투자증권, 부국증권, 케이프투자증권
- 주요 운용사: 에이펙스자산운용, 코레이트자산운용, 다올자산운용, 미래에셋자산운용, 삼성자산운용, KB자산운용, 한화자산운용, 현대인베스트먼트자산운용, 교보악사자산운용, 키움투자자산운용
- 부서명: 채권운용부, 채권운용본부, 채권상품부, 채권영업부, 발행어음운용부, 종합금융운용팀, 리테일채권팀, 신디케이션부, 주식운용부문, 상품솔루션부, 채권전략팀, 채권운용1팀, FICC
- 계정: 고유, 집합, 일임

### 2. 한국인 이름 교정
- 3글자 한국 이름이 깨진 경우 문맥상 교정 (예: "우운"→"송혜인"은 불가, 하지만 "B|Ylo"→"이지영", "해"→"황대우" 등 명백한 오류는 교정)
- 한글이 아닌 문자(영문/특수문자)가 섞인 이름은 가장 가까운 한글 이름으로 교정

### 3. 숫자 오인식
O→0, l/I→1, S→5, B→8, Z→2, G→6

### 4. 재무제표 용어
유등자산→유동자산, 금용→금융, 총겨→총계, 현금묘현금섬자산→현금및현금성자산

### 5. 날짜/시간 포맷
"2026-03-1615:52:13" → "2026-03-16 15:52:13" (날짜와 시간 사이 공백)

### 6. 명백한 오타만 교정
확실하지 않으면 원본 유지. 숫자 데이터는 절대 변경하지 마세요.

## 출력 형식
반드시 JSON 배열로 반환하세요. 각 행은 배열입니다.
다른 텍스트 없이 JSON만 출력하세요.
"""


def _is_available() -> bool:
    """Groq API 사용 가능 여부를 확인한다."""
    return GROQ_AVAILABLE and bool(GROQ_API_KEY)


def _chunk_dataframe(df: pd.DataFrame, max_rows: int = 30) -> list[pd.DataFrame]:
    """큰 테이블을 청크로 분할한다 (API 토큰 제한 대응)."""
    if len(df) <= max_rows:
        return [df]
    return [df.iloc[i : i + max_rows] for i in range(0, len(df), max_rows)]


def _df_to_text(df: pd.DataFrame) -> str:
    """DataFrame을 AI에 보낼 텍스트로 변환한다."""
    header = " | ".join(str(c) for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(v) for v in row))
    return f"[헤더] {header}\n" + "\n".join(f"[행{i+1}] {r}" for i, r in enumerate(rows))


def _parse_response(response_text: str, df: pd.DataFrame) -> pd.DataFrame:
    """AI 응답(JSON)을 DataFrame으로 파싱한다. 실패 시 원본 반환."""
    try:
        # JSON 블록 추출 (```json ... ``` 또는 순수 JSON)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            raw = json_match.group(1).strip()
        else:
            raw = response_text.strip()

        data = json.loads(raw)

        if not isinstance(data, list) or len(data) == 0:
            return df

        # 행 수가 맞으면 교정 적용
        if len(data) == len(df):
            corrected = pd.DataFrame(data, columns=df.columns)
            return corrected

        return df
    except (json.JSONDecodeError, ValueError, KeyError):
        return df


def correct_with_ai(
    tables: list[pd.DataFrame],
    api_key: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
) -> list[pd.DataFrame]:
    """Groq API로 OCR 추출 결과를 교정한다.

    Args:
        tables: OCR로 추출된 DataFrame 리스트
        api_key: Groq API 키 (None이면 환경변수 사용)
        progress_cb: 진행률 콜백

    Returns:
        교정된 DataFrame 리스트 (API 실패 시 원본 반환)
    """
    key = api_key or GROQ_API_KEY
    if not key or not GROQ_AVAILABLE:
        return tables

    client = Groq(api_key=key)
    corrected_tables: list[pd.DataFrame] = []
    total = len(tables)

    for t_idx, df in enumerate(tables):
        if df.empty:
            corrected_tables.append(df)
            if progress_cb:
                progress_cb(int((t_idx + 1) / total * 100))
            continue

        chunks = _chunk_dataframe(df)
        corrected_chunks: list[pd.DataFrame] = []

        for chunk in chunks:
            table_text = _df_to_text(chunk)
            prompt = (
                f"다음은 OCR로 추출한 표 데이터입니다. 오인식된 부분을 교정해주세요.\n\n"
                f"{table_text}\n\n"
                f"교정된 데이터를 {len(chunk)}행의 JSON 2차원 배열로 반환하세요. "
                f"열 수는 {len(chunk.columns)}개입니다.\n\n"
                f"주의: 숫자 데이터는 절대 변경하지 마세요. "
                f"한글이 완전히 깨져서 원래 단어를 추정할 수 없으면 원본 그대로 유지하세요. "
                f"확실한 교정만 수행하세요."
            )

            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_completion_tokens=4096,
                )
                result_text = response.choices[0].message.content or ""
                corrected_chunk = _parse_response(result_text, chunk)
            except Exception:
                # API 오류 시 원본 유지
                corrected_chunk = chunk

            corrected_chunks.append(corrected_chunk)

        # 청크 재결합
        if len(corrected_chunks) == 1:
            corrected_tables.append(corrected_chunks[0])
        else:
            combined = pd.concat(corrected_chunks, ignore_index=True)
            corrected_tables.append(combined)

        if progress_cb:
            progress_cb(int((t_idx + 1) / total * 100))

    return corrected_tables
