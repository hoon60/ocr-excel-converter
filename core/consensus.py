"""다중 실행 합의 투표 — Groq OCR의 확률적 오류를 다수결로 제거한다.

동일 이미지를 N회 병렬 호출하여 셀별 다수결을 적용.
숫자 열은 이미 정확하므로 텍스트 열(회사명/부서명/이름/계정)에 집중.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd


def consensus_extract(
    image_path: str,
    api_key: str,
    num_runs: int = 3,
    progress_cb: Callable[[int], None] | None = None,
    prompt_override: str | None = None,
) -> tuple[list[pd.DataFrame], dict] | None:
    """Groq OCR을 num_runs회 병렬 실행, 셀별 다수결로 합의 결과 반환."""
    from .vision_ocr import extract_with_groq

    results = []

    def _run(_idx):
        return extract_with_groq(image_path, api_key, prompt_override=prompt_override)

    with ThreadPoolExecutor(max_workers=min(num_runs, 5)) as pool:
        futures = {pool.submit(_run, i): i for i in range(num_runs)}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    if progress_cb:
        progress_cb(80)

    if not results:
        return None
    if len(results) == 1:
        return results[0]

    # 모든 결과에서 DataFrame + metadata 분리
    all_dfs = [r[0][0] if r[0] else pd.DataFrame() for r in results]
    all_metas = [r[1] for r in results]

    # 빈 DataFrame 필터링
    all_dfs = [df for df in all_dfs if not df.empty]
    if not all_dfs:
        return results[0]

    # 행 수 합의 (다수결)
    row_counts = Counter(len(df) for df in all_dfs)
    target_rows = row_counts.most_common(1)[0][0]
    valid_dfs = [df for df in all_dfs if len(df) == target_rows]
    if not valid_dfs:
        valid_dfs = all_dfs

    # 열 수 합의
    col_counts = Counter(len(df.columns) for df in valid_dfs)
    target_cols = col_counts.most_common(1)[0][0]
    valid_dfs = [df for df in valid_dfs if len(df.columns) == target_cols]
    if not valid_dfs:
        valid_dfs = [all_dfs[0]]

    # 헤더 합의
    headers = _vote_headers(valid_dfs)

    # 셀별 합의
    merged_rows = []
    for row_idx in range(target_rows):
        row = []
        for col_idx in range(target_cols):
            candidates = []
            for df in valid_dfs:
                if row_idx < len(df) and col_idx < len(df.columns):
                    candidates.append(str(df.iloc[row_idx, col_idx]).strip())
            voted = _vote_cell(candidates, col_idx, target_cols)
            row.append(voted)
        merged_rows.append(row)

    merged_df = pd.DataFrame(merged_rows, columns=headers)

    # 메타데이터는 첫 번째 것 사용
    meta = all_metas[0] if all_metas else {}

    if progress_cb:
        progress_cb(100)

    return ([merged_df], meta)


def _vote_headers(dfs: list[pd.DataFrame]) -> list[str]:
    """열 헤더의 다수결."""
    if not dfs:
        return []
    n_cols = len(dfs[0].columns)
    headers = []
    for ci in range(n_cols):
        candidates = [str(df.columns[ci]).strip() for df in dfs if ci < len(df.columns)]
        cnt = Counter(candidates)
        headers.append(cnt.most_common(1)[0][0])
    return headers


def _vote_cell(candidates: list[str], col_idx: int, total_cols: int) -> str:
    """셀 값의 다수결. 텍스트 열은 문자열 단위, 숫자는 정확 매칭."""
    if not candidates:
        return ""

    # 과반수 정확 매칭
    cnt = Counter(candidates)
    most_common, freq = cnt.most_common(1)[0]
    if freq > len(candidates) // 2:
        return most_common

    # 텍스트 열 (왼쪽 6열): 자모 레벨 투표는 위험 → 문자열 다수결 우선
    if col_idx < 6:
        return _vote_text(candidates)

    # 숫자 열: 가장 많이 나온 값
    return most_common


def _vote_text(candidates: list[str]) -> str:
    """텍스트 셀의 다수결. 길이가 같은 후보끼리 문자별 투표."""
    if not candidates:
        return ""

    # 먼저 정확 매칭 시도
    cnt = Counter(candidates)
    most_common, freq = cnt.most_common(1)[0]
    if freq >= 2:
        return most_common

    # 길이별 그룹
    by_len = {}
    for c in candidates:
        by_len.setdefault(len(c), []).append(c)
    # 가장 많은 길이 그룹
    target_len = max(by_len, key=lambda k: len(by_len[k]))
    same_len = by_len[target_len]

    if len(same_len) == 1:
        return same_len[0]

    # 문자별 투표
    result = []
    for i in range(target_len):
        chars = [s[i] for s in same_len if i < len(s)]
        char_cnt = Counter(chars)
        result.append(char_cnt.most_common(1)[0][0])
    return "".join(result)
