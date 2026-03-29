"""엔티티 사전 — 한국 금융기관/부서/계정 사전 기반 자모 퍼지 매칭.

OCR 결과의 회사명/부서명을 알려진 정식 명칭으로 교정한다.
자모(초성/중성/종성) 레벨 편집거리로 하/아, 완/환 등 유사 혼동을 정밀 처리.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .pattern_db import _conn

# ── DB 스키마 ──

ENTITY_DDL = """
CREATE TABLE IF NOT EXISTS known_entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,
    canonical_name  TEXT NOT NULL,
    frequency       INTEGER DEFAULT 1,
    UNIQUE(entity_type, canonical_name)
);
CREATE TABLE IF NOT EXISTS entity_variants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       INTEGER REFERENCES known_entities(id),
    variant_text    TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    UNIQUE(entity_id, variant_text)
);
"""

_initialized = False

def _ensure_tables():
    global _initialized
    if _initialized:
        return
    with _conn() as con:
        con.executescript(ENTITY_DDL)
    _initialized = True


# ── 한글 자모 분해 ──

# 초성 19개, 중성 21개, 종성 28개 (0=없음)
_CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3


def _decompose_korean(char: str) -> list[str] | None:
    """한글 한 글자를 자모 시퀀스로 분해. 비한글이면 None."""
    code = ord(char)
    if _HANGUL_BASE <= code <= _HANGUL_END:
        offset = code - _HANGUL_BASE
        cho = offset // 588
        jung = (offset % 588) // 28
        jong = offset % 28
        result = [_CHOSEONG[cho], _JUNGSEONG[jung]]
        if jong > 0:
            result.append(_JONGSEONG[jong])
        return result
    return None


def _to_jamo_sequence(text: str) -> list[str]:
    """문자열을 자모 시퀀스로 변환. 비한글은 그대로 유지."""
    seq = []
    for ch in text:
        decomposed = _decompose_korean(ch)
        if decomposed:
            seq.extend(decomposed)
        else:
            seq.append(ch)
    return seq


def levenshtein_jamo(s1: str, s2: str) -> int:
    """자모 레벨 편집거리. 하(ㅎ+ㅏ) vs 아(ㅇ+ㅏ) = 거리 1."""
    seq1 = _to_jamo_sequence(s1)
    seq2 = _to_jamo_sequence(s2)
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


# ── 엔티티 조회/학습 ──

def get_all_entities(entity_type: str) -> list[tuple[int, str]]:
    """주어진 타입의 모든 엔티티를 (id, canonical_name) 리스트로 반환."""
    _ensure_tables()
    with _conn() as con:
        rows = con.execute(
            "SELECT id, canonical_name FROM known_entities WHERE entity_type=? ORDER BY frequency DESC",
            (entity_type,),
        ).fetchall()
    return [(r["id"], r["canonical_name"]) for r in rows]


def fuzzy_match_entity(
    text: str,
    entity_type: str,
    max_jamo_dist: int = 3,
) -> str | None:
    """자모 편집거리 기반 퍼지 매칭. 매칭 성공 시 canonical_name 반환."""
    if not text or len(text) < 2:
        return None
    entities = get_all_entities(entity_type)
    if not entities:
        return None

    # 공백 정규화 매칭 (부서명에서 공백 누락/추가 처리)
    text_no_space = text.replace(" ", "")
    for _eid, canonical in entities:
        if text_no_space == canonical.replace(" ", ""):
            return canonical  # 공백만 다른 경우 → 정식 명칭 반환

    best_name = None
    best_dist = max_jamo_dist + 1

    for _eid, canonical in entities:
        # 길이 차이가 너무 크면 스킵
        if abs(len(text) - len(canonical)) > 3:
            continue
        dist = levenshtein_jamo(text, canonical)
        if dist < best_dist:
            best_dist = dist
            best_name = canonical
        if dist == 0:
            return canonical  # 정확 매칭

    if best_name and best_dist <= max_jamo_dist:
        return best_name
    return None


def exact_variant_lookup(text: str, entity_type: str) -> str | None:
    """entity_variants에서 정확 매칭 조회."""
    _ensure_tables()
    with _conn() as con:
        row = con.execute(
            """SELECT ke.canonical_name FROM entity_variants ev
               JOIN known_entities ke ON ev.entity_id = ke.id
               WHERE ev.variant_text=? AND ke.entity_type=?""",
            (text, entity_type),
        ).fetchone()
    return row["canonical_name"] if row else None


def correct_text_column(
    series: pd.Series,
    entity_type: str,
    max_jamo_dist: int = 3,
) -> pd.Series:
    """Series의 각 값을 엔티티 사전으로 교정한다."""
    # 알려진 엔티티 이름 set (정확 매칭 시 교정 스킵용)
    known_names = {name for _, name in get_all_entities(entity_type)}

    result = series.copy()
    for idx, val in series.items():
        text = str(val).strip()
        if not text or len(text) < 2:
            continue
        # 0) 이미 알려진 정식 엔티티면 교정하지 않음
        if text in known_names:
            continue
        # 1) 변형 사전 정확 매칭
        canonical = exact_variant_lookup(text, entity_type)
        if canonical:
            result.at[idx] = canonical
            continue
        # 2) 자모 퍼지 매칭
        canonical = fuzzy_match_entity(text, entity_type, max_jamo_dist)
        if canonical and canonical != text:
            result.at[idx] = canonical
            # 변형으로 기록
            _record_variant(text, canonical, entity_type)
    return result


def _record_variant(variant: str, canonical: str, entity_type: str):
    """새 변형을 entity_variants에 기록."""
    _ensure_tables()
    with _conn() as con:
        row = con.execute(
            "SELECT id FROM known_entities WHERE entity_type=? AND canonical_name=?",
            (entity_type, canonical),
        ).fetchone()
        if row:
            con.execute(
                """INSERT INTO entity_variants (entity_id, variant_text, occurrence_count)
                   VALUES (?,?,1)
                   ON CONFLICT(entity_id, variant_text)
                   DO UPDATE SET occurrence_count=occurrence_count+1""",
                (row["id"], variant),
            )


def learn_entity(text: str, entity_type: str) -> None:
    """새 엔티티를 등록하거나 기존 frequency를 증가."""
    if not text or len(text) < 2:
        return
    _ensure_tables()
    with _conn() as con:
        con.execute(
            """INSERT INTO known_entities (entity_type, canonical_name, frequency)
               VALUES (?,?,1)
               ON CONFLICT(entity_type, canonical_name)
               DO UPDATE SET frequency=frequency+1""",
            (entity_type, text),
        )


def initialize_seed_entities() -> None:
    """한국 금융기관, 부서명, 계정유형 시드 데이터를 삽입한다."""
    _ensure_tables()

    # 증권사
    _SECURITIES = [
        "KB증권", "NH투자증권", "삼성증권", "미래에셋증권", "한국투자증권",
        "신한투자증권", "키움증권", "메리츠증권", "대신증권", "하나증권",
        "SK증권", "DB증권", "유진투자증권", "한양증권", "BNK투자증권",
        "교보증권", "부국증권", "현대차증권", "LS증권", "IBK투자증권",
        "흥국증권", "케이프투자증권", "다올투자증권", "iM증권",
        "한화투자증권", "유안타증권", "KTB투자증권",
    ]

    # 자산운용사
    _ASSET_MGMT = [
        "삼성자산운용", "미래에셋자산운용", "KB자산운용", "한국투자신탁운용",
        "신한자산운용", "한화자산운용", "키움투자자산운용", "NH아문디자산운용",
        "교보악사자산운용", "흥국자산운용", "하나자산운용", "다올자산운용",
        "웰컴자산운용", "에이펙스자산운용", "코레이트자산운용",
        "현대인베스트먼트자산운용", "동양자산운용", "BNK자산운용",
        "DB자산운용", "유리자산운용", "트러스톤자산운용",
        "마이다스에셋자산운용", "브이아이자산운용", "메리츠자산운용",
        "이스트스프링자산운용", "JB자산운용", "HDC자산운용",
    ]

    # 기타 금융기관
    _OTHERS = [
        "한국산업은행", "한국수출입은행", "중소기업은행", "농협은행",
        "수협은행", "신한은행", "국민은행", "하나은행", "우리은행",
        "한국예탁결제원", "한국거래소", "금융감독원",
        "메리츠화재", "삼성화재", "DB손해보험", "현대해상",
    ]

    # 부서명
    _DEPARTMENTS = [
        "채권운용본부", "채권운용팀", "채권전략팀", "채권상품부",
        "트레이딩팀", "종합금융팀", "발행어음운용부", "운용지원팀",
        "상품솔루션부", "종합금융운용팀", "채권운용1팀", "채권운용2팀",
        "신디케이션부", "채권영업부", "BTS팀", "FICC",
        "금융상품지원부", "채권본부", "투자전략팀",
        "리테일채권팀", "채권운용본부 투자전략팀",
        "주식운용부문",
    ]

    # 계정유형
    _ACCOUNTS = ["집합", "고유", "일임", "사모"]

    with _conn() as con:
        for name in _SECURITIES + _ASSET_MGMT + _OTHERS:
            con.execute(
                """INSERT OR IGNORE INTO known_entities
                   (entity_type, canonical_name, frequency) VALUES (?,?,1)""",
                ("company", name),
            )
        for name in _DEPARTMENTS:
            con.execute(
                """INSERT OR IGNORE INTO known_entities
                   (entity_type, canonical_name, frequency) VALUES (?,?,1)""",
                ("department", name),
            )
        for name in _ACCOUNTS:
            con.execute(
                """INSERT OR IGNORE INTO known_entities
                   (entity_type, canonical_name, frequency) VALUES (?,?,1)""",
                ("account", name),
            )
