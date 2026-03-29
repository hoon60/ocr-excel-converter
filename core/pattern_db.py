"""패턴 DB — 모든 처리 이력과 학습 데이터를 SQLite에 저장한다.

처리 건수가 쌓일수록 분류 정확도가 올라가는 피드백 루프의 핵심 저장소.
"""

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from typing import Generator


# ── DB 경로 ──

def get_db_path() -> str:
    override_dir = os.environ.get("OCR_DATA_DIR", "").strip()
    if override_dir:
        os.makedirs(override_dir, exist_ok=True)
        return os.path.join(override_dir, "patterns.db")
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "patterns.db")


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    db = get_db_path()
    con = sqlite3.connect(db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── 스키마 초기화 ──

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    processed_at    TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_hash       TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    confidence      REAL DEFAULT 0.0,
    engine_used     TEXT,
    num_tables      INTEGER DEFAULT 0,
    num_rows        INTEGER DEFAULT 0,
    num_cols        INTEGER DEFAULT 0,
    confirmed_type  TEXT
);

CREATE TABLE IF NOT EXISTS column_templates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type        TEXT NOT NULL,
    column_name     TEXT NOT NULL,
    column_position INTEGER,
    frequency       INTEGER DEFAULT 1,
    last_seen_at    TEXT NOT NULL,
    UNIQUE(doc_type, column_name)
);

CREATE TABLE IF NOT EXISTS keyword_fingerprints (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type         TEXT NOT NULL,
    keyword          TEXT NOT NULL,
    weight           REAL DEFAULT 1.0,
    occurrence_count INTEGER DEFAULT 1,
    UNIQUE(doc_type, keyword)
);

CREATE TABLE IF NOT EXISTS ocr_corrections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type         TEXT NOT NULL,
    wrong_text       TEXT NOT NULL,
    correct_text     TEXT NOT NULL,
    correction_type  TEXT NOT NULL,
    confidence       REAL DEFAULT 1.0,
    occurrence_count INTEGER DEFAULT 1,
    UNIQUE(doc_type, wrong_text, correct_text)
);

CREATE TABLE IF NOT EXISTS metadata_patterns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type      TEXT NOT NULL,
    pattern_key   TEXT NOT NULL,
    pattern_value TEXT NOT NULL,
    frequency     INTEGER DEFAULT 1,
    UNIQUE(doc_type, pattern_key)
);

CREATE TABLE IF NOT EXISTS header_patterns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type     TEXT NOT NULL,
    header_json  TEXT NOT NULL,
    row_count    INTEGER,
    frequency    INTEGER DEFAULT 1,
    last_seen_at TEXT NOT NULL,
    UNIQUE(doc_type, header_json)
);

CREATE TABLE IF NOT EXISTS column_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type     TEXT NOT NULL,
    column_name  TEXT NOT NULL,
    col_min      REAL,
    col_max      REAL,
    col_mean     REAL,
    sample_count INTEGER DEFAULT 0,
    UNIQUE(doc_type, column_name)
);
"""

# 초기 시드 데이터 (기존 코드에 하드코딩된 지식을 DB로 이식)
_SEED_KEYWORDS = {
    "demand_forecast": [
        ("수요예측", 3.0), ("참여금액", 3.0), ("모집밴드", 2.5), ("총참여금액", 3.0),
        ("최종참여일시", 2.5), ("회사명", 2.0), ("부서명", 2.0), ("계정", 1.5),
        ("발행정보", 1.5), ("시장정보", 1.5), ("개별민평", 2.0), ("누적합계", 2.5),
        ("누적비중", 2.0), ("전일기준", 1.5), ("스프레드", 1.5), ("모집금액", 1.5),
        ("사채만기", 2.0), ("수요예측결과", 3.0),
    ],
    "financial_statement": [
        ("유동자산", 3.0), ("비유동자산", 3.0), ("자산총계", 3.0), ("부채총계", 3.0),
        ("자본총계", 3.0), ("영업이익", 2.5), ("매출액", 2.0), ("당기순이익", 2.5),
        ("유동부채", 2.5), ("비유동부채", 2.5), ("이익잉여금", 2.0), ("자본금", 2.0),
        ("재무상태표", 3.0), ("손익계산서", 3.0), ("현금흐름표", 3.0),
        ("주당순이익", 2.0), ("요약재무", 2.5),
    ],
}

_SEED_CORRECTIONS = {
    "demand_forecast": [
        # 흥 ↔ 홍 혼동
        ("홍국증권", "흥국증권", "company"),
        ("홍국자산운용", "흥국자산운용", "company"),
        # 이름 오인식
        ("김정환", "김정완", "name"),
        # 날짜 붙여쓰기
        ("2026-03-1615:", "2026-03-16 15:", "date"),
    ],
    "financial_statement": [
        ("유동자신", "유동자산", "term"),
        ("자산총겨", "자산총계", "term"),
    ],
    "default": [],
}


def initialize_db() -> None:
    """DB 초기화 (최초 1회). 이미 있으면 no-op."""
    with _conn() as con:
        con.executescript(_DDL)

    _seed_initial_data()


def _seed_initial_data() -> None:
    """기존 하드코딩 지식을 DB에 초기 시드로 삽입 (이미 있으면 스킵)."""
    with _conn() as con:
        for doc_type, kws in _SEED_KEYWORDS.items():
            for kw, weight in kws:
                con.execute(
                    """INSERT OR IGNORE INTO keyword_fingerprints
                       (doc_type, keyword, weight, occurrence_count)
                       VALUES (?,?,?,1)""",
                    (doc_type, kw, weight),
                )

        for doc_type, corrections in _SEED_CORRECTIONS.items():
            for wrong, correct, ctype in corrections:
                con.execute(
                    """INSERT OR IGNORE INTO ocr_corrections
                       (doc_type, wrong_text, correct_text, correction_type, confidence, occurrence_count)
                       VALUES (?,?,?,?,1.0,1)""",
                    (doc_type, wrong, correct, ctype),
                )


# ── Documents ──

def compute_file_hash(file_path: str) -> str:
    h = sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def insert_document(
    file_name: str,
    file_hash: str,
    doc_type: str,
    confidence: float,
    engine_used: str,
    num_tables: int,
    num_rows: int,
    num_cols: int,
) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO documents
               (processed_at, file_name, file_hash, doc_type, confidence,
                engine_used, num_tables, num_rows, num_cols)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts, file_name, file_hash, doc_type, confidence,
             engine_used, num_tables, num_rows, num_cols),
        )
        return cur.lastrowid


def confirm_document_type(doc_id: int, confirmed_type: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE documents SET confirmed_type=? WHERE id=?",
            (confirmed_type, doc_id),
        )


def get_confirmed_documents() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT id, file_name, doc_type, confirmed_type, confidence,
                      engine_used, num_rows, num_cols, processed_at
               FROM documents WHERE confirmed_type IS NOT NULL
               ORDER BY processed_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_documents(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT id, file_name, doc_type, confidence, engine_used,
                      num_rows, num_cols, processed_at, confirmed_type
               FROM documents ORDER BY processed_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """처리 통계 요약."""
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        confirmed = con.execute(
            "SELECT COUNT(*) FROM documents WHERE confirmed_type IS NOT NULL"
        ).fetchone()[0]
        by_type = con.execute(
            "SELECT doc_type, COUNT(*) as cnt FROM documents GROUP BY doc_type"
        ).fetchall()
        corrections = con.execute(
            "SELECT SUM(occurrence_count) FROM ocr_corrections"
        ).fetchone()[0] or 0
    return {
        "total_documents": total,
        "confirmed_documents": confirmed,
        "by_type": {r["doc_type"]: r["cnt"] for r in by_type},
        "learned_corrections": corrections,
    }


# ── Column Templates ──

def upsert_column_template(doc_type: str, column_name: str, position: int) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            """INSERT INTO column_templates (doc_type, column_name, column_position, frequency, last_seen_at)
               VALUES (?,?,?,1,?)
               ON CONFLICT(doc_type, column_name)
               DO UPDATE SET frequency=frequency+1, last_seen_at=excluded.last_seen_at,
                             column_position=(column_position+excluded.column_position)/2""",
            (doc_type, column_name, position, ts),
        )


def get_column_template(doc_type: str, min_frequency: int = 1) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT column_name, column_position, frequency
               FROM column_templates WHERE doc_type=? AND frequency>=?
               ORDER BY column_position, frequency DESC""",
            (doc_type, min_frequency),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Keyword Fingerprints ──

def upsert_keyword(doc_type: str, keyword: str, weight: float = 1.0) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO keyword_fingerprints (doc_type, keyword, weight, occurrence_count)
               VALUES (?,?,?,1)
               ON CONFLICT(doc_type, keyword)
               DO UPDATE SET occurrence_count=occurrence_count+1,
                             weight=MAX(weight, excluded.weight)""",
            (doc_type, keyword, weight),
        )


def get_all_keywords_by_type() -> dict[str, list[tuple[str, float]]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT doc_type, keyword, weight FROM keyword_fingerprints ORDER BY weight DESC"
        ).fetchall()
    result: dict[str, list] = {}
    for r in rows:
        result.setdefault(r["doc_type"], []).append((r["keyword"], r["weight"]))
    return result


# ── OCR Corrections ──

def upsert_correction(
    doc_type: str,
    wrong_text: str,
    correct_text: str,
    correction_type: str,
    confidence: float = 1.0,
) -> None:
    with _conn() as con:
        con.execute(
            """INSERT INTO ocr_corrections
               (doc_type, wrong_text, correct_text, correction_type, confidence, occurrence_count)
               VALUES (?,?,?,?,?,1)
               ON CONFLICT(doc_type, wrong_text, correct_text)
               DO UPDATE SET occurrence_count=occurrence_count+1,
                             confidence=MAX(confidence, excluded.confidence)""",
            (doc_type, wrong_text, correct_text, correction_type, confidence),
        )


def get_corrections(
    doc_type: str,
    correction_type: str | None = None,
    min_confidence: float = 0.5,
    exclude_types: list[str] | None = None,
) -> list[tuple[str, str]]:
    with _conn() as con:
        if correction_type:
            rows = con.execute(
                """SELECT wrong_text, correct_text FROM ocr_corrections
                   WHERE doc_type=? AND correction_type=? AND confidence>=?
                   ORDER BY occurrence_count DESC""",
                (doc_type, correction_type, min_confidence),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT wrong_text, correct_text FROM ocr_corrections
                   WHERE doc_type=? AND confidence>=?
                   ORDER BY occurrence_count DESC""",
                (doc_type, min_confidence),
            ).fetchall()
    result = [(r["wrong_text"], r["correct_text"]) for r in rows]
    if exclude_types:
        # exclude_types 필터는 메모리에서 수행 (타입 컬럼 재조회)
        with _conn() as con:
            typed_rows = con.execute(
                """SELECT wrong_text, correct_text, correction_type FROM ocr_corrections
                   WHERE doc_type=? AND confidence>=?""",
                (doc_type, min_confidence),
            ).fetchall()
        excluded = {(r["wrong_text"], r["correct_text"]) for r in typed_rows
                    if r["correction_type"] in exclude_types}
        result = [(w, c) for w, c in result if (w, c) not in excluded]
    return result


# ── Metadata Patterns ──

def upsert_metadata_pattern(
    doc_type: str,
    pattern_key: str,
    pattern_value: list | dict,
) -> None:
    val = json.dumps(pattern_value, ensure_ascii=False)
    with _conn() as con:
        con.execute(
            """INSERT INTO metadata_patterns (doc_type, pattern_key, pattern_value, frequency)
               VALUES (?,?,?,1)
               ON CONFLICT(doc_type, pattern_key)
               DO UPDATE SET frequency=frequency+1, pattern_value=excluded.pattern_value""",
            (doc_type, pattern_key, val),
        )


def get_metadata_patterns(doc_type: str) -> dict:
    with _conn() as con:
        rows = con.execute(
            "SELECT pattern_key, pattern_value FROM metadata_patterns WHERE doc_type=?",
            (doc_type,),
        ).fetchall()
    return {r["pattern_key"]: json.loads(r["pattern_value"]) for r in rows}


# ── Header Patterns ──

def insert_header_pattern(
    doc_type: str,
    headers: list[str],
    row_count: int,
) -> None:
    key = json.dumps(headers, ensure_ascii=False)
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute(
            """INSERT INTO header_patterns (doc_type, header_json, row_count, frequency, last_seen_at)
               VALUES (?,?,?,1,?)
               ON CONFLICT(doc_type, header_json)
               DO UPDATE SET frequency=frequency+1, last_seen_at=excluded.last_seen_at""",
            (doc_type, key, row_count, ts),
        )


def get_top_header_patterns(doc_type: str, top_n: int = 3) -> list[list[str]]:
    with _conn() as con:
        rows = con.execute(
            """SELECT header_json FROM header_patterns WHERE doc_type=?
               ORDER BY frequency DESC LIMIT ?""",
            (doc_type, top_n),
        ).fetchall()
    return [json.loads(r["header_json"]) for r in rows]


# ── Column Stats (Welford 온라인 알고리즘) ──

def upsert_column_stats(doc_type: str, column_name: str, values: list[float]) -> None:
    if not values:
        return
    with _conn() as con:
        row = con.execute(
            "SELECT col_min, col_max, col_mean, sample_count FROM column_stats WHERE doc_type=? AND column_name=?",
            (doc_type, column_name),
        ).fetchone()

        new_vals = list(values)
        if row:
            prev_min = row["col_min"]
            prev_max = row["col_max"]
            prev_mean = row["col_mean"]
            prev_n = row["sample_count"]
            new_n = prev_n + len(new_vals)
            new_mean = (prev_mean * prev_n + sum(new_vals)) / new_n
            new_min = min(prev_min, min(new_vals))
            new_max = max(prev_max, max(new_vals))
            con.execute(
                """UPDATE column_stats SET col_min=?, col_max=?, col_mean=?, sample_count=?
                   WHERE doc_type=? AND column_name=?""",
                (new_min, new_max, new_mean, new_n, doc_type, column_name),
            )
        else:
            n = len(new_vals)
            con.execute(
                """INSERT INTO column_stats (doc_type, column_name, col_min, col_max, col_mean, sample_count)
                   VALUES (?,?,?,?,?,?)""",
                (doc_type, column_name, min(new_vals), max(new_vals), sum(new_vals)/n, n),
            )


def get_column_stats(doc_type: str) -> dict[str, dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT column_name, col_min, col_max, col_mean, sample_count FROM column_stats WHERE doc_type=?",
            (doc_type,),
        ).fetchall()
    return {
        r["column_name"]: {
            "min": r["col_min"], "max": r["col_max"],
            "mean": r["col_mean"], "count": r["sample_count"],
        }
        for r in rows
    }
