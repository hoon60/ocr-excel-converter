"""결과 캐싱 — 동일 이미지의 OCR 결과를 로컬 DB에 저장하여 API 재호출을 방지한다.

SHA-256 정확 매칭 + pHash 유사 매칭 지원.
"""

import json
from datetime import datetime, timezone

import cv2
import numpy as np

from .pattern_db import _conn, compute_file_hash

# ── DB 스키마 ──

CACHE_DDL = """
CREATE TABLE IF NOT EXISTS result_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash       TEXT NOT NULL UNIQUE,
    perceptual_hash TEXT,
    doc_type        TEXT NOT NULL DEFAULT 'unknown',
    result_json     TEXT NOT NULL,
    verified        INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_used_at    TEXT,
    use_count       INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_cache_phash ON result_cache(perceptual_hash);
"""


def _ensure_table():
    with _conn() as con:
        con.executescript(CACHE_DDL)


# ── Perceptual Hash (DCT 기반) ──

def compute_perceptual_hash(image_path: str) -> str:
    """이미지의 64비트 DCT 기반 perceptual hash를 계산한다."""
    from .ocr_extractor import _imread_unicode
    img = _imread_unicode(image_path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    return "".join("1" if b else "0" for b in bits)


def _hamming_distance(h1: str, h2: str) -> int:
    if len(h1) != len(h2):
        return 64
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ── 캐시 조회/저장 ──

def lookup_cache(file_hash: str) -> dict | None:
    """SHA-256 정확 매칭으로 캐시된 결과를 반환한다."""
    _ensure_table()
    with _conn() as con:
        row = con.execute(
            "SELECT result_json, doc_type, verified FROM result_cache WHERE file_hash=?",
            (file_hash,),
        ).fetchone()
        if row:
            ts = datetime.now(timezone.utc).isoformat()
            con.execute(
                "UPDATE result_cache SET use_count=use_count+1, last_used_at=? WHERE file_hash=?",
                (ts, file_hash),
            )
            return {
                "result": json.loads(row["result_json"]),
                "doc_type": row["doc_type"],
                "verified": bool(row["verified"]),
            }
    return None


def lookup_similar(phash: str, max_hamming: int = 5) -> dict | None:
    """perceptual hash 유사 매칭 (verified 엔트리만)."""
    if not phash:
        return None
    _ensure_table()
    with _conn() as con:
        rows = con.execute(
            "SELECT file_hash, perceptual_hash, result_json, doc_type FROM result_cache WHERE verified=1"
        ).fetchall()
    best = None
    best_dist = max_hamming + 1
    for r in rows:
        dist = _hamming_distance(phash, r["perceptual_hash"] or "")
        if dist < best_dist:
            best_dist = dist
            best = r
    if best:
        return {
            "result": json.loads(best["result_json"]),
            "doc_type": best["doc_type"],
            "verified": True,
        }
    return None


def store_result(
    file_hash: str,
    phash: str,
    doc_type: str,
    result_json: dict | list,
    verified: bool = False,
) -> None:
    """OCR 결과를 캐시에 저장한다."""
    _ensure_table()
    ts = datetime.now(timezone.utc).isoformat()
    val = json.dumps(result_json, ensure_ascii=False, default=str)
    with _conn() as con:
        con.execute(
            """INSERT INTO result_cache
               (file_hash, perceptual_hash, doc_type, result_json, verified, created_at, use_count)
               VALUES (?,?,?,?,?,?,1)
               ON CONFLICT(file_hash)
               DO UPDATE SET result_json=excluded.result_json,
                             doc_type=excluded.doc_type,
                             verified=MAX(verified, excluded.verified),
                             last_used_at=excluded.created_at,
                             use_count=use_count+1""",
            (file_hash, phash, doc_type, val, int(verified), ts),
        )


def mark_verified(file_hash: str) -> None:
    """캐시 엔트리를 verified 상태로 승격한다."""
    _ensure_table()
    with _conn() as con:
        con.execute(
            "UPDATE result_cache SET verified=1 WHERE file_hash=?",
            (file_hash,),
        )


def cache_to_dataframes(cached: dict) -> tuple[list, dict]:
    """캐시된 JSON 결과를 (list[DataFrame], metadata) 로 변환한다."""
    import pandas as pd
    data = cached["result"]
    if isinstance(data, dict):
        # 단일 테이블: {metadata, headers, rows}
        metadata = data.get("metadata", {})
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if headers else None)
            return [df], metadata
        return [], metadata
    elif isinstance(data, list):
        # 다중 테이블
        dfs = []
        metadata = {}
        for item in data:
            if isinstance(item, dict):
                h = item.get("headers", [])
                r = item.get("rows", [])
                if not metadata:
                    metadata = item.get("metadata", {})
                if h and r:
                    dfs.append(pd.DataFrame(r, columns=h[:len(r[0])] if h else None))
        return dfs, metadata
    return [], {}
