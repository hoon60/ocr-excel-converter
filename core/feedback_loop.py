"""사용자 교정 피드백 루프 — Excel 수정을 감지하여 자동 학습.

사용자가 출력 Excel을 수정하면 → 원본과 비교 → 교정 패턴을 DB에 학습.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .pattern_db import _conn, upsert_correction

# ── DB 스키마 ──

TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS output_tracking (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    output_path         TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    doc_type            TEXT NOT NULL DEFAULT 'default',
    original_json       TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    corrections_extracted INTEGER DEFAULT 0
);
"""


def _ensure_table():
    with _conn() as con:
        con.executescript(TRACKING_DDL)


def register_output(
    output_path: str,
    source_hash: str,
    doc_type: str,
    tables: list[pd.DataFrame],
) -> None:
    """Excel 생성 직후 원본 DataFrame을 기록한다."""
    _ensure_table()
    ts = datetime.now(timezone.utc).isoformat()

    # DataFrame → JSON 직렬화
    data = []
    for df in tables:
        data.append({
            "columns": list(df.columns),
            "values": df.astype(str).values.tolist(),
        })
    val = json.dumps(data, ensure_ascii=False)

    with _conn() as con:
        # 동일 output_path가 있으면 업데이트
        con.execute(
            """INSERT INTO output_tracking
               (output_path, source_hash, doc_type, original_json, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(output_path) DO UPDATE SET
                   source_hash=excluded.source_hash,
                   doc_type=excluded.doc_type,
                   original_json=excluded.original_json,
                   created_at=excluded.created_at,
                   corrections_extracted=0""",
            (output_path, source_hash, doc_type, val, ts),
        )


# output_path UNIQUE 제약 추가
_UNIQUE_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_output_path ON output_tracking(output_path);
"""


def _ensure_unique_index():
    try:
        with _conn() as con:
            con.executescript(_UNIQUE_DDL)
    except Exception:
        pass  # 이미 있으면 무시


def check_for_corrections(output_path: str) -> list[dict] | None:
    """현재 Excel 파일 내용을 원본과 비교하여 사용자 교정을 감지한다."""
    _ensure_table()

    with _conn() as con:
        row = con.execute(
            "SELECT id, original_json, doc_type, corrections_extracted FROM output_tracking WHERE output_path=?",
            (output_path,),
        ).fetchone()

    if not row or row["corrections_extracted"]:
        return None

    if not os.path.isfile(output_path):
        return None

    # 원본 JSON → DataFrame 복원
    original_data = json.loads(row["original_json"])
    if not original_data:
        return None

    # 현재 Excel 읽기
    try:
        current_df = pd.read_excel(output_path, header=0, dtype=str).fillna("")
    except Exception:
        return None

    orig = original_data[0]  # 첫 번째 테이블
    orig_cols = orig["columns"]
    orig_vals = orig["values"]

    corrections = []
    n_rows = min(len(orig_vals), len(current_df))
    n_cols = min(len(orig_cols), len(current_df.columns))

    for r in range(n_rows):
        for c in range(n_cols):
            orig_val = str(orig_vals[r][c]).strip() if c < len(orig_vals[r]) else ""
            curr_val = str(current_df.iloc[r, c]).strip()

            if orig_val != curr_val and orig_val and curr_val:
                col_name = orig_cols[c] if c < len(orig_cols) else f"col_{c}"
                corrections.append({
                    "row": r,
                    "col": c,
                    "col_name": col_name,
                    "original": orig_val,
                    "corrected": curr_val,
                    "correction_type": _infer_type(col_name, orig_val, curr_val),
                })

    return corrections if corrections else None


def _infer_type(col_name: str, original: str, corrected: str) -> str:
    """교정의 타입을 추론한다."""
    cn = col_name.lower()
    if "회사" in cn or "기관" in cn:
        return "company"
    if "부서" in cn or "팀" in cn:
        return "department"
    if "이름" in cn or "성명" in cn:
        return "name"
    if "날짜" in cn or "일시" in cn or "참여일" in cn:
        return "date"
    if "금액" in cn or "합계" in cn:
        return "number"
    # 내용 기반 추론
    if original.replace(",", "").replace("-", "").isdigit():
        return "number"
    return "term"


def apply_learned_corrections(
    corrections: list[dict],
    doc_type: str,
) -> int:
    """감지된 교정을 pattern_db, entity_dict, result_cache에 반영한다.

    구조 오류 방지: 양쪽 다 유효한 엔티티면 학습하지 않음 (행 밀림 오류).
    """
    # 알려진 엔티티 로드 (구조 오류 판별용)
    known_entities = {}
    try:
        from .entity_dict import get_all_entities
        for etype in ("company", "department", "account"):
            known_entities[etype] = {name for _, name in get_all_entities(etype)}
    except Exception:
        pass

    count = 0
    for corr in corrections:
        wrong = corr["original"]
        correct = corr["corrected"]
        ctype = corr["correction_type"]

        # 구조 오류 필터: 양쪽 다 유효한 엔티티면 스킵
        if ctype in known_entities:
            known = known_entities[ctype]
            if wrong in known and correct in known:
                # 둘 다 알려진 엔티티 → 행 밀림, 학습하면 안 됨
                continue

        # 1) pattern_db에 교정 추가
        upsert_correction(
            doc_type=doc_type,
            wrong_text=wrong,
            correct_text=correct,
            correction_type=ctype,
            confidence=1.0,
        )

        # 2) entity_dict에 학습 + 변형(variant) 등록
        try:
            from .entity_dict import learn_entity, _record_variant
            if ctype == "company":
                learn_entity(correct, "company")
                _record_variant(wrong, correct, "company")
            elif ctype == "department":
                learn_entity(correct, "department")
                _record_variant(wrong, correct, "department")
        except Exception:
            pass

        count += 1

    # 3) 해당 이미지의 캐시 무효화 (다음 실행 시 교정이 반영된 새 결과 생성)
    if count > 0:
        _invalidate_related_cache(doc_type)

    return count


def _invalidate_related_cache(doc_type: str):
    """교정이 적용된 문서 유형의 캐시를 무효화한다."""
    try:
        with _conn() as con:
            # verified=0인 캐시만 삭제 (사용자 확인 전 캐시)
            con.execute(
                "DELETE FROM result_cache WHERE doc_type=? AND verified=0",
                (doc_type,),
            )
    except Exception:
        pass


def learn_from_corrected_excel(
    original_excel: str,
    corrected_excel: str,
    doc_type: str = "demand_forecast",
) -> int:
    """사용자가 수정한 Excel을 원본과 비교하여 교정 패턴을 학습한다.

    Args:
        original_excel: 프로그램이 생성한 원본 Excel 경로
        corrected_excel: 사용자가 수정한 Excel 경로
        doc_type: 문서 유형

    Returns:
        학습된 교정 건수
    """
    try:
        orig_df = pd.read_excel(original_excel, header=0, dtype=str).fillna("")
        corr_df = pd.read_excel(corrected_excel, header=0, dtype=str).fillna("")
    except Exception as e:
        print(f"[FeedbackLoop] Excel 읽기 실패: {e}")
        return 0

    corrections = []
    n_rows = min(len(orig_df), len(corr_df))
    n_cols = min(len(orig_df.columns), len(corr_df.columns))

    for r in range(n_rows):
        for c in range(n_cols):
            orig_val = str(orig_df.iloc[r, c]).strip()
            corr_val = str(corr_df.iloc[r, c]).strip()

            if orig_val != corr_val and orig_val and corr_val:
                col_name = str(orig_df.columns[c])
                corrections.append({
                    "row": r,
                    "col": c,
                    "col_name": col_name,
                    "original": orig_val,
                    "corrected": corr_val,
                    "correction_type": _infer_type(col_name, orig_val, corr_val),
                })

    if not corrections:
        print("[FeedbackLoop] 변경사항 없음")
        return 0

    count = apply_learned_corrections(corrections, doc_type)
    print(f"[FeedbackLoop] {count}개 교정 학습 완료:")
    for corr in corrections:
        print(f"  {repr(corr['original'])} → {repr(corr['corrected'])} [{corr['correction_type']}]")

    return count


def learn_from_gt_excel(
    ocr_result_dfs: list,
    gt_excel: str,
    gt_sheet: str = None,
    doc_type: str = "demand_forecast",
) -> int:
    """GT Excel과 OCR 결과를 비교하여 교정 패턴을 학습한다.

    Args:
        ocr_result_dfs: OCR 추출 결과 DataFrames
        gt_excel: Ground Truth Excel 경로
        gt_sheet: 시트 이름 (None이면 첫 번째)
        doc_type: 문서 유형

    Returns:
        학습된 교정 건수
    """
    try:
        if gt_sheet:
            gt_df = pd.read_excel(gt_excel, sheet_name=gt_sheet, header=0, dtype=str).fillna("")
        else:
            gt_df = pd.read_excel(gt_excel, header=0, dtype=str).fillna("")
    except Exception as e:
        print(f"[FeedbackLoop] GT Excel 읽기 실패: {e}")
        return 0

    if not ocr_result_dfs:
        return 0

    ocr_df = ocr_result_dfs[0]
    gt_data = gt_df[gt_df.iloc[:, 0].str.strip() != ""].reset_index(drop=True)

    # OCR 데이터에서 합계 등 제외
    data_rows = [i for i in range(len(ocr_df))
                 if str(ocr_df.iloc[i, 0]).strip() not in
                 ("합계", "비중(%)", "누적합계", "누적비중(%)", "")]
    ocr_data = ocr_df.iloc[data_rows].reset_index(drop=True)

    corrections = []
    n = min(len(gt_data), len(ocr_data))
    col_types = ["company", "department", "name", "account"]

    for r in range(n):
        for ci in range(min(4, ocr_data.shape[1], gt_data.shape[1])):
            gt_val = str(gt_data.iloc[r, ci]).strip()
            ocr_val = str(ocr_data.iloc[r, ci]).strip()
            if gt_val != ocr_val and gt_val and ocr_val:
                ctype = col_types[ci] if ci < len(col_types) else "term"
                corrections.append({
                    "row": r,
                    "col": ci,
                    "col_name": str(ocr_data.columns[ci]),
                    "original": ocr_val,
                    "corrected": gt_val,
                    "correction_type": ctype,
                })

    if not corrections:
        print("[FeedbackLoop] GT와 일치 - 학습할 것 없음")
        return 0

    count = apply_learned_corrections(corrections, doc_type)
    print(f"[FeedbackLoop] GT 기반 {count}개 교정 학습:")
    for corr in corrections:
        print(f"  {repr(corr['original'])} → {repr(corr['corrected'])} [{corr['correction_type']}]")
    return count


def scan_all_outputs() -> int:
    """등록된 모든 출력 파일을 순회하여 사용자 교정을 감지/학습한다."""
    _ensure_table()

    with _conn() as con:
        rows = con.execute(
            "SELECT output_path, doc_type FROM output_tracking WHERE corrections_extracted=0"
        ).fetchall()

    total_corrections = 0
    for row in rows:
        path = row["output_path"]
        doc_type = row["doc_type"]

        corrections = check_for_corrections(path)
        if corrections:
            count = apply_learned_corrections(corrections, doc_type)
            total_corrections += count
            # 교정 추출 완료 표시
            with _conn() as con:
                con.execute(
                    "UPDATE output_tracking SET corrections_extracted=1 WHERE output_path=?",
                    (path,),
                )

    return total_corrections
