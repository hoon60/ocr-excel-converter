"""문서 학습기 — 처리 결과에서 패턴을 추출하고 ML 분류기를 학습시킨다.

처리 건수가 쌓일수록:
1. TF-IDF + LogisticRegression 분류기가 정확해짐
2. OCR 교정 사전이 풍부해짐
3. 문서별 전용 프롬프트가 구체적으로 진화함
"""

import re
from collections import Counter

import pandas as pd

from .pattern_db import (
    get_all_keywords_by_type,
    get_column_template,
    get_confirmed_documents,
    get_corrections,
    get_top_header_patterns,
    insert_header_pattern,
    upsert_column_stats,
    upsert_column_template,
    upsert_correction,
    upsert_keyword,
    upsert_metadata_pattern,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── 상수 ──
_KNOWN_TYPES = ["demand_forecast", "financial_statement", "default"]
_MIN_TRAIN_SAMPLES = 5  # 이 수 이상 확정 문서가 있어야 ML 모델 사용

# 하드코딩 폴백 키워드 (DB가 비어있을 때 사용)
_FALLBACK_DEMAND = {"수요예측", "참여금액", "모집밴드", "총참여금액", "최종참여일시", "개별민평", "사채만기", "누적합계"}
_FALLBACK_FINANCIAL = {"유동자산", "비유동자산", "자산총계", "부채총계", "자본총계", "영업이익", "당기순이익"}

# 한국어 불용어 (분류에 의미 없는 조사/어미)
_KO_STOPWORDS = {"은", "는", "이", "가", "을", "를", "의", "에", "서", "도", "로", "와", "과", "으로"}


class DocumentClassifier:
    """
    2단계 문서 타입 분류기.

    확정 문서 < 5건: DB 키워드 가중합 기반 폴백
    확정 문서 >= 5건: TF-IDF(char n-gram) + Logistic Regression
    """

    def __init__(self):
        self._pipeline: "Pipeline | None" = None
        self._is_trained = False
        self._training_count = 0

    def train_from_db(self) -> bool:
        """DB의 확정 문서로 분류기를 재학습한다. 성공 시 True 반환."""
        if not SKLEARN_AVAILABLE:
            return False

        confirmed = get_confirmed_documents()
        if len(confirmed) < _MIN_TRAIN_SAMPLES:
            self._is_trained = False
            return False

        texts, labels = [], []
        for doc in confirmed:
            # 각 문서의 텍스트 피처: 파일명 + 확정 타입 키워드
            # 실제 테이블 텍스트는 DB에 저장하지 않으므로 파일명 + 타입 키워드 조합
            label = doc["confirmed_type"] or doc["doc_type"]
            text = doc["file_name"] + " " + label
            texts.append(text)
            labels.append(label)

        # 클래스 수 부족 시 스킵
        if len(set(labels)) < 2:
            return False

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=3000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=1.0, max_iter=500, multi_class="multinomial",
                solver="lbfgs", class_weight="balanced",
            )),
        ])
        try:
            self._pipeline.fit(texts, labels)
            self._is_trained = True
            self._training_count = len(confirmed)
            return True
        except Exception:
            self._is_trained = False
            return False

    def classify(
        self,
        tables: list[pd.DataFrame],
        metadata: dict | None = None,
    ) -> tuple[str, float]:
        """(doc_type, confidence) 반환. confidence는 0.0~1.0."""
        text = self._make_text(tables, metadata)

        if self._is_trained and self._pipeline is not None:
            try:
                proba = self._pipeline.predict_proba([text])[0]
                classes = self._pipeline.classes_
                best_idx = proba.argmax()
                return str(classes[best_idx]), float(proba[best_idx])
            except Exception:
                pass

        return self._keyword_fallback(text)

    def _make_text(
        self,
        tables: list[pd.DataFrame],
        metadata: dict | None,
    ) -> str:
        """분류용 텍스트 조합: 제목 + 헤더 + 첫 열 값."""
        parts = []
        if metadata:
            parts.append(str(metadata.get("title", "")))
            for row in metadata.get("info", []):
                parts.extend(str(v) for v in row)

        for df in tables:
            parts.extend(str(c) for c in df.columns)
            if len(df) > 0:
                parts.extend(str(v) for v in df.iloc[:5, 0].astype(str))

        return " ".join(parts)

    def _keyword_fallback(self, text: str) -> tuple[str, float]:
        """DB 키워드 가중합으로 분류 (ML 미학습 시 폴백)."""
        all_kws = get_all_keywords_by_type()

        scores: dict[str, float] = {}
        for doc_type, kw_list in all_kws.items():
            score = sum(w for kw, w in kw_list if kw in text)
            scores[doc_type] = score

        # 폴백 하드코딩도 추가
        scores["demand_forecast"] = scores.get("demand_forecast", 0) + sum(
            1.5 for kw in _FALLBACK_DEMAND if kw in text
        )
        scores["financial_statement"] = scores.get("financial_statement", 0) + sum(
            1.5 for kw in _FALLBACK_FINANCIAL if kw in text
        )

        if not scores or max(scores.values()) == 0:
            return "default", 0.3

        best = max(scores, key=lambda k: scores[k])
        total = sum(max(v, 0) for v in scores.values())
        conf = scores[best] / total if total > 0 else 0.3
        return best, min(conf, 0.99)


# ── 패턴 추출 (처리 완료 후 호출) ──

def extract_patterns_from_result(
    file_path: str,
    tables: list[pd.DataFrame],
    metadata: dict,
    doc_type: str,
    engine_used: str,
) -> None:
    """파이프라인 처리 결과에서 패턴을 추출해 DB에 저장한다."""
    if not tables:
        return

    try:
        _save_column_patterns(doc_type, tables)
        _save_header_patterns(doc_type, tables)
        _save_keyword_fingerprints(doc_type, tables, metadata)
        _save_metadata_patterns(doc_type, metadata)
        _save_numeric_stats(doc_type, tables)
    except Exception:
        pass  # 학습 실패는 메인 파이프라인에 영향 없음


def _save_column_patterns(doc_type: str, tables: list[pd.DataFrame]) -> None:
    for df in tables:
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            if col_str and not col_str.startswith("Col_"):
                upsert_column_template(doc_type, col_str, i)


def _save_header_patterns(doc_type: str, tables: list[pd.DataFrame]) -> None:
    for df in tables:
        headers = [str(c).strip() for c in df.columns if str(c).strip() and not str(c).startswith("Col_")]
        if len(headers) >= 3:
            insert_header_pattern(doc_type, headers, len(df))


def _save_keyword_fingerprints(
    doc_type: str,
    tables: list[pd.DataFrame],
    metadata: dict,
) -> None:
    """제목/헤더/첫 열에서 유의미한 키워드를 추출해 저장."""
    # 제목 키워드 (가중치 3.0)
    title = str(metadata.get("title", ""))
    for token in _tokenize_ko(title):
        upsert_keyword(doc_type, token, weight=3.0)

    for df in tables:
        # 헤더 키워드 (가중치 2.0)
        for col in df.columns:
            for token in _tokenize_ko(str(col)):
                upsert_keyword(doc_type, token, weight=2.0)

        # 첫 열 고유값 (가중치 1.0) — 회사명/계정과목명 등
        if len(df.columns) > 0:
            unique_vals = df.iloc[:, 0].astype(str).unique()[:20]
            for val in unique_vals:
                for token in _tokenize_ko(val):
                    if len(token) >= 2:
                        upsert_keyword(doc_type, token, weight=1.0)


def _save_metadata_patterns(doc_type: str, metadata: dict) -> None:
    if not metadata:
        return
    info_labels = [row[0] for row in metadata.get("info", []) if row]
    market_labels = [row[0] for row in metadata.get("market_info", []) if row]
    if info_labels:
        upsert_metadata_pattern(doc_type, "info_labels", info_labels)
    if market_labels:
        upsert_metadata_pattern(doc_type, "market_info_labels", market_labels)
    if metadata.get("title"):
        upsert_metadata_pattern(doc_type, "title_sample", metadata["title"])


def _save_numeric_stats(doc_type: str, tables: list[pd.DataFrame]) -> None:
    for df in tables:
        for col in df.columns:
            vals = []
            for v in df[col]:
                try:
                    s = str(v).replace(",", "").strip()
                    if s:
                        vals.append(float(s))
                except (ValueError, TypeError):
                    pass
            if vals:
                upsert_column_stats(doc_type, str(col), vals)


def _tokenize_ko(text: str) -> list[str]:
    """한국어 텍스트에서 유의미한 토큰을 추출한다."""
    text = re.sub(r"[^\w가-힣a-zA-Z0-9]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in _KO_STOPWORDS]


# ── OCR 교정 학습 ──

def learn_ocr_correction(
    doc_type: str,
    wrong: str,
    correct: str,
    correction_type: str = "term",
    confidence: float = 1.0,
) -> None:
    """사용자가 수동 교정한 내용을 DB에 저장한다."""
    if wrong.strip() and correct.strip() and wrong != correct:
        upsert_correction(doc_type, wrong.strip(), correct.strip(), correction_type, confidence)


def learn_from_ai_corrections(
    doc_type: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> None:
    """AI 교정 전후 DataFrame을 비교해 교정 패턴을 학습한다."""
    if before_df.shape != after_df.shape:
        return

    for i in range(min(len(before_df), len(after_df))):
        for j in range(min(len(before_df.columns), len(after_df.columns))):
            try:
                orig = str(before_df.iloc[i, j]).strip()
                corrected = str(after_df.iloc[i, j]).strip()
                if orig and corrected and orig != corrected and len(orig) < 50:
                    ctype = _infer_correction_type(orig, corrected)
                    upsert_correction(doc_type, orig, corrected, ctype, confidence=0.9)
            except (IndexError, TypeError):
                pass


def _infer_correction_type(wrong: str, correct: str) -> str:
    """교정 타입을 자동 추론한다."""
    if re.search(r"\d{4}-\d{2}-\d{2}", correct):
        return "date"
    if re.match(r"^\d+\.?\d*$", correct):
        return "number"
    if any(kw in correct for kw in ["운용", "증권", "보험", "은행", "투자", "자산"]):
        return "company"
    if len(wrong) <= 4 and len(correct) <= 4:
        return "name"
    return "term"


def get_learned_corrections(doc_type: str) -> dict[str, str]:
    """DB에서 학습된 교정 맵을 반환한다. {틀린표현: 올바른표현}"""
    pairs = get_corrections(doc_type, min_confidence=0.5)
    # 일반 corrections도 포함
    pairs += get_corrections("default", min_confidence=0.5)
    return {wrong: correct for wrong, correct in pairs}
