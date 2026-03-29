"""통합 파이프라인 — 파일 타입에 따라 적절한 추출 경로를 선택한다.

Vision AI 폴백 체인: Gemini → Groq → RapidOCR
ML 패턴 학습: 처리마다 패턴 DB에 학습 데이터 누적 → 분류/프롬프트 자동 강화
"""

import os
from typing import Callable

import pandas as pd

from .ai_postprocess import correct_with_ai
from .classifier import classify_pdf
from .excel_writer import write_excel
from .ocr_extractor import extract_tables_from_image_file, extract_tables_from_scan_pdf
from .postprocess import validate_and_annotate
from .text_extractor import extract_tables_from_text_pdf

# ── ML 학습 모듈 (lazy import — 실패해도 파이프라인은 계속 동작)
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from .pattern_db import initialize_db
            from .doc_learner import DocumentClassifier
            initialize_db()
            _classifier = DocumentClassifier()
            _classifier.train_from_db()
        except Exception:
            _classifier = _FallbackClassifier()
    return _classifier


def _get_classifier_for_mode(benchmark_mode: bool):
    if benchmark_mode:
        return _FallbackClassifier()
    return _get_classifier()


class _FallbackClassifier:
    """패턴 DB 없을 때 사용하는 키워드 기반 폴백."""
    _DEMAND = {"수요예측", "참여금액", "모집밴드", "총참여금액", "최종참여일시"}
    _FINANCIAL = {"유동자산", "비유동자산", "자산총계", "부채총계", "자본총계"}

    def classify(self, tables, metadata=None):
        text = str(metadata or "")
        for df in tables:
            text += " ".join(str(c) for c in df.columns)
            if len(df) > 0:
                text += " ".join(str(v) for v in df.iloc[:5, 0].astype(str))
        if any(k in text for k in self._DEMAND):
            return "demand_forecast", 0.8
        if any(k in text for k in self._FINANCIAL):
            return "financial_statement", 0.8
        return "default", 0.3


# ── 지원 형식 ──
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSION = ".pdf"


def _generate_output_path(input_path: str, output_dir: str | None = None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"{base}_변환결과.xlsx"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, out_name)
    return os.path.join(os.path.dirname(input_path), out_name)


def _get_image_size(file_path: str) -> tuple[int, int]:
    try:
        from .ocr_extractor import _imread_unicode
        img = _imread_unicode(file_path)
        if img is not None:
            return img.shape[1], img.shape[0]  # width, height
    except Exception:
        pass
    return 0, 0


def run_pipeline(
    file_path: str,
    output_path: str | None = None,
    progress_cb: Callable[[int], None] | None = None,
    use_ai: bool = False,
    groq_api_key: str | None = None,
    gemini_api_key: str | None = None,
    vision_engine: str = "auto",
    benchmark_mode: bool = False,
) -> str:
    """파일을 분석하여 표를 추출하고 Excel로 저장한다.

    Args:
        file_path: 입력 파일 경로 (PDF 또는 이미지)
        output_path: 출력 Excel 경로 (None이면 자동 생성)
        progress_cb: 진행률 콜백 (0~100)
        use_ai: Vision AI 사용 여부
        groq_api_key: Groq API 키
        gemini_api_key: Gemini API 키
        vision_engine: "auto", "gemini", "groq"
        benchmark_mode: 평가용 읽기 전용 모드. 캐시/학습/피드백/교정 기록을 비활성화한다.

    Returns:
        저장된 Excel 파일 경로
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if output_path is None:
        output_path = _generate_output_path(file_path)

    tables: list[pd.DataFrame] = []
    metadata: dict = {}
    engine_used = "rapidocr"

    # ── [통합1] 이미지 크기 + OCR 키워드 기반 문서 타입 추정 → 동적 프롬프트 ──
    pre_doc_type = None
    dynamic_prompt = None
    if use_ai and ext in IMAGE_EXTENSIONS:
        try:
            from .prompt_builder import build_prompt, estimate_doc_type_from_image
            img_w, img_h = _get_image_size(file_path)
            # 빠른 OCR 스캔으로 키워드 추출 (문서 타입 추정용)
            ocr_hint_text = None
            try:
                from .paddle_engine import _get_ocr, _imread_unicode
                _quick_img = _imread_unicode(file_path)
                if _quick_img is not None:
                    _quick_ocr = _get_ocr()
                    _quick_raw = _quick_ocr.ocr(file_path, cls=False)
                    if _quick_raw and _quick_raw[0]:
                        ocr_hint_text = " ".join(
                            item[1][0] for item in _quick_raw[0][:30]
                        )
            except Exception:
                pass
            pre_doc_type = estimate_doc_type_from_image(img_w, img_h, ocr_hint_text)
            dynamic_prompt = build_prompt(doc_type=pre_doc_type, image_width_px=img_w)
        except Exception:
            pass

    # ── 파일 타입별 추출 ──
    if ext == PDF_EXTENSION:
        pdf_type = classify_pdf(file_path)
        if progress_cb:
            progress_cb(5)

        if pdf_type == "text":
            tables = extract_tables_from_text_pdf(
                file_path,
                progress_cb=lambda p: progress_cb(5 + int(p * 0.7)) if progress_cb else None,
            )
        else:
            tables = extract_tables_from_scan_pdf(
                file_path,
                progress_cb=lambda p: progress_cb(5 + int(p * 0.7)) if progress_cb else None,
            )

    elif ext in IMAGE_EXTENSIONS:
        # [통합2] Vision AI — 동적 프롬프트 주입 + 합의 투표
        if use_ai:
            from .vision_ocr import extract_with_vision

            # 합의 투표 설정 로드
            _consensus_runs = 1
            try:
                import json as _json
                _cfg_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config.json"
                )
                if os.path.isfile(_cfg_path):
                    with open(_cfg_path, encoding="utf-8") as _f:
                        _cfg = _json.load(_f)
                    _consensus_runs = _cfg.get("groq_consensus_runs", 1)
            except Exception:
                pass

            result = extract_with_vision(
                file_path,
                gemini_api_key=gemini_api_key,
                groq_api_key=groq_api_key,
                engine=vision_engine,
                prompt_override=dynamic_prompt,
                progress_cb=lambda p: progress_cb(5 + int(p * 0.7)) if progress_cb else None,
                consensus_runs=_consensus_runs,
                benchmark_mode=benchmark_mode,
            )
            if result:
                tables, metadata = result
                engine_used = vision_engine if vision_engine != "auto" else "gemini"

        # RapidOCR 폴백
        if not tables:
            tables = extract_tables_from_image_file(
                file_path,
                progress_cb=lambda p: progress_cb(5 + int(p * 0.7)) if progress_cb else None,
            )
            engine_used = "rapidocr"
    else:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {ext}\n"
            f"지원 형식: PDF, {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if progress_cb:
        progress_cb(75)

    # AI 보정 (RapidOCR 경로일 때만)
    if use_ai and tables and not metadata:
        try:
            tables = correct_with_ai(
                tables,
                api_key=groq_api_key,
                progress_cb=lambda p: progress_cb(75 + int(p * 0.1)) if progress_cb else None,
            )
        except Exception:
            pass

    # ── [통합3] ML 분류기로 문서 타입 확정 ──
    doc_type = "default"
    confidence = 0.3
    doc_id = None
    try:
        clf = _get_classifier_for_mode(benchmark_mode)
        doc_type, confidence = clf.classify(tables, metadata)

        if not benchmark_mode:
            # DB에 처리 이력 저장
            from .pattern_db import compute_file_hash, insert_document
            file_hash = compute_file_hash(file_path)
            doc_id = insert_document(
                file_name=os.path.basename(file_path),
                file_hash=file_hash,
                doc_type=doc_type,
                confidence=confidence,
                engine_used=engine_used,
                num_tables=len(tables),
                num_rows=sum(len(df) for df in tables),
                num_cols=max((len(df.columns) for df in tables), default=0),
            )
    except Exception:
        pass

    style_profile = doc_type
    if progress_cb:
        progress_cb(88)

    # 숫자 검증
    tables, error_cells = validate_and_annotate(tables)

    # 수요예측표 전용: 격자 보정 + 합계 교차검증
    if doc_type == "demand_forecast" and ext in IMAGE_EXTENSIONS:
        # OpenCV 격자 기반 스프레드 열 보정 (Vision AI 경로에 있던 것을 공통 적용)
        try:
            from .vision_ocr import correct_spread_placement
            tables = correct_spread_placement(file_path, tables, metadata)
        except Exception:
            pass
        # 열 합계 교차검증
        try:
            from .postprocess import validate_demand_forecast_sums
            tables, sum_errors = validate_demand_forecast_sums(tables, doc_type)
            for i, errs in enumerate(sum_errors):
                if i < len(error_cells):
                    error_cells[i].extend(errs)
        except Exception:
            pass

    if progress_cb:
        progress_cb(93)

    # Excel 저장
    result_path = write_excel(
        tables, error_cells, output_path,
        style_profile=style_profile,
        metadata=metadata,
    )

    # ── [통합4] 패턴 학습 + 엔티티 학습 + 템플릿 학습 ──
    if not benchmark_mode:
        try:
            from .doc_learner import extract_patterns_from_result
            extract_patterns_from_result(
                file_path=file_path,
                tables=tables,
                metadata=metadata,
                doc_type=doc_type,
                engine_used=engine_used,
            )
            # 분류기 재학습 (새 패턴 반영)
            _get_classifier().train_from_db()
        except Exception:
            pass

    # ── [통합5] 엔티티 학습 (텍스트 열에서 새 엔티티 추출) ──
    if not benchmark_mode:
        try:
            from .entity_dict import learn_entity
            for df in tables:
                if len(df.columns) >= 4 and len(df) > 0:
                    for val in df.iloc[:, 0].dropna().unique():
                        v = str(val).strip()
                        if len(v) >= 2 and not v.replace(",", "").replace("-", "").isdigit():
                            learn_entity(v, "company")
                    for val in df.iloc[:, 1].dropna().unique():
                        v = str(val).strip()
                        if len(v) >= 2 and not v.replace(",", "").replace("-", "").isdigit():
                            learn_entity(v, "department")
        except Exception:
            pass

    # ── [통합6] 템플릿 학습 (이미지 → 격자 구조 저장) ──
    if ext in IMAGE_EXTENSIONS and not benchmark_mode:
        try:
            from .template_ocr import learn_template
            learn_template(file_path, tables, doc_type)
        except Exception:
            pass

    # ── [통합7] 피드백 루프 — 출력 파일 등록 + 이전 교정 감지 ──
    if not benchmark_mode:
        try:
            from .feedback_loop import register_output, scan_all_outputs
            file_hash_fb = compute_file_hash(file_path) if 'compute_file_hash' in dir() else ""
            register_output(result_path, file_hash_fb, doc_type, tables)
            # 이전 출력 파일의 사용자 교정 감지
            n_corrections = scan_all_outputs()
            if n_corrections > 0:
                print(f"[FeedbackLoop] {n_corrections}개 사용자 교정 학습 완료")
        except Exception:
            pass

    if progress_cb:
        progress_cb(100)

    return result_path
