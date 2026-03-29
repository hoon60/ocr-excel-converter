"""PDF 타입 분류기 — 텍스트 PDF vs 스캔 PDF 자동 판별."""

import fitz  # PyMuPDF


def classify_pdf(path: str) -> str:
    """PDF를 텍스트/스캔으로 분류한다.

    판별 기준: 페이지당 평균 문자 수가 50자 이상이면 텍스트 PDF,
    미만이면 스캔(이미지) PDF로 분류한다.
    """
    doc = fitz.open(path)
    if doc.page_count == 0:
        doc.close()
        return "scan"

    total_chars = sum(len(page.get_text("text").strip()) for page in doc)
    chars_per_page = total_chars / doc.page_count
    doc.close()

    # 페이지당 50자 미만이면 스캔 PDF로 판별
    return "text" if chars_per_page >= 50 else "scan"


def get_page_count(path: str) -> int:
    """PDF 페이지 수를 반환한다."""
    doc = fitz.open(path)
    count = doc.page_count
    doc.close()
    return count
