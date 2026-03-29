"""텍스트 PDF에서 표를 추출한다 (pdfplumber 기반)."""

from typing import Callable

import pandas as pd
import pdfplumber


def extract_tables_from_text_pdf(
    pdf_path: str,
    progress_cb: Callable[[int], None] | None = None,
) -> list[pd.DataFrame]:
    """텍스트 PDF의 모든 페이지에서 표를 추출한다.

    Returns:
        각 표를 DataFrame으로 변환한 리스트.
        표가 없으면 빈 리스트를 반환한다.
    """
    results: list[pd.DataFrame] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )

            # 선 기반으로 못 찾으면 텍스트 기반으로 재시도
            if not tables:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_y_tolerance": 5,
                        "intersection_y_tolerance": 10,
                    }
                )

            for table in tables:
                if not table or len(table) < 2:
                    continue

                # 첫 행을 헤더로 사용
                header = [str(c).strip() if c else f"Col_{j}" for j, c in enumerate(table[0])]
                rows = []
                for row in table[1:]:
                    cleaned = [str(c).strip() if c else "" for c in row]
                    rows.append(cleaned)

                df = pd.DataFrame(rows, columns=header)

                # 완전히 빈 행/열 제거
                df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
                df = df.fillna("")

                if not df.empty:
                    results.append(df)

            if progress_cb:
                progress_cb(int((i + 1) / total_pages * 100))

    return results
