"""Clean benchmark runner for OCR Excel outputs.

Runs repo-local benchmark cases with evaluation contamination disabled and writes
both Excel outputs and a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import openpyxl

from core.pipeline import run_pipeline
from test_pipeline import EXPECTED_DATA as STATEMENT_EXPECTED
from test_samsung_real import REAL_DATA as SAMSUNG_EXPECTED


ROOT = Path(__file__).resolve().parent


def _normalize(text: object) -> str:
    return (
        str(text or "")
        .replace(",", "")
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("·", "")
        .replace("ㆍ", "")
        .strip()
    )


def _clean_expected_rows(rows: Iterable[Iterable[object]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows:
        values = [str(v).strip() for v in row]
        if not any(values):
            continue
        if not _normalize(values[0]):
            continue
        cleaned.append(values[:4])
    return cleaned


def _load_sheet_rows(path: Path) -> list[list[str]]:
    wb = openpyxl.load_workbook(path)
    ws = wb.worksheets[0]
    rows = [
        ["" if value is None else str(value).strip() for value in row]
        for row in ws.iter_rows(values_only=True)
    ]
    wb.close()
    return rows


def _evaluate_output(expected_rows: list[list[str]], actual_rows: list[list[str]]) -> dict:
    row_hits = 0
    exact_row_hits = 0
    cell_matches = 0
    total_cells = len(expected_rows) * 4
    misses: list[dict] = []

    for expected in expected_rows:
        matched = None
        for actual in actual_rows:
            if actual and _normalize(actual[0]) == _normalize(expected[0]):
                matched = actual[:4]
                break

        if matched is None:
            misses.append({"label": expected[0], "reason": "row_not_found"})
            continue

        row_hits += 1
        normalized_expected = [_normalize(v) for v in expected[:4]]
        normalized_actual = [_normalize(v) for v in matched[:4]]

        if normalized_expected == normalized_actual:
            exact_row_hits += 1

        for exp_val, act_val in zip(normalized_expected, normalized_actual):
            if exp_val == act_val:
                cell_matches += 1

        if normalized_expected != normalized_actual and len(misses) < 10:
            misses.append(
                {
                    "label": expected[0],
                    "expected": expected[:4],
                    "actual": matched[:4],
                }
            )

    return {
        "expected_rows": len(expected_rows),
        "row_hits": row_hits,
        "exact_row_hits": exact_row_hits,
        "row_recall": round(row_hits / len(expected_rows), 4) if expected_rows else 0.0,
        "exact_row_accuracy": round(exact_row_hits / len(expected_rows), 4) if expected_rows else 0.0,
        "cell_matches": cell_matches,
        "total_cells": total_cells,
        "cell_accuracy": round(cell_matches / total_cells, 4) if total_cells else 0.0,
        "sample_mismatches": misses,
    }


@contextmanager
def _benchmark_env(temp_data_dir: str):
    keys = {
        "OCR_BENCHMARK_MODE": "1",
        "OCR_DISABLE_CACHE": "1",
        "OCR_DISABLE_LEARNING": "1",
        "OCR_DISABLE_FEEDBACK": "1",
        "OCR_DISABLE_ENTITY_CORRECTIONS": "1",
        "OCR_DISABLE_DB_CORRECTIONS": "1",
        "OCR_DATA_DIR": temp_data_dir,
    }
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(keys)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_case(
    case_name: str,
    input_path: Path,
    expected_rows: list[list[str]],
    save_dir: Path,
    use_ai: bool,
) -> dict:
    output_name = f"{case_name}_{'ai' if use_ai else 'local'}_benchmark.xlsx"
    output_path = save_dir / output_name

    result_path = run_pipeline(
        str(input_path),
        output_path=str(output_path),
        use_ai=use_ai,
        benchmark_mode=True,
    )
    actual_rows = _load_sheet_rows(Path(result_path))
    metrics = _evaluate_output(expected_rows, actual_rows)
    metrics["case"] = case_name
    metrics["input_path"] = str(input_path)
    metrics["output_path"] = str(output_path)
    metrics["use_ai"] = use_ai
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean OCR benchmarks.")
    parser.add_argument(
        "--case",
        choices=["all", "samsung", "statement"],
        default="all",
        help="Benchmark case to run.",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Enable the AI path while keeping evaluation contamination disabled.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "benchmark_outputs"),
        help="Directory for generated benchmark Excel files and report.",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "samsung": (
            ROOT / "samsung_dart_financial.png",
            _clean_expected_rows(SAMSUNG_EXPECTED),
        ),
        "statement": (
            ROOT / "test_financial_statement.png",
            _clean_expected_rows(STATEMENT_EXPECTED),
        ),
    }

    selected = cases.items() if args.case == "all" else [(args.case, cases[args.case])]

    with TemporaryDirectory(prefix="ocr_excel_benchmark_") as temp_dir:
        with _benchmark_env(temp_dir):
            results = [
                _run_case(name, input_path, expected_rows, save_dir, args.use_ai)
                for name, (input_path, expected_rows) in selected
            ]

    summary = {
        "benchmark_mode": True,
        "use_ai": args.use_ai,
        "results": results,
        "aggregate": {
            "cases": len(results),
            "avg_row_recall": round(sum(r["row_recall"] for r in results) / len(results), 4),
            "avg_exact_row_accuracy": round(
                sum(r["exact_row_accuracy"] for r in results) / len(results), 4
            ),
            "avg_cell_accuracy": round(sum(r["cell_accuracy"] for r in results) / len(results), 4),
        },
    }

    report_path = save_dir / f"benchmark_report_{'ai' if args.use_ai else 'local'}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Clean Benchmark Summary")
    print("=" * 72)
    for result in results:
        print(
            f"{result['case']}: row_recall={result['row_recall']:.2%}, "
            f"exact_row_accuracy={result['exact_row_accuracy']:.2%}, "
            f"cell_accuracy={result['cell_accuracy']:.2%}"
        )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
