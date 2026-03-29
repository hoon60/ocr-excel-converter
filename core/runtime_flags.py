"""런타임 플래그 — 벤치마크 시 평가 오염을 막기 위한 제어."""

import os
from dataclasses import dataclass
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeFlags:
    benchmark_mode: bool
    disable_cache: bool
    disable_learning: bool
    disable_feedback: bool
    disable_entity_corrections: bool
    disable_db_corrections: bool


def get_runtime_flags(benchmark_override: Optional[bool] = None) -> RuntimeFlags:
    """환경변수 기반 런타임 플래그를 반환한다."""
    benchmark_mode = (
        benchmark_override
        if benchmark_override is not None
        else _env_flag("OCR_BENCHMARK_MODE", False)
    )
    return RuntimeFlags(
        benchmark_mode=benchmark_mode,
        disable_cache=_env_flag("OCR_DISABLE_CACHE", benchmark_mode),
        disable_learning=_env_flag("OCR_DISABLE_LEARNING", benchmark_mode),
        disable_feedback=_env_flag("OCR_DISABLE_FEEDBACK", benchmark_mode),
        disable_entity_corrections=_env_flag("OCR_DISABLE_ENTITY_CORRECTIONS", benchmark_mode),
        disable_db_corrections=_env_flag("OCR_DISABLE_DB_CORRECTIONS", benchmark_mode),
    )
