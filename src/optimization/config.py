import json
import os
from pathlib import Path
from typing import Dict

from .constants import (
    FIELD_BURNING_MATERIAL,
    FIELD_COMMENTS,
    FIELD_EVENT_DATE,
    FIELD_EVENT_ENDED,
    FIELD_EVENT_TIME,
    FIELD_HOME_ADDRESS,
    FIELD_LOCATION,
    FIELD_REPORTER_LOCATION,
    FIELD_REPORTER_NAME,
    FIELD_REPORT_TYPE,
    FIELD_SMELL_INTENSITY,
    FIELD_SMELL_TYPE,
    FIELD_SMOKE_COLOR,
    FIELD_SYMPTOMS,
)

DEFAULT_WORKBOOK = Path(__file__).resolve().parent.parent.parent / "data" / "start_messages.xlsx"


def _require_api_key() -> str:
    """Return the configured OpenAI API key.

    Returns:
        str: The API key from the environment; raises if missing.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set. Export your key before running the optimizer.")
    return key


OPENAI_API_KEY = _require_api_key()

INFERENCE_LM = {
    "model": "openai/gpt-5-mini",
    "max_tokens": 16000,
    "temperature": 1.0,
    "reasoning_effort": "low",
    "api_key": OPENAI_API_KEY,
}

TEACHER_LM = {
    "model": "openai/gpt-5",
    "max_tokens": 20000,
    "temperature": 1.0,
    "reasoning_effort": "high",
    "api_key": OPENAI_API_KEY,
}

# Unified field configuration describing both metric strategy and execution mode.
# metric: "llm" (semantic judge) or "exact" (literal equality)
# mode: "optimize" (GEPA) or "evaluate" (baseline evaluation only)
SUMMARY_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "summary.json"

FALLBACK_FIELD_BEHAVIORS: Dict[str, Dict[str, str]] = {
    FIELD_COMMENTS: {"metric": "llm", "mode": "optimize"},
}


def _load_field_behaviors_from_summary() -> Dict[str, Dict[str, str]]:
    """Derive per-field behaviors from artifacts/summary.json when available.

    Returns:
        Dict mapping field names to behavior dicts with ``metric`` and ``mode`` keys.
        Returns an empty dict when no valid summary is present.
    """
    if not SUMMARY_PATH.exists():
        return {}
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    behaviors: Dict[str, Dict[str, str]] = {}
    allowed_fields = set(FALLBACK_FIELD_BEHAVIORS)
    for field in summary:
        if field not in allowed_fields:
            continue
        metric = FALLBACK_FIELD_BEHAVIORS.get(field, {}).get("metric", "llm")
        mode = "optimize"
        behaviors[field] = {"metric": metric, "mode": mode}
    return behaviors


FIELD_BEHAVIORS: Dict[str, Dict[str, str]] = (
    _load_field_behaviors_from_summary() or FALLBACK_FIELD_BEHAVIORS
)
