import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .constants import (
    FIELD_BURNING_MATERIAL,
    FIELD_COMMENTS,
    FIELD_EVENT_DATE,
    FIELD_EVENT_ENDED,
    FIELD_EVENT_TIME,
    FIELD_HOME_ADDRESS,
    FIELD_LOCATION,
    FIELD_REPORT_TYPE,
    FIELD_REPORTER_LOCATION,
    FIELD_REPORTER_NAME,
    FIELD_SMELL_INTENSITY,
    FIELD_SMELL_TYPE,
    FIELD_SMOKE_COLOR,
    FIELD_SYMPTOMS,
)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DEFAULT_WORKBOOK = Path(__file__).resolve().parent.parent.parent / "data" / "start_messages.xlsx"

_project_root = Path(__file__).resolve().parent.parent.parent

_artifacts_dir_env = os.environ.get("ARTIFACTS_DIR", "artifacts_gemini")
if "openai" in _artifacts_dir_env.lower():
    MODEL_PROVIDER = "openai"
    ARTIFACT_DIR = _project_root / "artifacts_openai"
else:
    MODEL_PROVIDER = "gemini"
    ARTIFACT_DIR = _project_root / "artifacts_gemini"

SUMMARY_PATH = ARTIFACT_DIR / "summary.json"

# Artifact roots for mixed-provider loading
ARTIFACT_ROOTS: Dict[str, Path] = {
    "openai": _project_root / "artifacts_openai",
    "gemini": _project_root / "artifacts_gemini",
}

# Whether to use mixed-provider model selection (default: True)
USE_MIXED_PROVIDERS = os.environ.get("USE_MIXED_PROVIDERS", "true").lower() == "true"


def _get_google_api_key() -> str:
    """Load Google API key from environment variable."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    return key


def _get_openai_api_key() -> str:
    """Load OpenAI API key from environment variable."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    return key


def get_provider_lm_config(provider: str) -> Dict[str, Any]:
    """Get LM configuration for a specific provider.

    Args:
        provider: Either "openai" or "gemini".

    Returns:
        Dict with LM configuration parameters.

    Raises:
        ValueError: If provider is unknown.
    """
    if provider == "openai":
        return {
            "model": "openai/gpt-5-mini",
            "max_tokens": 16000,
            "temperature": 1.0,
            "reasoning_effort": "low",
            "api_key": _get_openai_api_key(),
        }
    elif provider == "gemini":
        return {
            "model": "gemini/gemini-3-flash-preview",
            "max_tokens": 2048,
            "temperature": 1.0,
            "thinking_level": "minimal",
            "api_key": _get_google_api_key(),
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Legacy configs for backward compatibility (lazy evaluation)
GEMINI_LM_CONFIG = {
    "model": "gemini/gemini-3-flash-preview",
    "max_tokens": 2048,
    "temperature": 1.0,
    "thinking_level": "minimal",
    "api_key": _get_google_api_key() if MODEL_PROVIDER == "gemini" else None,
}

OPENAI_LM_CONFIG = {
    "model": "openai/gpt-5-mini",
    "max_tokens": 16000,
    "temperature": 1.0,
    "reasoning_effort": "low",
    "api_key": _get_openai_api_key() if MODEL_PROVIDER == "openai" else None,
}

if MODEL_PROVIDER == "openai":
    INFERENCE_LM = OPENAI_LM_CONFIG
    TEACHER_LM = OPENAI_LM_CONFIG
else:
    INFERENCE_LM = GEMINI_LM_CONFIG
    TEACHER_LM = GEMINI_LM_CONFIG

FIELD_BEHAVIORS: Dict[str, Dict[str, str]] = {
    FIELD_REPORT_TYPE: {"metric": "exact", "mode": "optimize"},
    FIELD_SMELL_TYPE: {"metric": "exact", "mode": "optimize"},
    FIELD_BURNING_MATERIAL: {"metric": "exact", "mode": "optimize"},
    FIELD_SMELL_INTENSITY: {"metric": "exact", "mode": "optimize"},
    FIELD_SYMPTOMS: {"metric": "llm", "mode": "optimize"},
    FIELD_LOCATION: {"metric": "llm", "mode": "optimize"},
    FIELD_SMOKE_COLOR: {"metric": "exact", "mode": "optimize"},
    FIELD_REPORTER_LOCATION: {"metric": "exact", "mode": "optimize"},
    FIELD_EVENT_ENDED: {"metric": "exact", "mode": "optimize"},
    FIELD_EVENT_DATE: {"metric": "exact", "mode": "optimize"},
    FIELD_EVENT_TIME: {"metric": "exact", "mode": "optimize"},
    FIELD_REPORTER_NAME: {"metric": "llm", "mode": "optimize"},
    FIELD_HOME_ADDRESS: {"metric": "llm", "mode": "optimize"},
    FIELD_COMMENTS: {"metric": "llm", "mode": "optimize"},
}
