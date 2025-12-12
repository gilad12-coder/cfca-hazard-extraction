from pathlib import Path
from typing import Dict, Sequence

import dspy
from loguru import logger

from ..optimization.config import INFERENCE_LM
from ..optimization.constants import FIELDS
from ..optimization.optimizer import HazardSchemaProgram

DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parent.parent.parent / "artifacts"
DEFAULT_STAGE = "gepa"

_lm_configured = False

def configure_inference_lm() -> None:
    """Configure DSPy with the shared inference LM.

    Args:
        None.

    Returns:
        None. Configures global DSPy settings.
    """
    global _lm_configured
    if _lm_configured:
        return
    dspy.configure(lm=dspy.LM(**INFERENCE_LM))
    _lm_configured = True


def _artifact_path(field: str, artifact_root: Path, stage: str) -> Path:
    """Return the artifact directory for a given field/stage.

    Args:
        field: Schema field name.
        artifact_root: Base artifacts directory.
        stage: Stage subdirectory name (e.g., ``gepa``).

    Returns:
        Path to the stage directory for the requested field.
    """
    return Path(artifact_root) / field / stage


def load_field_program(
    field: str,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    stage: str = DEFAULT_STAGE,
) -> dspy.Module:
    """Load a single optimized field program saved by the optimizer.

    Args:
        field: Schema field name to load.
        artifact_root: Root directory containing artifacts.
        stage: Stage subdirectory name.

    Returns:
        dspy.Module instance for the requested field.

    Raises:
        FileNotFoundError: When the artifact directory is missing.
    """
    path = _artifact_path(field, artifact_root, stage)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing optimized program for '{field}' at {path}. "
            "Run the optimizer before loading artifacts."
        )
    logger.info(f"Loading optimized program for '{field}' from {path}")
    return dspy.load(path)


def load_field_programs(
    fields: Sequence[str] | None = None,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    stage: str = DEFAULT_STAGE,
    allow_partial: bool = False,
) -> Dict[str, dspy.Module]:
    """Load optimized field programs for the requested fields.

    Args:
        fields: Optional subset of fields to load; defaults to all fields.
        artifact_root: Root directory containing artifacts.
        stage: Stage subdirectory name.
        allow_partial: When True, skip missing artifacts instead of raising.

    Returns:
        Dict mapping field names to loaded dspy.Module instances.

    Raises:
        FileNotFoundError: When any requested artifact is missing.
    """
    configure_inference_lm()
    targets = list(fields) if fields else list(FIELDS)
    programs: Dict[str, dspy.Module] = {}
    missing: list[str] = []
    for field in targets:
        try:
            programs[field] = load_field_program(
                field, artifact_root=artifact_root, stage=stage
            )
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            missing.append(field)
    if missing and not allow_partial:
        raise FileNotFoundError(
            "Missing optimized programs for fields: "
            f"{', '.join(missing)} (expected under {artifact_root})."
        )
    return programs


def load_schema_program(
    fields: Sequence[str] | None = None,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    stage: str = DEFAULT_STAGE,
    allow_partial: bool = False,
) -> HazardSchemaProgram:
    """Assemble a HazardSchemaProgram from optimized field artifacts.

    Args:
        fields: Optional subset of fields to load; defaults to all.
        artifact_root: Root directory containing artifacts.
        stage: Stage subdirectory name.
        allow_partial: When True, skip missing artifacts instead of raising.

    Returns:
        HazardSchemaProgram composed of per-field modules.

    Raises:
        FileNotFoundError: When any required artifact is missing.
    """
    modules = load_field_programs(
        fields=fields, artifact_root=artifact_root, stage=stage, allow_partial=allow_partial
    )
    return HazardSchemaProgram(field_modules=modules)
