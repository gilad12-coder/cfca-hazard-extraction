import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import dspy
from loguru import logger

from ..optimization.config import (
    ARTIFACT_ROOTS,
    get_inference_lm_config,
    get_provider_lm_config,
)
from ..optimization.constants import FIELDS
from ..optimization.optimizer import HazardSchemaProgram
from .model_selector import FieldSelection, ModelSelector

# Configurable via ARTIFACTS_DIR env var (defaults to artifacts_openai)
_default_artifacts = Path(__file__).resolve().parent.parent.parent / "artifacts_openai"
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("ARTIFACTS_DIR", str(_default_artifacts)))
DEFAULT_STAGE = "gepa"

_lm_configured = False


@dataclass
class LoadedFieldModule:
    """A loaded DSPy module with its associated provider and LM."""

    field: str
    module: dspy.Module
    provider: str
    lm: dspy.LM
    score: float


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
    dspy.configure(lm=dspy.LM(**get_inference_lm_config()))
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


class MixedProviderLoader:
    """Load field artifacts from multiple providers based on score selection."""

    def __init__(
        self,
        provider_roots: Optional[Dict[str, Path]] = None,
        stage: str = DEFAULT_STAGE,
    ):
        """Initialize the mixed provider loader.

        Args:
            provider_roots: Mapping of provider names to artifact directories.
                           Defaults to ARTIFACT_ROOTS from config.
            stage: Stage subdirectory name (e.g., "gepa").
        """
        self.provider_roots = {
            k: Path(v) for k, v in (provider_roots or ARTIFACT_ROOTS).items()
        }
        self.stage = stage
        self._selector = ModelSelector(self.provider_roots)
        self._provider_lms: Dict[str, dspy.LM] = {}

    def _get_or_create_lm(self, provider: str) -> dspy.LM:
        """Get or create LM instance for a provider.

        Args:
            provider: Provider name ("openai" or "gemini").

        Returns:
            Configured dspy.LM instance for the provider.
        """
        if provider not in self._provider_lms:
            config = get_provider_lm_config(provider)
            self._provider_lms[provider] = dspy.LM(**config)
        return self._provider_lms[provider]

    def load_best_field_modules(
        self,
        fields: Optional[Sequence[str]] = None,
        default_provider: str = "gemini",
    ) -> Dict[str, LoadedFieldModule]:
        """Load the best-scoring artifact for each field from any provider.

        Args:
            fields: Subset of fields; None loads all available.
            default_provider: Tie-breaker preference.

        Returns:
            Dict mapping field name to LoadedFieldModule.
        """
        selections = self._selector.select_best_providers(
            fields=fields,
            default_provider=default_provider,
        )

        loaded: Dict[str, LoadedFieldModule] = {}

        for field, selection in selections.items():
            # Try selected provider first, then fallback to alternative
            providers_to_try = [(selection.selected_provider, selection.artifact_path, selection.selected_score)]
            if selection.alternative_provider:
                alt_path = self.provider_roots[selection.alternative_provider] / field / self.stage
                providers_to_try.append((selection.alternative_provider, alt_path, selection.alternative_score))

            for provider, artifact_path, score in providers_to_try:
                if not artifact_path.exists():
                    logger.warning(f"Artifact path does not exist: {artifact_path}")
                    continue

                try:
                    lm = self._get_or_create_lm(provider)
                    dspy.configure(lm=lm)
                    module = dspy.load(artifact_path)
                    loaded[field] = LoadedFieldModule(
                        field=field,
                        module=module,
                        provider=provider,
                        lm=lm,
                        score=score,
                    )
                    if provider != selection.selected_provider:
                        logger.warning(
                            f"Loaded '{field}' from fallback provider '{provider}' "
                            f"(score={score:.4f}) - primary provider failed"
                        )
                    else:
                        logger.debug(
                            f"Loaded '{field}' from {provider} "
                            f"(score={score:.4f})"
                        )
                    break  # Successfully loaded, stop trying
                except Exception as e:
                    logger.error(f"Failed to load '{field}' from {provider}: {e}")
            else:
                logger.error(f"Could not load '{field}' from any provider")

        # Log summary by provider
        provider_fields: Dict[str, list[str]] = {}
        for field, mod in loaded.items():
            provider_fields.setdefault(mod.provider, []).append(field)

        for provider, fields_list in sorted(provider_fields.items()):
            logger.info(f"Using {provider} for {len(fields_list)} fields: {', '.join(sorted(fields_list))}")

        return loaded
