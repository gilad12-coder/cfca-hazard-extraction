"""Intelligent model selection based on optimization scores.

Compares evaluation scores from multiple providers and selects
the best-performing model for each field.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from loguru import logger


@dataclass
class FieldScore:
    """Score information for a single field from a provider."""

    field: str
    provider: str
    score: float
    artifact_path: Path


@dataclass
class FieldSelection:
    """The selected provider for a field."""

    field: str
    selected_provider: str
    selected_score: float
    artifact_path: Path
    alternative_provider: Optional[str] = None
    alternative_score: Optional[float] = None


class ModelSelector:
    """Select the best model provider for each field based on optimization scores."""

    DEFAULT_PROVIDER = "openai"

    def __init__(
        self,
        provider_roots: Dict[str, Path],
        score_key: str = "gepa_optimized_score",
    ):
        """Initialize the model selector.

        Args:
            provider_roots: Mapping of provider names to artifact directories.
            score_key: Key in summary.json to use for score comparison.
        """
        self.provider_roots = {k: Path(v) for k, v in provider_roots.items()}
        self.score_key = score_key
        self._scores: Dict[str, Dict[str, FieldScore]] = {}

    def load_summaries(self) -> None:
        """Load summary.json from each provider directory."""
        for provider, root in self.provider_roots.items():
            summary_path = root / "summary.json"
            if not summary_path.exists():
                logger.warning(
                    f"No summary.json found for provider '{provider}' at {summary_path}"
                )
                continue

            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)

            self._scores[provider] = {}
            for field, metrics in data.items():
                score = metrics.get(self.score_key)
                if score is None:
                    logger.warning(
                        f"Missing {self.score_key} for field '{field}' in provider '{provider}'"
                    )
                    continue
                self._scores[provider][field] = FieldScore(
                    field=field,
                    provider=provider,
                    score=float(score),
                    artifact_path=root / field / "gepa",
                )

    def select_best_providers(
        self,
        fields: Optional[Sequence[str]] = None,
        default_provider: str = DEFAULT_PROVIDER,
    ) -> Dict[str, FieldSelection]:
        """Select the best provider for each field.

        Args:
            fields: Subset of fields to select; None means all available.
            default_provider: Provider to prefer in case of ties.

        Returns:
            Dict mapping field names to FieldSelection objects.
        """
        if not self._scores:
            self.load_summaries()

        # Gather all fields across providers
        all_fields = set()
        for provider_scores in self._scores.values():
            all_fields.update(provider_scores.keys())

        if fields:
            all_fields &= set(fields)

        selections: Dict[str, FieldSelection] = {}

        for field in all_fields:
            candidates = []
            for provider, provider_scores in self._scores.items():
                if field in provider_scores:
                    candidates.append(provider_scores[field])

            if not candidates:
                logger.warning(f"No scores found for field '{field}'")
                continue

            # Sort by score descending, then by provider preference for ties
            candidates.sort(
                key=lambda c: (c.score, c.provider == default_provider), reverse=True
            )

            best = candidates[0]
            alternative = candidates[1] if len(candidates) > 1 else None

            selections[field] = FieldSelection(
                field=field,
                selected_provider=best.provider,
                selected_score=best.score,
                artifact_path=best.artifact_path,
                alternative_provider=alternative.provider if alternative else None,
                alternative_score=alternative.score if alternative else None,
            )

            if alternative and best.score > alternative.score:
                logger.info(
                    f"[{field}] Selected '{best.provider}' (score={best.score:.4f}) "
                    f"over '{alternative.provider}' (score={alternative.score:.4f})"
                )
            elif alternative:
                logger.info(
                    f"[{field}] Selected '{best.provider}' (score={best.score:.4f}) "
                    f"- tie with '{alternative.provider}'"
                )
            else:
                logger.info(
                    f"[{field}] Selected '{best.provider}' (score={best.score:.4f})"
                )

        return selections
