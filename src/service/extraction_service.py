import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Annotated, Dict, Mapping, Sequence, cast

import dspy
from fastapi import FastAPI, HTTPException, Request, Depends
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..optimization.constants import FIELDS
from ..optimization.schema import HazardReport
from .artifact_loader import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_STAGE,
    configure_inference_lm,
    load_field_programs,
)


class ExtractionRequest(BaseModel):
    """Request model for the FastAPI /extract endpoint."""

    incident_text: str = Field(
        ...,
        description="Incident narrative text.",
        min_length=1,
        max_length=10_000,
    )
    fields: Sequence[str] | None = Field(
        default=None,
        description="Optional subset of fields to extract.",
    )

    @model_validator(mode="after")
    def _validate_fields_subset(self) -> "ExtractionRequest":
        """Ensure requested fields are known schema fields.

        Returns:
            The validated ExtractionRequest.

        Raises:
            ValueError: When unknown field names are supplied.
        """
        if self.fields is None:
            return self
        valid_fields = set(FIELDS)
        invalid = [f for f in self.fields if f not in valid_fields]
        if invalid:
            raise ValueError(f"Unknown fields requested: {', '.join(invalid)}")
        return self


class AsyncHazardExtractionService:
    """Run Chain-of-Thought field extractors concurrently."""

    def __init__(self, field_modules: Mapping[str, dspy.Module]):
        """Create a service backed by preloaded field modules.

        Args:
            field_modules: Mapping of schema field names to DSPy modules.

        Raises:
            ValueError: If no field modules are provided.
        """
        self.field_modules = dict(field_modules)
        if not self.field_modules:
            raise ValueError("At least one field module is required.")
        configure_inference_lm()

    @classmethod
    def from_artifacts(
            cls,
            artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
            stage: str = DEFAULT_STAGE,
            allow_partial: bool = False,
    ) -> "AsyncHazardExtractionService":
        """Instantiate the service by loading optimized programs from disk.

        Args:
            artifact_root: Directory containing saved DSPy programs.
            stage: Artifact stage subdirectory.
            allow_partial: Whether to skip missing artifacts instead of raising.

        Returns:
            AsyncHazardExtractionService ready for inference.
        """
        modules = load_field_programs(
            fields=None,
            artifact_root=Path(artifact_root),
            stage=stage,
            allow_partial=allow_partial,
        )
        return cls(modules)

    async def extract(
            self,
            incident_text: str,
            requested_fields: Sequence[str] | None = None
    ) -> Dict[str, Any]:
        """Extract hazard schema fields asynchronously from an incident narrative.

        Args:
            incident_text: Hebrew incident narrative.
            requested_fields: Optional subset of fields to filter execution.

        Returns:
            Dict keyed by schema field aliases with extracted values.
        """
        if requested_fields:
            missing = [f for f in requested_fields if f not in self.field_modules]
            if missing:
                raise ValueError(f"Artifacts missing for fields: {', '.join(missing)}")
            modules_to_run = {k: v for k, v in self.field_modules.items() if k in requested_fields}
        else:
            modules_to_run = self.field_modules

        field_values = await self._predict_fields_concurrently(incident_text, modules_to_run)
        try:
            hazard = HazardReport(**field_values)
            payload = hazard.model_dump(by_alias=True, exclude_none=False)
            payload.setdefault("symptoms", hazard.symptoms or [])
            return payload
        except ValidationError as exc:
            logger.warning(f"Schema validation failed; returning raw field outputs. Details: {exc}")
            payload = {field: field_values.get(field) for field in FIELDS}
            payload.setdefault("symptoms", field_values.get("symptoms") or [])
            return payload

    async def _predict_fields_concurrently(
            self,
            incident_text: str,
            modules: Dict[str, dspy.Module]
    ) -> Dict[str, Any]:
        """Run selected modules concurrently.

        Args:
            incident_text: Narrative to pass to each field extractor.
            modules: Dictionary of DSPy modules to execute.

        Returns:
            Dict mapping field name to predicted value.
        """
        ordered_items = [
            (field, modules[field])
            for field in FIELDS
            if field in modules
        ]

        tasks = [
            self._predict_single_field(field, module, incident_text)
            for field, module in ordered_items
        ]

        results = await asyncio.gather(*tasks)
        return dict(results)

    @staticmethod
    async def _predict_single_field(
            field: str, module: dspy.Module, incident_text: str
    ) -> tuple[str, Any]:
        """Run a single field extractor in a worker thread to avoid blocking the event loop.

        Args:
            field: The schema field name.
            module: The DSPy module instance.
            incident_text: The input text for prediction.

        Returns:
            Tuple of (field_name, predicted_value).
        """
        try:
            prediction = await asyncio.to_thread(module, incident_text=incident_text)
            value = getattr(prediction, field, None)
            logger.debug(f"[{field}] Predicted value: {value}")
            return field, value
        except Exception as e:
            logger.error(f"Error extracting field '{field}': {e}")
            return field, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default extraction service during app startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        None. Ensures the default service is available during app lifetime.
    """
    app_state = getattr(app, "state")

    try:
        service = AsyncHazardExtractionService.from_artifacts(allow_partial=True)
        app_state.extraction_service = service
        logger.info("Hazard extraction service initialized successfully.")
    except FileNotFoundError as e:
        logger.critical(f"Failed to load artifacts: {e}")
        app_state.extraction_service = None

    yield

    app_state.extraction_service = None


def get_service(request: Request) -> AsyncHazardExtractionService:
    """Dependency to retrieve the service from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The active AsyncHazardExtractionService instance.

    Raises:
        HTTPException: If the service was not initialized properly.
    """
    app_state = getattr(request.app, "state")
    service = getattr(app_state, "extraction_service", None)

    service = cast(AsyncHazardExtractionService | None, service)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized (artifacts missing).",
        )
    return service


app = FastAPI(
    title="Hazard Extraction Service",
    description="Expose optimized DSPy extractors over HTTP.",
    version="0.1.0",
    lifespan=lifespan,
)


async def extract_incident_to_json(
    incident_text: str,
    *,
    fields: Sequence[str] | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    stage: str = DEFAULT_STAGE,
) -> Dict[str, Any]:
    """Convenience helper to load artifacts, run extraction, and return JSON.

    Args:
        incident_text: Hebrew incident narrative.
        fields: Optional subset of fields to extract.
        artifact_root: Root directory containing saved DSPy programs.
        stage: Artifact stage subdirectory.

    Returns:
        Dict containing extracted fields keyed by alias.
    """
    service = AsyncHazardExtractionService.from_artifacts(
        artifact_root=artifact_root,
        stage=stage,
    )
    return await service.extract(incident_text, requested_fields=fields)


@app.post("/extract")
async def extract_endpoint(
        request: ExtractionRequest,
        service: Annotated[AsyncHazardExtractionService, Depends(get_service)]
) -> Dict[str, Any]:
    """Run Chain-of-Thought extraction for a single incident text.

    Args:
        request: Pydantic payload containing incident_text and optional fields.
        service: Injected extraction service instance.

    Returns:
        Dict containing a data field with extracted schema values.

    Raises:
        HTTPException: If the extraction process encounters an unhandled error.
    """
    try:
        result = await service.extract(
            request.incident_text,
            requested_fields=request.fields
        )
        return {"data": result}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Extraction process failed")
        raise HTTPException(status_code=500, detail="Internal extraction error")
