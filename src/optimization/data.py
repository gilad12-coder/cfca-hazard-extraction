from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import dspy
import pandas as pd

from .constants import FIELD_EVENT_DATE, FIELD_EVENT_TIME, FIELDS, NULL_LITERAL
from .schema import HazardReport


@dataclass
class HazardExample:
    """Represents a training or evaluation example for a single hazard report."""

    text: str
    labels: Mapping[str, object]

    def as_dspy_kwargs(self) -> Dict[str, object]:
        """Return keyword arguments consumable by DSPy signatures.

        Args:
            None.

        Returns:
            Dict containing ``incident_text`` (context is currently ignored).
        """
        return {"incident_text": self.text}

    def hazard_report(self) -> HazardReport:
        """Return the labels materialized as a Pydantic ``HazardReport`` instance.

        Args:
            None.

        Returns:
            Validated ``HazardReport`` populated with all schema fields.
        """
        payload: Dict[str, object] = {
            field: self.labels.get(field, NULL_LITERAL) for field in FIELDS
        }
        return HazardReport(**payload)


def make_example(example: HazardExample, field: str) -> dspy.Example:
    """Create a DSPy example for a given field.

    Args:
        example: Source hazard record with labels.
        field: Field name to supervise.

    Returns:
        dspy.Example: Example configured for the requested field.
    """
    outputs = {field: example.labels.get(field, NULL_LITERAL)}
    return dspy.Example(**example.as_dspy_kwargs(), **outputs)


def make_schema_example(example: HazardExample) -> dspy.Example:
    """Create a DSPy example containing the entire schema.

    Args:
        example: Labeled hazard record.

    Returns:
        dspy.Example: Example with outputs for every schema field.
    """
    sample = dspy.Example(**example.as_dspy_kwargs(), report=example.hazard_report())
    return sample.with_inputs("incident_text")


def dataset_from_records(
    records: Iterable[HazardExample], field: str
) -> List[dspy.Example]:
    """Convert project records into DSPy Examples for a specific field.

    Args:
        records: Hazard examples to convert.
        field: Target field name.

    Returns:
        list[dspy.Example]: Examples with inputs/outputs bound to the field.
    """
    return [make_example(record, field).with_inputs("incident_text") for record in records]


def schema_dataset_from_records(
    records: Iterable[HazardExample],
) -> List[dspy.Example]:
    """Convert hazard records into DSPy Examples covering the full schema.

    Args:
        records: Hazard examples to convert.

    Returns:
        list[dspy.Example]: Examples with the full HazardReport output.
    """
    return [make_schema_example(record) for record in records]


def load_hazard_examples(
    path: Path,
    *,
    text_field: str = "incident_text",
) -> List[HazardExample]:
    """Load hazard examples from an Excel workbook containing columnar labels.

    Args:
        path: Filesystem path to the Excel workbook.
        text_field: Column containing the raw incident narrative. All schema labels are
            expected to be stored in dedicated columns matching ``src.constants.FIELDS``.

    Returns:
        List of ``HazardExample`` instances populated from the workbook rows.
    """
    df = pd.read_excel(path)
    if text_field in df.columns:
        text_series = df[text_field]
    else:
        escaped_field = text_field.replace("_", "\\_")
        if escaped_field in df.columns:
            text_series = df[escaped_field]
        else:
            raise ValueError(f"Workbook must contain '{text_field}' column.")

    labels_records = _labels_from_schema_columns(df)

    texts = text_series.fillna("").astype(str).tolist()
    if len(texts) != len(labels_records):
        raise ValueError("Mismatch between incident_text rows and label rows.")

    return [
        HazardExample(text=text, labels=labels)
        for text, labels in zip(texts, labels_records)
    ]


def _normalize_labels(raw: Mapping[str, object]) -> Dict[str, object]:
    """Normalize workbook labels and split comma-delimited symptom strings.

    Args:
        raw: Original labels mapping pulled directly from the workbook row.

    Returns:
        dict: Labels dictionary with ``symptoms`` coerced into a list.
    """
    labels = dict(raw)
    labels["symptoms"] = _parse_symptom_values(labels.get("symptoms"))
    return labels


def _labels_from_schema_columns(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Extract normalized label dictionaries from columnar schema data.

    Args:
        df: DataFrame that must include one column per schema field.

    Returns:
        list[dict[str, object]]: Sanitized label mappings ready for schema validation.
    """
    missing = [field for field in FIELDS if field not in df.columns]
    if missing:
        raise ValueError(f"Workbook is missing required schema columns: {missing}")

    records = df[list(FIELDS)].to_dict(orient="records")
    normalized: List[Dict[str, object]] = []
    for record in records:
        sanitized = {
            field: _clean_cell_value(field, record.get(field)) for field in FIELDS
        }
        normalized.append(_normalize_labels(sanitized))
    return normalized


def _clean_cell_value(field: str, value: object) -> object:
    """Return a consistent literal for downstream schema validation.

    Args:
        field: Schema field name whose constraints influence conversion.
        value: Raw value drawn directly from the workbook cell.

    Returns:
        object: Normalized literal (string/int/date/time) or the ``NULL_LITERAL`` sentinel.
    """
    if _is_missing(value):
        return NULL_LITERAL
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if field == FIELD_EVENT_TIME:
            value = value.time()
        elif field == FIELD_EVENT_DATE:
            return value.date().isoformat()
        else:
            return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.replace(microsecond=0).isoformat()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else NULL_LITERAL
    return value


def _is_missing(value: object) -> bool:
    """Identify NaN/empty/null-like literals.

    Args:
        value: Arbitrary workbook cell value.

    Returns:
        bool: ``True`` when the value should be treated as missing.
    """
    if value in (None, "", NULL_LITERAL):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _parse_symptom_values(value: object) -> List[object]:
    """Convert comma-delimited symptom strings into a list of literals.

    Args:
        value: Workbook cell that may contain a string, list, tuple, or null-like literal.

    Returns:
        list[object]: List of cleaned symptom tokens.
    """
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        canonical = value.replace(";", ",").replace("\n", ",")
        items = [token.strip() for token in canonical.split(",")]
    else:
        items = [value]

    return [item for item in items if not _is_missing(item)]
