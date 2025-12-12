import re
from typing import Mapping, Optional

import dspy
from loguru import logger

from .config import TEACHER_LM

NULL_STRINGS = {"null"}


def _normalize_value(value):
    """Normalize values for comparison/logging.

    Args:
        value: Arbitrary object returned by the model or present in gold labels.

    Returns:
        Normalized representation with enums converted to strings, numerics cast to
        strings, nested lists processed recursively, and "null" strings mapped to None.
    """
    from enum import Enum

    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _extract_incident_text(example: dspy.Example) -> Optional[str]:
    """Return the incident text associated with an example, when available.

    Args:
        example: DSPy example containing inputs/labels.

    Returns:
        The incident_text string when present, otherwise None.
    """
    text = getattr(example, "incident_text", None)
    if text is not None:
        return text
    try:
        return example.inputs().toDict().get("incident_text")
    except Exception:  # noqa: BLE001
        return None


class LLMJudge:
    """Judge predictions by asking an LLM to score semantic adherence."""

    def __init__(
        self,
        field: str,
        spec: Mapping[str, object],
        lm: Optional[object] = None,
    ):
        """Initialize the judge with field metadata and an LM instance.

        Args:
            field: Schema field name being scored.
            spec: Specification dictionary from FIELD_SPECS.
            lm: Optional DSPy LM override.
        """

        self.field = field
        self.spec = spec
        self.lm = lm or dspy.LM(**TEACHER_LM)

    def _render_prompt(self, expected: object, predicted: object) -> str:
        """Create the grading prompt for a semantic similarity score.

        Args:
            expected: Reference value for the field.
            predicted: Model output for the field.

        Returns:
            str: Prompt instructing the LM to produce a similarity score.
        """
        requires = self.spec.get("requires")
        allowed = ", ".join(str(v) for v in self.spec.get("allowed_values", []))
        guidance = f"Precondition: {requires}.\n" if requires else ""
        return (
            "You are grading a field-level extraction for semantic agreement.\n"
            f"Field: {self.field}\n"
            f"Allowed values: {allowed}\n"
            f"{guidance}"
            "Return a decimal score between 0 and 1 inclusive on a single line:\n"
            "- 1 means the prediction conveys the exact same meaning and respects constraints.\n"
            "- 0 means it contradicts or refers to a different meaning.\n"
            "- Intermediate scores reflect partial semantic overlap (e.g., similar but not identical wording).\n"
            "Respond with only the numeric score.\n"
            f"Gold label: {expected}\n"
            f"Prediction: {predicted}"
        )

    def score_values(self, expected: object, predicted: object) -> float:
        """Return a semantic similarity score in [0, 1].

        Args:
            expected: Reference value for the field.
            predicted: Model output for the field.

        Returns:
            float: Score clipped to the [0, 1] range.
        """
        prompt = self._render_prompt(expected, predicted)
        response = self.lm(prompt)
        raw_text = str(getattr(response, "text", response)).strip()
        match = re.search(r"-?\d+(\.\d+)?", raw_text)
        if not match:
            return 0.0
        try:
            score = float(match.group())
        except ValueError:
            return 0.0
        return min(max(score, 0.0), 1.0)


class FieldFeedbackMetric:
    """Field-level GEPA metric that scores a single schema attribute."""

    def __init__(self, field: str, judge: LLMJudge):
        """Initialize a GEPA feedback metric.

        Args:
            field: Schema field name.
            judge: LLMJudge evaluating the field.
        """
        self.field = field
        self.judge = judge

    @staticmethod
    def _format_feedback(score: float, field: str, gold: object, predicted: object) -> str:
        if score >= 1.0:
            return (
                f"Score {score:.2f}. Field '{field}' matches the reference label. "
                "Maintain the same extraction strategy."
            )
        return (
            f"Score {score:.2f}. Field '{field}' is incorrect. "
            f"Expected {gold!r} but predicted {predicted!r}. "
            "Copy the gold enum literally or output null when unspecified."
        )

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ) -> dspy.Prediction:
        """Evaluate a prediction and emit GEPA feedback.

        Args:
            gold: Gold-labeled example.
            pred: Prediction under evaluation.
            trace: Optional DSPy trace.
            pred_name: Optional predictor identifier.
            pred_trace: Optional predictor trace.

        Returns:
            dspy.Prediction: Object containing score and textual feedback.
        """
        gold_value = _normalize_value(getattr(gold, self.field, None))
        pred_value = _normalize_value(getattr(pred, self.field, None))
        score = self.judge.score_values(gold_value, pred_value)
        feedback = self._format_feedback(score, self.field, gold_value, pred_value)
        logger.info(
            f"GEPA field metric -> field={self.field} score={score} pred_name={pred_name} feedback={feedback}"
        )
        return dspy.Prediction(score=score, feedback=feedback)


class FieldScalarMetric:
    """Field-level scalar metric consumed by optimization/evaluation loops."""

    def __init__(self, field: str, judge: LLMJudge):
        """Initialize the scalar metric.

        Args:
            field: Schema field name.
            judge: LLMJudge evaluating the field.
        """
        self.field = field
        self.judge = judge

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
    ) -> float:
        """Return the scalar score required by optimization loops.

        Args:
            example: Gold-labeled example.
            prediction: Model prediction to score.
            trace: Optional DSPy trace.

        Returns:
            float: Similarity score in [0, 1].
        """
        gold_value = _normalize_value(getattr(example, self.field, None))
        pred_value = _normalize_value(getattr(prediction, self.field, None))
        incident_text = _extract_incident_text(example)
        score = self.judge.score_values(gold_value, pred_value)
        logger.info(
            f"[{self.field}] LLM metric inputs\n"
            f"    incident={incident_text}\n"
            f"    gold={gold_value}\n"
            f"    pred={pred_value}"
        )
        logger.info(f"[{self.field}] LLM metric score={score}")
        return score


class ExactFieldFeedbackMetric:
    """Field-level GEPA metric that relies on literal equality (no LLM calls)."""

    def __init__(self, field: str):
        """Initialize an exact-match metric.

        Args:
            field: Schema field name.
        """
        self.field = field

    @staticmethod
    def _format_feedback(score: float, field: str, gold: object, predicted: object) -> str:
        if score >= 1.0:
            return (
                f"Score {score:.2f}. Field '{field}' matched exactly."
            )
        return (
            f"Score {score:.2f}. Field '{field}' must match exactly. "
            f"Expected {gold!r} but predicted {predicted!r}."
        )

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ) -> dspy.Prediction:
        """Evaluate a prediction using exact matching.

        Args:
            gold: Gold-labeled example.
            pred: Model prediction.
            trace: Optional DSPy trace.
            pred_name: Optional predictor identifier.
            pred_trace: Optional predictor trace.

        Returns:
            dspy.Prediction: Score/feedback object.
        """
        gold_value = _normalize_value(getattr(gold, self.field, None))
        pred_value = _normalize_value(getattr(pred, self.field, None))
        score = 1.0 if gold_value == pred_value else 0.0
        feedback = self._format_feedback(score, self.field, gold_value, pred_value)
        logger.info(
            f"GEPA exact metric -> field={self.field} score={score} pred_name={pred_name} feedback={feedback}"
        )
        return dspy.Prediction(score=score, feedback=feedback)


class ExactFieldScalarMetric:
    """Exact-match scalar metric for optimization/evaluation loops."""

    def __init__(self, field: str):
        """Initialize the exact-match scalar metric.

        Args:
            field: Schema field name.
        """
        self.field = field

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
    ) -> float:
        """Return 1.0 for exact matches, else 0.0.

        Args:
            example: Gold-labeled example.
            prediction: Model prediction.
            trace: Optional DSPy trace.

        Returns:
            float: Exact-match score.
        """
        gold_value = _normalize_value(getattr(example, self.field, None))
        pred_value = _normalize_value(getattr(prediction, self.field, None))
        incident_text = _extract_incident_text(example)
        score = 1.0 if gold_value == pred_value else 0.0
        logger.info(
            f"[{self.field}] Exact metric inputs\n"
            f"    incident={incident_text}\n"
            f"    gold={gold_value}\n"
            f"    pred={pred_value}"
        )
        logger.info(f"[{self.field}] Exact metric score={score}")
        return score
