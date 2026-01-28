import json
import os
import random
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from loguru import logger
from pydantic import ValidationError

from .config import FIELD_BEHAVIORS, INFERENCE_LM, TEACHER_LM
from .constants import FIELD_SPECS, FIELDS
from .data import HazardExample, dataset_from_records
from .metrics import (
    ExactFieldFeedbackMetric,
    ExactFieldScalarMetric,
    FieldFeedbackMetric,
    FieldScalarMetric,
    LLMJudge,
)
from .schema import HazardReport
from .signatures import FIELD_SIGNATURES

FIELD_METHOD_LLM = "llm"
FIELD_METHOD_EXACT = "exact"
ALLOWED_FIELD_METHODS = {FIELD_METHOD_LLM, FIELD_METHOD_EXACT}

FIELD_MODE_OPTIMIZE = "optimize"
FIELD_MODE_EVALUATE = "evaluate"
ALLOWED_FIELD_MODES = {FIELD_MODE_OPTIMIZE, FIELD_MODE_EVALUATE}


class HazardSchemaProgram(dspy.Module):
    """DSPy module that either predicts the schema monolithically or via field ensembles."""

    def __init__(
        self,
        signature: Optional[type] = None,
        field_modules: Optional[Mapping[str, dspy.Module]] = None,
    ):
        """Initialize either a monolithic or ensemble-style program."""
        super().__init__()
        if signature is None and field_modules is None:
            raise ValueError("Provide either a signature or field_modules.")
        if signature is not None and field_modules is not None:
            raise ValueError("Cannot specify both signature and field_modules.")
        self.signature = signature
        self.field_modules = dict(field_modules) if field_modules else None
        if signature is not None:
            self.extract = dspy.ChainOfThought(signature)

    def forward(self, incident_text: str) -> dspy.Prediction:
        """Run extraction for a single hazard report.

        Args:
            incident_text: Hebrew incident narrative describing the hazard.

        Returns:
            dspy.Prediction containing either a HazardReport or per-field outputs.
        """
        if self.field_modules is None:
            return self.extract(incident_text=incident_text)

        outputs: Dict[str, object] = {}
        for field, module in self.field_modules.items():
            prediction = module(incident_text=incident_text)
            outputs[field] = getattr(prediction, field, None)

        try:
            hazard_report = HazardReport(**outputs)
            prediction_payload = dict(outputs)
            prediction_payload["report"] = hazard_report
            return dspy.Prediction(**prediction_payload)
        except ValidationError as exc:
            logger.warning(f"Failed to assemble HazardReport from field outputs: {exc}")
            return dspy.Prediction(**outputs)

    def field_programs(self) -> Mapping[str, dspy.Module]:
        """Return the optimized field-level modules when running in ensemble mode.

        Returns:
            Mapping of field names to DSPy modules.
        """
        return self.field_modules or {}


class HazardFieldProgram(dspy.Module):
    """DSPy module that predicts a single schema field."""

    def __init__(self, signature: type, field: str):
        """Initialize a field-specific program.

        Args:
            signature: DSPy signature tailored to the field.
            field: Name of the schema field.
        """
        super().__init__()
        self.field = field
        self.extract = dspy.ChainOfThought(signature)

    def forward(self, incident_text: str) -> dspy.Prediction:
        """Predict a single field value.

        Args:
            incident_text: Hebrew incident narrative.

        Returns:
            dspy.Prediction containing the field output.
        """
        return self.extract(incident_text=incident_text)


class HazardSchemaOptimizer:
    """Compile per-field extractors using GEPA."""

    def __init__(
        self,
        *,
        task_lm: Optional[object] = None,
        reflection_lm: Optional[object] = None,
        judge_lm: Optional[object] = None,
        track_stats: bool = False,
        gepa_log_dir: Optional[str] = None,
        fields: Optional[Sequence[str]] = None,
        artifact_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Configure GEPA for schema optimization.

        Args:
            task_lm: LM used for inference/prediction.
            reflection_lm: LM used for GEPA reflections.
            judge_lm: LM used by field-level judges.
            track_stats: Whether to keep optimizer diagnostics.
            gepa_log_dir: Root directory for GEPA logs.
            fields: Optional subset of fields to optimize.
            artifact_dir: Directory used to persist optimized programs.
        """
        self.task_lm = task_lm or dspy.LM(**INFERENCE_LM)
        dspy.configure(lm=self.task_lm)
        dspy.settings.configure(num_threads=1)
        self.reflection_lm = reflection_lm or dspy.LM(**TEACHER_LM)
        self.judge_lm = judge_lm or dspy.LM(**TEACHER_LM)
        self.track_stats = track_stats
        self.gepa_log_dir = gepa_log_dir
        self.field_behaviors = self._resolve_field_behaviors(fields)
        self.artifact_dir = artifact_dir or os.path.join(os.getcwd(), "artifacts")
        self.seed = seed
        os.makedirs(self.artifact_dir, exist_ok=True)

    def optimize_schema(
        self,
        train_records: Sequence[HazardExample],
        val_records: Optional[Sequence[HazardExample]] = None,
    ) -> HazardSchemaProgram:
        """Compile per-field predictors and assemble them into a schema-level program.

        Args:
            train_records: Labeled hazard examples for training.
            val_records: Optional hold-out examples (merged with train_records for splitting).
        """
        records = list(train_records)
        if val_records:
            records.extend(val_records)
        if len(records) < 3:
            raise ValueError("Need at least three labeled hazard records to optimize.")

        field_programs: Dict[str, dspy.Module] = {}
        summary: Dict[str, Dict[str, object]] = {}

        for field, behavior in self.field_behaviors.items():
            method = behavior.get("metric", FIELD_METHOD_LLM)
            mode = behavior.get("mode", FIELD_MODE_OPTIMIZE)
            logger.info(
                f"[{field}] Starting with method='{method}' mode='{mode}'."
            )
            examples = dataset_from_records(records, field)
            if len(examples) < 3:
                logger.warning(f"[{field}] Skipping; fewer than 3 labeled examples.")
                continue
            try:
                trainset, valset, testset = self._split_field_examples(examples)
            except ValueError as exc:
                logger.warning(f"[{field}] {exc}")
                continue
            logger.info(
                f"[{field}] Split sizes -> train={len(trainset)} val={len(valset)} test={len(testset)}."
            )

            signature_cls = FIELD_SIGNATURES.get(field)
            if signature_cls is None:
                raise ValueError(f"No signature registered for field '{field}'.")
            base_program = HazardFieldProgram(signature_cls, field)

            if method == FIELD_METHOD_LLM:
                judge = LLMJudge(field, FIELD_SPECS[field], lm=self.judge_lm)
                gepa_metric = FieldFeedbackMetric(field, judge)
                scalar_metric = FieldScalarMetric(field, judge)
            elif method == FIELD_METHOD_EXACT:
                gepa_metric = ExactFieldFeedbackMetric(field)
                scalar_metric = ExactFieldScalarMetric(field)
            else:
                raise ValueError(f"Unsupported metric method '{method}' for field '{field}'.")

            baseline_score = self._evaluate_field_program(
                field, base_program, testset, scalar_metric
            )
            logger.info(f"[{field}] Baseline score: {baseline_score:.4f}")

            if mode == FIELD_MODE_EVALUATE:
                field_programs[field] = base_program
                summary[field] = {
                    "execution_mode": mode,
                    "baseline_score": baseline_score,
                    "gepa_optimized_score": None,
                }
                logger.info(f"[{field}] Evaluation-only mode complete.")
                continue

            gepa_program = self._run_gepa_optimizer(
                field=field,
                program=base_program,
                trainset=trainset,
                valset=valset,
                gepa_metric=gepa_metric,
            )

            gepa_score = self._evaluate_field_program(
                field, gepa_program, testset, scalar_metric
            )
            logger.info(f"[{field}] GEPA score: {gepa_score:.4f}")

            field_programs[field] = gepa_program
            summary[field] = {
                "execution_mode": mode,
                "baseline_score": baseline_score,
                "gepa_optimized_score": gepa_score,
            }
            logger.info(f"[{field}] GEPA optimization complete.")

        if not field_programs:
            raise RuntimeError("No field programs were successfully optimized.")

        if summary:
            self._write_summary(summary)
        else:
            logger.warning("No summary data generated; summary.json not written.")

        return HazardSchemaProgram(field_modules=field_programs)

    @staticmethod
    def _resolve_field_behaviors(
        include_fields: Optional[Sequence[str]],
    ) -> Dict[str, Dict[str, str]]:
        """Determine the metric/mode behavior for each field.

        Args:
            include_fields: Optional subset of field names to process.

        Returns:
            dict where each field maps to a dict containing ``metric`` and ``mode`` keys.
        """
        include_set = set(include_fields) if include_fields else None
        mapping: Dict[str, Dict[str, str]] = {}
        plan = FIELD_BEHAVIORS or {}
        for field, behavior in plan.items():
            if include_set and field not in include_set:
                continue
            metric = str(behavior.get("metric", FIELD_METHOD_LLM)).lower()
            mode = str(behavior.get("mode", FIELD_MODE_OPTIMIZE)).lower()
            if metric not in ALLOWED_FIELD_METHODS:
                raise ValueError(
                    f"Unsupported metric method '{metric}' for field '{field}'. "
                    f"Allowed: {sorted(ALLOWED_FIELD_METHODS)}."
                )
            if mode not in ALLOWED_FIELD_MODES:
                raise ValueError(
                    f"Unsupported execution mode '{mode}' for field '{field}'. "
                    f"Allowed: {sorted(ALLOWED_FIELD_MODES)}."
                )
            mapping[field] = {"metric": metric, "mode": mode}
            logger.info(f"[{field}] Registered metric='{metric}' mode='{mode}'.")

        if include_set:
            for field in include_set:
                if field not in mapping:
                    mapping[field] = {
                        "metric": FIELD_METHOD_LLM,
                        "mode": FIELD_MODE_OPTIMIZE,
                    }
                    logger.info(
                        f"[{field}] Defaulting behavior to metric='{FIELD_METHOD_LLM}' mode='{FIELD_MODE_OPTIMIZE}'."
                    )

        if not mapping:
            for field in FIELDS:
                mapping[field] = {
                    "metric": FIELD_METHOD_LLM,
                    "mode": FIELD_MODE_OPTIMIZE,
                }
        return mapping

    def _split_field_examples(
        self, examples: Sequence[dspy.Example]
    ) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
        """Split examples into train/validation/test folds.

        Args:
            examples: Field-specific DSPy examples.

        Returns:
            tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
                A tuple of (train, validation, test) splits respecting the 50/17/33 ratio.
        """
        if len(examples) < 3:
            raise ValueError("Need at least 3 examples to split into train/val/test.")
        rng = random.Random(self.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = max(1, int(round(n * 0.5)))
        if train_end >= n - 1:
            train_end = n - 2
        val_size = max(1, int(round(n * 0.17)))
        val_end = train_end + val_size
        if val_end >= n:
            val_end = n - 1
        if train_end >= val_end:
            train_end = max(1, val_end - 1)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        if not val:
            val = [train[-1]]
        if not test:
            test = [val[-1]]

        return train, val, test

    def _run_gepa_optimizer(
        self,
        *,
        field: str,
        program: HazardFieldProgram,
        trainset: Sequence[dspy.Example],
        valset: Sequence[dspy.Example],
        gepa_metric,
    ) -> dspy.Module:
        """Run GEPA for a single field.

        Args:
            field: Field being optimized.
            program: Starting DSPy module for the field.
            trainset: Training examples for the field.
            valset: Validation examples for the field.
            gepa_metric: Metric callable for GEPA.

        Returns:
            GEPA-optimized DSPy module.
        """
        gepa = GEPA(
            metric=gepa_metric,
            auto="heavy",
            reflection_lm=self.reflection_lm,
            track_stats=self.track_stats,
            log_dir=self._scoped_log_dir(self.gepa_log_dir, field),
            candidate_selection_strategy="pareto",
            reflection_minibatch_size=10,
            num_threads=8,
            max_merge_invocations=10,
            use_merge=True,
        )
        gepa_compiled = gepa.compile(
            student=program,
            trainset=trainset,
            valset=valset,
        )
        logger.info(f"[{field}] GEPA compilation complete.")
        self._save_field_program(field, gepa_compiled, stage="gepa")
        return gepa_compiled

    @staticmethod
    def _scoped_log_dir(root: Optional[str], field: str) -> Optional[str]:
        """Return a field-scoped logging directory.

        Args:
            root: Base log directory.
            field: Field name.

        Returns:
            String path or None when no root provided.
        """
        if not root:
            return None
        return os.path.join(root, field)

    def _save_field_program(self, field: str, program: dspy.Module, stage: str) -> None:
        """Persist the optimized program for a field.

        Args:
            field: Field name.
            program: Optimized DSPy module.
            stage: Identifier for the optimization stage.
        """
        if not self.artifact_dir:
            return
        field_dir = os.path.join(self.artifact_dir, field, stage)
        os.makedirs(field_dir, exist_ok=True)
        program.save(field_dir, save_program=True)
        logger.info(
            f"Saved optimized program for '{field}' stage '{stage}' to {field_dir} (load with dspy.load)."
        )

    @staticmethod
    def _evaluate_field_program(
        field: str,
        program: dspy.Module,
        devset: Sequence[dspy.Example],
        metric,
    ) -> float:
        """Evaluate a DSPy program on a devset and return a score in [0, 1].

        Args:
            field: Schema field name being evaluated.
            program: DSPy module to evaluate.
            devset: Validation examples for the field.
            metric: Metric callable compatible with ``dspy.Evaluate``.

        Returns:
            float: Normalized score between 0 and 1.
        """
        evaluator = Evaluate(
            devset=devset,
            metric=metric,
            display_progress=True,
            num_threads=1,
        )
        result = evaluator(program)
        score_pct = result.score
        score = score_pct / 100.0
        logger.info(
            f"[{field}] dspy.Evaluate score={score:.4f} ({score_pct:.2f}%) on {len(devset)} examples."
        )
        return score

    def _write_summary(self, summary: Dict[str, Dict[str, object]]) -> None:
        """Persist a merged summary of optimization results to disk.

        Args:
            summary: Mapping of field name to summary metrics for the current run.

        Returns:
            None. Writes/merges into ``summary.json`` under the artifact directory.
        """
        summary_path = Path(self.artifact_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        merged: Dict[str, Dict[str, object]] = {}
        if summary_path.exists():
            try:
                merged = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"Existing summary at {summary_path} is invalid JSON; overwriting.")
                merged = {}
        merged.update(summary)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
        logger.info(f"Wrote optimization summary to {summary_path} (merged {len(summary)} fields).")
