from pathlib import Path
from random import Random

from loguru import logger

from src.optimization.config import ARTIFACT_DIR, DEFAULT_WORKBOOK
from src.optimization.data import load_hazard_examples
from src.optimization.optimizer import HazardSchemaOptimizer

DEFAULT_SPLIT_RATIO = 0.6
SPLIT_SEED = 42


def main() -> None:
    """Load default data, split 60/40, and run the optimizer with canonical settings.

    Returns:
        None
    """
    logger.info(f"Loading workbook at {DEFAULT_WORKBOOK}")
    records = load_hazard_examples(Path(DEFAULT_WORKBOOK))

    splitter = Random(SPLIT_SEED)
    combined = list(records)
    splitter.shuffle(combined)
    split_idx = max(1, int(DEFAULT_SPLIT_RATIO * len(combined)))
    train_records = combined[:split_idx]
    val_records = combined[split_idx:]
    logger.info(
        f"Using {len(train_records)} training examples and {len(val_records)} "
        f"validation examples (seed={SPLIT_SEED})."
    )

    optimizer = HazardSchemaOptimizer(
        track_stats=True,
        artifact_dir=str(ARTIFACT_DIR),
    )
    program = optimizer.optimize_schema(
        train_records,
        val_records=val_records,
    )
    logger.success(f"Compilation complete. Program ready for inference: {program}")


if __name__ == "__main__":
    main()
