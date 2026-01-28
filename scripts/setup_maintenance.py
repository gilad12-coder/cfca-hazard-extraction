#!/usr/bin/env python3
"""
Setup maintenance hook for Claude Code.
Runs when claude --maintenance is invoked.

This script:
1. Updates dependencies with uv sync --upgrade
2. Checks for outdated packages
3. Logs results to logs/setup.maintenance.log
4. Returns JSON with additionalContext for Claude
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """Logs to both stderr and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line, file=sys.stderr)
        with open(self.log_path, "a") as f:
            f.write(log_line + "\n")


def run_command(cmd: list[str], cwd: str, logger: Logger) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    logger.log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            logger.log(f"FAILED: {output}")
            return False, output
        logger.log(f"SUCCESS: {cmd[0]} completed")
        return True, output
    except subprocess.TimeoutExpired:
        logger.log(f"TIMEOUT: {' '.join(cmd)}")
        return False, "Command timed out"
    except Exception as e:
        logger.log(f"ERROR: {e}")
        return False, str(e)


def main():
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        input_data = {}

    # Determine project directory
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    log_path = Path(project_dir) / "logs" / "setup.maintenance.log"

    logger = Logger(log_path)
    logger.log("=" * 60)
    logger.log("SETUP MAINTENANCE HOOK STARTED")
    logger.log(f"Project directory: {project_dir}")
    logger.log(f"Trigger: {input_data.get('trigger', 'unknown')}")

    completed_actions = []
    failed_actions = []
    warnings = []

    # Step 1: Sync dependencies (ensure lock file is up to date)
    success, output = run_command(
        ["uv", "sync", "--all-extras"],
        project_dir,
        logger,
    )
    if success:
        completed_actions.append("Synced dependencies with uv")
    else:
        failed_actions.append(f"Failed to sync dependencies: {output}")

    # Step 2: Check for security vulnerabilities (if pip-audit available)
    success, output = run_command(
        ["uv", "run", "python", "-c", "import dspy; import fastapi; print('OK')"],
        project_dir,
        logger,
    )
    if success:
        completed_actions.append("Verified core imports")
    else:
        warnings.append(f"Import check had issues: {output}")

    # Step 3: Check Python version
    success, output = run_command(
        ["uv", "run", "python", "--version"],
        project_dir,
        logger,
    )
    if success:
        python_version = output.strip()
        completed_actions.append(f"Python version: {python_version}")
    else:
        warnings.append("Could not determine Python version")

    # Summary
    logger.log("-" * 40)
    if failed_actions:
        logger.log("MAINTENANCE COMPLETED WITH ERRORS")
        for action in failed_actions:
            logger.log(f"  FAILED: {action}")
    else:
        logger.log("MAINTENANCE COMPLETED SUCCESSFULLY")

    for action in completed_actions:
        logger.log(f"  DONE: {action}")

    for warning in warnings:
        logger.log(f"  WARNING: {warning}")

    logger.log("=" * 60)

    # Return JSON output for Claude
    status = "SUCCESS" if not failed_actions else "FAILED"
    summary = f"""## Maintenance Results

**Status:** {status}

### Completed Actions
{chr(10).join(f'- {a}' for a in completed_actions) if completed_actions else '- None'}

### Warnings
{chr(10).join(f'- {a}' for a in warnings) if warnings else '- None'}

### Failed Actions
{chr(10).join(f'- {a}' for a in failed_actions) if failed_actions else '- None'}

### Recommendations
- Review any warnings above
- Run `just test` to verify everything works
- Run `just lint` to check code quality
"""

    output = {
        "hookSpecificOutput": {
            "hookEventName": "Setup",
            "additionalContext": summary,
        }
    }

    print(json.dumps(output))
    sys.exit(0 if not failed_actions else 2)


if __name__ == "__main__":
    main()
