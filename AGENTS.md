# Repository Guidelines

## Development Setup

### Quick Start

```bash
# Install just (command runner)
brew install just  # or: cargo install just

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
just setup
```

### Development Commands

```bash
just setup   # Install all dependencies with uv
just serve   # Start development server (port 8080)
just test    # Run pytest
just lint    # Run black + ruff
just fmt     # Format code with auto-fix
```

### Claude Code Integration

```bash
just cli              # Deterministic init (runs setup hook)
just cli-install      # Agentic installation (hook + prompt + report)
just cli-install-hil  # Interactive installation (human-in-the-loop)
just clm              # Deterministic maintenance (runs maintenance hook)
just cli-maintain     # Agentic maintenance (hook + prompt + report)
```

Run `just --list` to see all available commands.

## Project Structure & Module Organization

Source code lives under `src/`:
- `src/optimization/` - DSPy optimizer pipeline (config, signatures, optimizer, metrics)
- `src/service/` - FastAPI extraction service

Key files:
- `pyproject.toml` - Dependencies and tool configuration (managed by uv)
- `justfile` - Command runner (replaces manual commands)
- `Dockerfile` - Cloud Run deployment (uses uv)

Data and artifacts:
- `data/` - Training data (`start_messages.xlsx`)
- `artifacts/` - Pre-trained DSPy programs

Mirror the layout under `tests/` and keep notebooks in `notebooks/`, promoting stable logic into modules before merging.

## Build, Test, and Development Commands

All commands are run via `just`:

```bash
just setup     # uv sync --all-extras
just serve     # uv run uvicorn src.service.extraction_service:app --reload
just test      # uv run pytest (add args like -k name for focused checks)
just lint      # uv run black + ruff check
just fmt       # uv run black + ruff check --fix
just optimize  # uv run python -m src.optimization.main
```

For one-off commands, use `uv run`:
```bash
uv run python -m src.<module>
```

## Coding Style & Naming Conventions

Target Python 3.13, PEP 8, and 4-space indentation with 88-character lines. Use type hints, keep modules lowercase, functions and variables `snake_case`, classes `PascalCase`, and constants `SCREAMING_SNAKE_CASE`. Store prompt templates as triple-quoted strings with comments referencing their spreadsheet sheet.

Run `just lint` before committing.

## Testing Guidelines

Author `pytest` cases that mirror module names in `tests/` (e.g., `tests/scoring/test_optimizer.py`). Parameterize across prompt variants and keep deterministic fixtures in `data/fixtures/`. Capture expected assistant responses as golden files when prompts change.

Run `just test` before every push and explain any skipped checks in your PR.

## Commit & Pull Request Guidelines

Write conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`) with imperative subjects ≤72 characters and avoid mixing prompt tweaks with code refactors. PRs must summarize intent, expected metric impact, linked issues, and include screenshots or transcript snippets for prompt changes. List the commands you ran (`just test`, manual evaluations) and call out follow-up tasks.

## Security & Configuration Tips

Never commit API keys or model credentials; load them from a local `.env` and document required variables in the PR. Anonymize any customer references in `start_messages.xlsx`. Keep large experiment exports out of Git—store locations in `data/README.md`.

## Docker Deployment

```bash
just docker-build  # Build Docker image
just docker-run    # Run container locally
```

The Dockerfile uses uv for fast, reproducible builds. Deploys to Google Cloud Run.
