set dotenv-load := true

# Show all available commands
default:
    just --list

# === Development Commands ===

# Install all dependencies with uv
setup:
    uv sync --all-extras

# Run the development server
serve:
    uv run uvicorn src.service.extraction_service:app --host 0.0.0.0 --port 8080 --reload

# Run tests
test *args:
    uv run pytest {{args}}

# Run linting (black + ruff)
lint:
    uv run black src tests
    uv run ruff check src tests

# Format code
fmt:
    uv run black src tests
    uv run ruff check --fix src tests

# Run the optimizer
optimize:
    uv run python -m src.optimization.main

# === Claude Code Commands ===

# Deterministic init (runs setup hook only)
cli:
    claude --model opus --dangerously-skip-permissions --init

# Deterministic maintenance (runs maintenance hook only)
clm:
    claude --model opus --dangerously-skip-permissions --maintenance

# Agentic installation (hook + prompt + report)
cli-install:
    claude --model opus --dangerously-skip-permissions --init "/install"

# Interactive agentic installation (human-in-the-loop)
cli-install-hil:
    claude --model opus --dangerously-skip-permissions --init "/install true"

# Agentic maintenance (hook + prompt + report)
cli-maintain:
    claude --model opus --dangerously-skip-permissions --maintenance "/maintain"

# === Docker Commands ===

# Build Docker image
docker-build:
    docker build -t cfca-extraction .

# Run Docker container
docker-run:
    docker run -p 8080:8080 --env-file .env cfca-extraction

# === Cleanup ===

# Reset all build artifacts
reset:
    rm -rf .venv
    rm -rf logs/*.log
    rm -rf __pycache__
    rm -rf .pytest_cache
    rm -rf .ruff_cache
