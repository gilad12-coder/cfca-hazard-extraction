# Repository Guidelines

## Project Structure & Module Organization
Use the committed `clean_venv/` folder only as the pinned Python 3.13 runtime; regenerate it when dependency pins change. Keep corpora, transcripts, and evaluation exports in `data/`. Baseline prompt seeds live in `start_messages.xlsx`; add sheets as `<scenario>_<yyyymmdd>` and mention them in your PR. Source code lives flat under `src/`—key modules include `config.py`, `constants.py`, `signatures.py`, `optimizer.py`, and `main.py` (DSPy pipeline). Mirror the layout under `tests/` and keep notebooks in `notebooks/`, promoting stable logic into modules before merging.

## Build, Test, and Development Commands
- `source clean_venv/bin/activate` – load the shared Python 3.13 env.
- `python -m pip install -r requirements.txt` – sync dependencies; regenerate the file when packages change.
- `python -m pytest` – run the suite; add `-k name` for focused checks.
- `python -m src.<module>` – pattern for CLIs you add (document actual module in README).

## Coding Style & Naming Conventions
Target Python 3.13, PEP 8, and 4-space indentation with 88-character lines. Use type hints, keep modules lowercase, functions and variables `snake_case`, classes `PascalCase`, and constants `SCREAMING_SNAKE_CASE`. Store prompt templates as triple-quoted strings with comments referencing their spreadsheet sheet. Run `python -m black src tests` and, when available, `python -m ruff check src tests` before committing.

## Testing Guidelines
Author `pytest` cases that mirror module names in `tests/` (e.g., `tests/scoring/test_optimizer.py`). Parameterize across prompt variants and keep deterministic fixtures in `data/fixtures/`. Capture expected assistant responses as golden files when prompts change. Run `python -m pytest --maxfail=1 --disable-warnings` before every push and explain any skipped checks in your PR.

## Commit & Pull Request Guidelines
Write conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`) with imperative subjects ≤72 characters and avoid mixing prompt tweaks with code refactors. PRs must summarize intent, expected metric impact, linked issues, and include screenshots or transcript snippets for prompt changes. List the commands you ran (`python -m pytest`, manual evaluations) and call out follow-up tasks.

## Security & Configuration Tips
Never commit API keys or model credentials; load them from a local `.env` and document required variables in the PR. Anonymize any customer references in `start_messages.xlsx`. Keep large experiment exports out of Git—store locations in `data/README.md`. When rebuilding `clean_venv/`, confirm `pyvenv.cfg` still targets Python 3.13.
