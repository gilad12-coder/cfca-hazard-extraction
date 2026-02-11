# Use official Python runtime as base image
FROM python:3.13-slim

# Install uv for faster, reproducible builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (production only, no dev deps)
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/

# Copy Gemini artifact directory
COPY artifacts_gemini/ ./artifacts_gemini/

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV ARTIFACTS_DIR=/app/artifacts_gemini
ENV USE_MIXED_PROVIDERS=false

# Run the FastAPI app with Uvicorn (Cloud Run friendly)
CMD ["uv", "run", "uvicorn", "src.service.extraction_service:app", "--host", "0.0.0.0", "--port", "8080"]
