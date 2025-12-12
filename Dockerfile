# Use official Python runtime as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Set environment variables
ENV PYTHONUNBUFFERED=True

# Run the FastAPI app with Uvicorn (Cloud Run friendly)
CMD exec uvicorn src.service.extraction_service:app --host 0.0.0.0 --port ${PORT:-8080}
