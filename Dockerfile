# Dockerfile for reproducible MLOps environment
# Steel Energy Prediction Pipeline

FROM python:3.11.3-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt requirements-lock.txt ./

# Install Python dependencies (use lock file for exact reproducibility)
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models mlruns

# Expose port for API
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "pytest", "tests/", "-v"]
