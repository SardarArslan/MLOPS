# Use a slim Python base image
FROM python:3.12-slim

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (much faster than pip)
RUN uv sync

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/processed /app/models

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]