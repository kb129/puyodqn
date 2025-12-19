# Use the official Python 3.11 image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy uv project files first for better caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# Copy only necessary source files
COPY ai/ ./ai/
COPY game_state/ ./game_state/
COPY server/ ./server/
COPY utils/ ./utils/
COPY cli.py ./

# Create necessary directories
RUN mkdir -p models logs

# Expose port
EXPOSE 8000

# Use uv to run the application
CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]