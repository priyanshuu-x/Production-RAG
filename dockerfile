FROM python:3.11-slim

WORKDIR /app

# Install uv for dependency management (same tool used in development)
RUN pip install uv

# Copy dependency files first (better layer caching -- only reinstalls if these change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy the rest of the project
COPY . .

# Default port exposed (overridden per-service in docker-compose)
EXPOSE 8000
EXPOSE 8501