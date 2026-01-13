# Use slim Python image for minimal size
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv - fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies using uv
RUN uv pip install --system --no-cache -e .

# Set entrypoint to fairness-check command
ENTRYPOINT ["fairness-check"]

# Default command shows help
CMD ["--help"]
