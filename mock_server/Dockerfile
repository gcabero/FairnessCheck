# Use slim Python image for minimal size
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv - fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY app.py .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
