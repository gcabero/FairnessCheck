"""
Simple mock classifier REST service for testing fairness-check CLI tool.

This service provides a /classify endpoint that returns random binary predictions.
Built with FastAPI for high performance and async support.
"""

import random
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Mock Classifier API",
    version="1.0",
    description="High-performance mock classifier for fairness testing"
)


# Request/Response models
class ClassifyRequest(BaseModel):
    features: Any


class ClassifyResponse(BaseModel):
    inference: int
    features: Any
    note: str | None = None


class InfoResponse(BaseModel):
    service: str
    version: str
    endpoints: dict[str, str]


class HealthResponse(BaseModel):
    status: str


@app.get("/", response_model=InfoResponse)
async def home():
    """Home page with API information."""
    return {
        "service": "Mock Classifier API",
        "version": "1.0",
        "endpoints": {
            "/classify": "POST - Classify features and return prediction",
            "/classify/random": "POST - Random predictions for testing",
            "/classify/biased": "POST - Biased predictions for testing",
            "/health": "GET - Health check"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """
    Classify endpoint that returns random predictions.

    Expects JSON: {"features": <any_value>}
    Returns JSON: {"inference": 0 or 1}
    """
    # Return random binary prediction
    inference = random.choice([0, 1])

    return {
        "inference": inference,
        "features": request.features
    }


@app.post("/classify/random", response_model=ClassifyResponse)
async def classify_random(request: ClassifyRequest):
    """
    Classify endpoint with random predictions (same as /classify).

    Expects JSON: {"features": <any_value>}
    Returns JSON: {"inference": 0 or 1}
    """
    # Return random binary prediction
    inference = random.choice([0, 1])

    return {
        "inference": inference,
        "features": request.features
    }


@app.post("/classify/biased", response_model=ClassifyResponse)
async def classify_biased(request: ClassifyRequest):
    """
    Biased classifier for testing fairness issues.

    This endpoint demonstrates a biased classifier that might fail fairness tests.
    Always returns positive predictions.
    """
    # Always return positive class (1) - this is intentionally biased!
    return {
        "inference": 1,
        "features": request.features,
        "note": "This is an intentionally biased endpoint for testing"
    }


if __name__ == "__main__":
    import uvicorn
    # Run with uvicorn for production-grade performance
    # Supports multiple workers and async requests
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    )
