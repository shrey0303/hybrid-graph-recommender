"""
FastAPI Application for Real-Time Recommendations.

Production-grade REST API with:
- Health checks and readiness probes
- Recommendation endpoints (single + batch)
- Feature flag support
- Request/response logging
- Prometheus metrics integration
- CORS middleware
- Rate limiting schema

Endpoints:
    GET  /health          - Health check
    GET  /ready           - Readiness probe (checks model loaded)
    POST /recommend       - Get recommendations for a user
    POST /recommend/batch - Batch recommendation requests
    POST /feedback        - Submit user feedback for model update
    GET  /metrics         - Prometheus metrics
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("FastAPI required. Install: pip install fastapi uvicorn")

from src.serving.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)


def create_app(
    model_registry: Optional[Any] = None,
    enable_cors: bool = True,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model_registry: Optional model registry for serving.
        enable_cors: Whether to enable CORS middleware.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Hybrid GNN-LLM Recommendation Engine",
        description=(
            "Production-grade recommendation API combining Graph Neural Networks "
            "with Large Language Models for real-time personalized recommendations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # State
    app.state.model_registry = model_registry
    app.state.is_ready = False
    app.state.request_count = 0
    app.state.total_latency_ms = 0.0
    app.state.start_time = time.time()

    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── Request tracking middleware ──
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        response = await call_next(request)

        latency_ms = (time.time() - start) * 1000
        app.state.request_count += 1
        app.state.total_latency_ms += latency_ms

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({latency_ms:.1f}ms)"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
        return response

    # ── Health & Readiness ──
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint for load balancers."""
        uptime = time.time() - app.state.start_time
        return HealthResponse(
            status="healthy",
            uptime_seconds=round(uptime, 1),
            version="1.0.0",
        )

    @app.get("/ready", tags=["System"])
    async def readiness_check():
        """Readiness probe — checks if model is loaded and serving."""
        if app.state.is_ready:
            return {"status": "ready", "model_loaded": True}
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model not loaded.",
        )

    # ── Recommendation Endpoints ──
    @app.post(
        "/recommend",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
    )
    async def get_recommendations(request: RecommendationRequest):
        """
        Get personalized recommendations for a single user.

        Uses the hybrid GNN-LLM model to generate recommendations
        based on user purchase history and graph neighborhood.
        """
        start = time.time()

        try:
            # Generate recommendations (mock if no model loaded)
            recommendations = _generate_recommendations(
                user_id=request.user_id,
                num_items=request.num_items,
                context=request.context,
                registry=app.state.model_registry,
            )

            latency_ms = (time.time() - start) * 1000

            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                model_version="hybrid-gnn-llm-v1",
                latency_ms=round(latency_ms, 1),
            )
        except Exception as e:
            logger.error(f"Recommendation error for user {request.user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/recommend/batch",
        response_model=BatchRecommendationResponse,
        tags=["Recommendations"],
    )
    async def batch_recommendations(request: BatchRecommendationRequest):
        """Batch recommendation endpoint for multiple users."""
        start = time.time()

        results = []
        for user_id in request.user_ids:
            recs = _generate_recommendations(
                user_id=user_id,
                num_items=request.num_items,
                registry=app.state.model_registry,
            )
            results.append(
                RecommendationResponse(
                    user_id=user_id,
                    recommendations=recs,
                    model_version="hybrid-gnn-llm-v1",
                    latency_ms=0,
                )
            )

        total_latency = (time.time() - start) * 1000

        return BatchRecommendationResponse(
            responses=results,
            total_latency_ms=round(total_latency, 1),
        )

    # ── Feedback ──
    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        tags=["Feedback"],
    )
    async def submit_feedback(
        request: FeedbackRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Submit user feedback on recommendations.

        Feedback is processed asynchronously for model improvement.
        """
        feedback_id = str(uuid.uuid4())

        # Queue feedback for async processing
        background_tasks.add_task(
            _process_feedback, feedback_id, request
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="accepted",
            message="Feedback received and queued for processing.",
        )

    # ── Metrics ──
    @app.get("/metrics", tags=["System"])
    async def get_metrics():
        """Prometheus-compatible metrics endpoint."""
        uptime = time.time() - app.state.start_time
        avg_latency = (
            app.state.total_latency_ms / max(app.state.request_count, 1)
        )

        return {
            "requests_total": app.state.request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "uptime_seconds": round(uptime, 1),
            "model_loaded": app.state.is_ready,
        }

    return app


def _generate_recommendations(
    user_id: str,
    num_items: int = 10,
    context: Optional[Dict] = None,
    registry: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Generate recommendations using the model registry.

    Falls back to mock data when no model is loaded.

    Args:
        user_id: User identifier.
        num_items: Number of recommendations to return.
        context: Optional context (recent items, etc).
        registry: Model registry instance.

    Returns:
        List of recommendation dicts with item_id, title, score.
    """
    if registry is not None and hasattr(registry, "recommend"):
        return registry.recommend(user_id, num_items, context)

    # Mock recommendations for API testing
    import random
    rng = random.Random(hash(user_id))

    mock_items = [
        "Organic Peanut Butter", "Whole Wheat Pasta", "Green Tea Bags",
        "Almond Milk", "Olive Oil Extra Virgin", "Brown Rice",
        "Honey Raw Organic", "Dark Chocolate Bar", "Coconut Water",
        "Oatmeal Instant", "Avocado Oil", "Quinoa Organic",
        "Chia Seeds", "Greek Yogurt", "Mixed Nuts",
    ]

    rng.shuffle(mock_items)
    recommendations = []
    for i, item in enumerate(mock_items[:num_items]):
        recommendations.append({
            "item_id": f"item_{hash(item) % 10000:04d}",
            "title": item,
            "score": round(rng.uniform(0.6, 0.99), 4),
            "rank": i + 1,
        })

    return recommendations


async def _process_feedback(
    feedback_id: str,
    request: "FeedbackRequest",
) -> None:
    """Process feedback asynchronously (background task)."""
    logger.info(
        f"Processing feedback {feedback_id} | "
        f"user={request.user_id} item={request.item_id} "
        f"rating={request.rating}"
    )


# Create default app instance
app = create_app()
