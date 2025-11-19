"""
Unit tests for the FastAPI recommendation API.

Tests cover:
- Health and readiness endpoints
- Recommendation request/response
- Batch recommendation endpoint
- Feedback submission
- Metrics endpoint
- Schema validation
"""

import pytest

from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.schemas import (
    RecommendationRequest,
    BatchRecommendationRequest,
    FeedbackRequest,
    HealthResponse,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    app.state.is_ready = True
    return TestClient(app)


@pytest.fixture
def client_not_ready():
    """Create a test client with model NOT ready."""
    app = create_app()
    app.state.is_ready = False
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and readiness probes."""

    def test_health_check(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert data["version"] == "1.0.0"

    def test_readiness_when_ready(self, client):
        """Test readiness when model is loaded."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_readiness_when_not_ready(self, client_not_ready):
        """Test readiness returns 503 when model not loaded."""
        response = client_not_ready.get("/ready")
        assert response.status_code == 503


class TestRecommendationEndpoints:
    """Tests for recommendation endpoints."""

    def test_single_recommendation(self, client):
        """Test single-user recommendation."""
        response = client.post("/recommend", json={
            "user_id": "user_123",
            "num_items": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_123"
        assert len(data["recommendations"]) == 5
        assert "latency_ms" in data

    def test_recommendation_defaults(self, client):
        """Test recommendation with default num_items."""
        response = client.post("/recommend", json={
            "user_id": "user_456",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 10  # Default

    def test_recommendation_with_context(self, client):
        """Test recommendation with context."""
        response = client.post("/recommend", json={
            "user_id": "user_789",
            "num_items": 3,
            "context": {"recent_items": ["item_001"]},
        })
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 3

    def test_recommendation_has_scores(self, client):
        """Test that recommendations include scores."""
        response = client.post("/recommend", json={
            "user_id": "user_123",
            "num_items": 3,
        })
        recs = response.json()["recommendations"]
        for rec in recs:
            assert "score" in rec
            assert "title" in rec
            assert "item_id" in rec

    def test_batch_recommendations(self, client):
        """Test batch recommendation endpoint."""
        response = client.post("/recommend/batch", json={
            "user_ids": ["user_1", "user_2", "user_3"],
            "num_items": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["responses"]) == 3
        assert "total_latency_ms" in data


class TestFeedbackEndpoint:
    """Tests for feedback submission."""

    def test_submit_feedback(self, client):
        """Test feedback submission returns accepted."""
        response = client.post("/feedback", json={
            "user_id": "user_123",
            "item_id": "item_456",
            "rating": 4.5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "feedback_id" in data

    def test_feedback_with_interaction_type(self, client):
        """Test feedback with custom interaction type."""
        response = client.post("/feedback", json={
            "user_id": "user_123",
            "item_id": "item_789",
            "rating": 3.0,
            "interaction_type": "purchase",
        })
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_stats(self, client):
        """Test that metrics endpoint returns stats."""
        # Make a few requests first
        client.get("/health")
        client.post("/recommend", json={"user_id": "u1"})

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "avg_latency_ms" in data
        assert "uptime_seconds" in data


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_recommendation_request_valid(self):
        """Test valid recommendation request."""
        req = RecommendationRequest(user_id="u1", num_items=5)
        assert req.user_id == "u1"
        assert req.num_items == 5

    def test_recommendation_request_defaults(self):
        """Test default values."""
        req = RecommendationRequest(user_id="u1")
        assert req.num_items == 10
        assert req.context is None

    def test_feedback_request_validation(self):
        """Test feedback schema validation."""
        req = FeedbackRequest(
            user_id="u1", item_id="i1", rating=4.0
        )
        assert req.rating == 4.0
        assert req.interaction_type == "click"

    def test_batch_request(self):
        """Test batch request schema."""
        req = BatchRecommendationRequest(
            user_ids=["u1", "u2"], num_items=5
        )
        assert len(req.user_ids) == 2

    def test_request_id_header(self, client):
        """Test that API returns X-Request-ID header."""
        response = client.get("/health")
        assert "x-request-id" in response.headers
        assert "x-latency-ms" in response.headers
