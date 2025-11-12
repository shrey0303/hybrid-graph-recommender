"""
Pydantic schemas for the recommendation API.

Defines request/response models with validation for all API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Schemas ──

class RecommendationRequest(BaseModel):
    """Request schema for single-user recommendations."""
    user_id: str = Field(..., description="Unique user identifier")
    num_items: int = Field(
        default=10, ge=1, le=100,
        description="Number of recommendations to return",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (recent items, session info)",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "user_id": "user_1234",
            "num_items": 5,
            "context": {"recent_items": ["item_001", "item_002"]},
        }]
    }}


class BatchRecommendationRequest(BaseModel):
    """Request schema for batch recommendations."""
    user_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of user IDs to get recommendations for",
    )
    num_items: int = Field(default=10, ge=1, le=100)


class FeedbackRequest(BaseModel):
    """Request schema for user feedback."""
    user_id: str = Field(..., description="User who provided feedback")
    item_id: str = Field(..., description="Item being rated")
    rating: float = Field(
        ..., ge=0, le=5,
        description="User rating (0-5 scale)",
    )
    interaction_type: str = Field(
        default="click",
        description="Type of interaction: click, purchase, view, skip",
    )
    timestamp: Optional[str] = Field(
        default=None, description="ISO timestamp of interaction"
    )


# ── Response Schemas ──

class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    user_id: str
    recommendations: List[Dict[str, Any]] = Field(
        description="Ordered list of recommended items with scores"
    )
    model_version: str = Field(description="Model version that generated recs")
    latency_ms: float = Field(description="Server-side latency in milliseconds")


class BatchRecommendationResponse(BaseModel):
    """Response schema for batch recommendations."""
    responses: List[RecommendationResponse]
    total_latency_ms: float


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""
    feedback_id: str
    status: str = Field(description="Processing status: accepted, rejected")
    message: str


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(description="Service health status")
    uptime_seconds: float
    version: str
