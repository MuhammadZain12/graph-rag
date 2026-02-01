"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="The user's question")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str = Field(..., description="The generated answer")
    is_department_related: bool = Field(..., description="Whether the question was classified as department-related")
    guardrail_reason: Optional[str] = Field(None, description="Reason from guardrail classification")
    sources: list[str] = Field(default_factory=list, description="List of source chunk IDs used for the answer")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str = "ok"
    version: str = "1.0.0"
