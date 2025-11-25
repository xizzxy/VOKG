"""
Natural language query endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query as QueryParam
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database.postgres import get_db
from backend.database.models import User, Video
from backend.database.neo4j_client import get_neo4j_client
from backend.api_gateway.dependencies import get_current_active_user
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/queries", tags=["queries"])


# Pydantic schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    video_id: int


class QueryResult(BaseModel):
    query: str
    results: List[dict]
    result_type: str  # objects, interactions, events, narrative
    confidence: Optional[float] = None


@router.post("", response_model=QueryResult)
async def execute_query(
    query_request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Execute natural language query on video graph

    Args:
        query_request: Query request
        db: Database session
        current_user: Authenticated user

    Returns:
        Query results
    """
    # Verify video ownership
    video = (
        db.query(Video)
        .filter(Video.id == query_request.video_id, Video.user_id == current_user.id)
        .first()
    )
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Parse query and execute
    # This is a simplified implementation - would need NL parsing logic
    query_lower = query_request.query.lower()

    try:
        neo4j = get_neo4j_client()

        # Simple pattern matching for common queries
        if "objects" in query_lower or "what" in query_lower:
            # Find objects by label
            label_keywords = extract_keywords(query_request.query)
            results = []
            for keyword in label_keywords:
                objects = neo4j.find_objects_by_label(query_request.video_id, keyword)
                results.extend(objects)

            return QueryResult(
                query=query_request.query,
                results=results,
                result_type="objects",
            )

        elif "interact" in query_lower or "relationship" in query_lower:
            # Get interactions
            graph_data = neo4j.get_video_graph(query_request.video_id)
            return QueryResult(
                query=query_request.query,
                results=graph_data.get("relationships", []),
                result_type="interactions",
            )

        elif "when" in query_lower or "timeline" in query_lower or "sequence" in query_lower:
            # Get temporal sequence
            sequence = neo4j.get_temporal_sequence(query_request.video_id)
            return QueryResult(
                query=query_request.query,
                results=sequence,
                result_type="events",
            )

        else:
            # Default: return full graph
            graph_data = neo4j.get_video_graph(query_request.video_id)
            return QueryResult(
                query=query_request.query,
                results=graph_data.get("objects", []),
                result_type="objects",
            )

    except Exception as e:
        logger.error(f"Query execution failed: {e}", query=query_request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute query",
        )


@router.get("/{video_id}/suggestions")
async def get_query_suggestions(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get suggested queries for a video

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        List of suggested queries
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Generate suggestions based on video content
    # This is a simplified implementation
    suggestions = [
        "What objects appear in the video?",
        "Show all interactions between objects",
        "What is the timeline of events?",
        "Which objects move the most?",
        "What causes the main events?",
        "Summarize the video narrative",
    ]

    return {"video_id": video_id, "suggestions": suggestions}


def extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from natural language query

    Args:
        query: Natural language query

    Returns:
        List of keywords
    """
    # Simple keyword extraction - would use NLP in production
    stop_words = {"the", "a", "an", "in", "on", "at", "is", "are", "what", "where", "when", "how"}
    words = query.lower().split()
    keywords = [w.strip("?.,!") for w in words if w not in stop_words and len(w) > 2]
    return keywords
