"""
Knowledge graph endpoints
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.postgres import get_db
from backend.database.models import User, Video, GraphSnapshot, DetectedObject, Interaction
from backend.database.neo4j_client import get_neo4j_client
from backend.api_gateway.dependencies import get_current_active_user
from backend.core.storage import get_storage_client
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graphs", tags=["graphs"])


# Pydantic schemas
class GraphResponse(BaseModel):
    video_id: int
    version: int
    node_count: int
    edge_count: int
    graph_metrics: Optional[Dict[str, Any]]
    llm_narrative: Optional[str]
    llm_model: Optional[str]

    class Config:
        from_attributes = True


class ObjectNode(BaseModel):
    id: int
    object_id: int
    frame_number: int
    bbox: List[float]
    confidence: float
    label: Optional[str]


class InteractionEdge(BaseModel):
    id: int
    object_id_1: int
    object_id_2: int
    interaction_type: str
    start_frame: int
    end_frame: int
    confidence: float


class FullGraphResponse(BaseModel):
    video_id: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    narrative: Optional[str]


@router.get("/{video_id}", response_model=FullGraphResponse)
async def get_video_graph(
    video_id: int,
    version: Optional[int] = Query(None, description="Graph version (latest if not specified)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get complete knowledge graph for a video

    Args:
        video_id: Video ID
        version: Optional graph version
        db: Database session
        current_user: Authenticated user

    Returns:
        Full knowledge graph with nodes and edges
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get graph from Neo4j
    try:
        neo4j = get_neo4j_client()
        graph_data = neo4j.get_video_graph(video_id)

        # Get latest snapshot for metrics and narrative
        snapshot_query = db.query(GraphSnapshot).filter(GraphSnapshot.video_id == video_id)
        if version:
            snapshot_query = snapshot_query.filter(GraphSnapshot.version == version)
        snapshot = snapshot_query.order_by(GraphSnapshot.version.desc()).first()

        return FullGraphResponse(
            video_id=video_id,
            nodes=graph_data.get("objects", []),
            edges=graph_data.get("relationships", []),
            metrics=snapshot.graph_metrics if snapshot else {},
            narrative=snapshot.llm_narrative if snapshot else None,
        )

    except Exception as e:
        logger.error(f"Failed to get graph: {e}", video_id=video_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve graph",
        )


@router.get("/{video_id}/objects", response_model=List[ObjectNode])
async def get_video_objects(
    video_id: int,
    frame_number: Optional[int] = Query(None),
    label: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get detected objects for a video

    Args:
        video_id: Video ID
        frame_number: Optional frame filter
        label: Optional label filter
        skip: Pagination skip
        limit: Pagination limit
        db: Database session
        current_user: Authenticated user

    Returns:
        List of detected objects
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Query objects
    query = db.query(DetectedObject).filter(DetectedObject.video_id == video_id)

    if frame_number is not None:
        query = query.join(DetectedObject.frame).filter_by(frame_number=frame_number)

    if label:
        query = query.filter(DetectedObject.clip_label.ilike(f"%{label}%"))

    objects = query.offset(skip).limit(limit).all()

    return [
        ObjectNode(
            id=obj.id,
            object_id=obj.object_id,
            frame_number=obj.frame.frame_number,
            bbox=[obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2],
            confidence=obj.confidence,
            label=obj.clip_label,
        )
        for obj in objects
    ]


@router.get("/{video_id}/interactions", response_model=List[InteractionEdge])
async def get_video_interactions(
    video_id: int,
    interaction_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get detected interactions for a video

    Args:
        video_id: Video ID
        interaction_type: Optional interaction type filter
        skip: Pagination skip
        limit: Pagination limit
        db: Database session
        current_user: Authenticated user

    Returns:
        List of interactions
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Query interactions
    query = db.query(Interaction).filter(Interaction.video_id == video_id)

    if interaction_type:
        query = query.filter(Interaction.interaction_type == interaction_type)

    interactions = query.offset(skip).limit(limit).all()

    return [
        InteractionEdge(
            id=inter.id,
            object_id_1=inter.object_id_1,
            object_id_2=inter.object_id_2,
            interaction_type=inter.interaction_type,
            start_frame=inter.start_frame,
            end_frame=inter.end_frame,
            confidence=inter.confidence,
        )
        for inter in interactions
    ]


@router.get("/{video_id}/objects/{object_id}/interactions")
async def get_object_interactions(
    video_id: int,
    object_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all interactions for a specific object

    Args:
        video_id: Video ID
        object_id: Object tracking ID
        db: Database session
        current_user: Authenticated user

    Returns:
        List of interactions
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get from Neo4j for richer graph context
    try:
        neo4j = get_neo4j_client()
        interactions = neo4j.get_object_interactions(video_id, object_id)
        return {"object_id": object_id, "interactions": interactions}
    except Exception as e:
        logger.error(f"Failed to get object interactions: {e}", object_id=object_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve interactions",
        )


@router.get("/{video_id}/metrics")
async def get_graph_metrics(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get graph metrics for a video

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Graph metrics
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    try:
        neo4j = get_neo4j_client()
        metrics = neo4j.compute_graph_metrics(video_id)
        return {"video_id": video_id, "metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}", video_id=video_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute metrics",
        )


@router.get("/{video_id}/narrative")
async def get_graph_narrative(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get LLM-generated narrative for video graph

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Narrative text
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get latest snapshot
    snapshot = (
        db.query(GraphSnapshot)
        .filter(GraphSnapshot.video_id == video_id)
        .order_by(GraphSnapshot.version.desc())
        .first()
    )

    if not snapshot or not snapshot.llm_narrative:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Narrative not yet generated",
        )

    return {
        "video_id": video_id,
        "narrative": snapshot.llm_narrative,
        "model": snapshot.llm_model,
        "generated_at": snapshot.created_at,
    }


@router.get("/{video_id}/download")
async def download_graph(
    video_id: int,
    format: str = Query("json", regex="^(json|gexf|graphml)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Download graph in specified format

    Args:
        video_id: Video ID
        format: Export format (json, gexf, graphml)
        db: Database session
        current_user: Authenticated user

    Returns:
        Presigned download URL
    """
    # Verify video ownership
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get presigned URL for graph JSON
    storage = get_storage_client()
    object_key = f"graphs/{video_id}/graph.{format}"

    if not storage.object_exists(object_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph file not found in {format} format",
        )

    url = storage.get_presigned_url(object_key, expiration=3600)

    return {"url": url, "format": format, "expires_in": 3600}
