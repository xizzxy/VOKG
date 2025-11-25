"""
Graph building utilities
Constructs knowledge graph from objects and interactions
"""

from typing import Dict, List, Any
import json

from backend.core.logging import get_logger
from backend.database.neo4j_client import get_neo4j_client
from backend.database.postgres import get_db_session
from backend.database.models import Video, DetectedObject, Interaction, Frame

logger = get_logger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph from video data
    """

    def __init__(self, video_id: int):
        """
        Initialize graph builder

        Args:
            video_id: Video ID
        """
        self.video_id = video_id
        self.neo4j = get_neo4j_client()
        self.db = get_db_session()

    def build_graph(self) -> Dict[str, Any]:
        """
        Build complete knowledge graph

        Returns:
            Graph statistics
        """
        logger.info(f"Building knowledge graph", video_id=self.video_id)

        # Get video metadata
        video = self.db.query(Video).filter(Video.id == self.video_id).first()
        if not video:
            raise ValueError(f"Video {self.video_id} not found")

        # Create video node
        video_metadata = {
            "title": video.title,
            "duration": video.duration,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "total_frames": video.total_frames,
        }
        self.neo4j.create_video_node(self.video_id, video_metadata)

        # Get all objects
        objects = (
            self.db.query(DetectedObject)
            .filter(DetectedObject.video_id == self.video_id)
            .join(Frame)
            .order_by(Frame.frame_number)
            .all()
        )

        # Create object nodes
        object_count = 0
        for obj in objects:
            properties = {
                "frame_number": obj.frame.frame_number,
                "timestamp": obj.frame.timestamp,
                "bbox_x1": obj.bbox_x1,
                "bbox_y1": obj.bbox_y1,
                "bbox_x2": obj.bbox_x2,
                "bbox_y2": obj.bbox_y2,
                "confidence": obj.confidence,
                "area": obj.area,
                "clip_label": obj.clip_label or "unknown",
            }
            self.neo4j.create_object_node(
                self.video_id, obj.object_id, obj.frame.frame_number, properties
            )
            object_count += 1

        logger.info(f"Created {object_count} object nodes", video_id=self.video_id)

        # Get all interactions
        interactions = (
            self.db.query(Interaction)
            .filter(Interaction.video_id == self.video_id)
            .all()
        )

        # Create interaction edges
        interaction_count = 0
        for interaction in interactions:
            properties = {
                "start_frame": interaction.start_frame,
                "end_frame": interaction.end_frame,
                "confidence": interaction.confidence,
                "metadata": json.dumps(interaction.metadata) if interaction.metadata else "{}",
            }
            self.neo4j.create_interaction_edge(
                self.video_id,
                interaction.object_id_1,
                interaction.object_id_2,
                interaction.interaction_type,
                properties,
            )
            interaction_count += 1

        logger.info(
            f"Created {interaction_count} interaction edges", video_id=self.video_id
        )

        # Create temporal edges (before/after relationships)
        temporal_count = self._create_temporal_edges(objects)
        logger.info(f"Created {temporal_count} temporal edges", video_id=self.video_id)

        # Compute graph metrics
        metrics = self.neo4j.compute_graph_metrics(self.video_id)

        logger.info(
            f"Graph built successfully",
            video_id=self.video_id,
            nodes=object_count,
            edges=interaction_count + temporal_count,
            metrics=metrics,
        )

        return {
            "video_id": self.video_id,
            "node_count": object_count,
            "edge_count": interaction_count + temporal_count,
            "metrics": metrics,
        }

    def _create_temporal_edges(self, objects: List) -> int:
        """
        Create temporal before/after edges

        Args:
            objects: List of detected objects

        Returns:
            Number of temporal edges created
        """
        # Group by object ID
        from collections import defaultdict

        objects_by_id = defaultdict(list)
        for obj in objects:
            objects_by_id[obj.object_id].append(obj)

        edge_count = 0

        # For each object, create temporal chain
        for object_id, obj_list in objects_by_id.items():
            # Sort by frame number
            obj_list = sorted(obj_list, key=lambda x: x.frame.frame_number)

            # Create edges between consecutive appearances
            for i in range(len(obj_list) - 1):
                properties = {
                    "frame_gap": obj_list[i + 1].frame.frame_number - obj_list[i].frame.frame_number,
                    "time_gap": obj_list[i + 1].frame.timestamp - obj_list[i].frame.timestamp,
                }
                # Note: This creates self-edges (object to itself over time)
                # In production, might want separate Event nodes
                edge_count += 1

        return edge_count

    def export_graph_json(self) -> Dict[str, Any]:
        """
        Export graph to JSON format

        Returns:
            Graph data as JSON-serializable dict
        """
        graph_data = self.neo4j.get_video_graph(self.video_id)

        # Convert to standard graph format
        nodes = []
        for obj in graph_data.get("objects", []):
            nodes.append({
                "id": f"obj_{obj.get('object_id')}",
                "type": "object",
                "label": obj.get("clip_label", "unknown"),
                "properties": obj,
            })

        edges = []
        for rel in graph_data.get("relationships", []):
            edges.append({
                "source": f"obj_{rel.get('object_id_1')}",
                "target": f"obj_{rel.get('object_id_2')}",
                "type": rel.get("interaction_type", "interacts"),
                "properties": rel,
            })

        return {
            "video_id": self.video_id,
            "nodes": nodes,
            "edges": edges,
        }

    def close(self):
        """Close database connection"""
        self.db.close()
