"""
Enhanced Knowledge Graph with semantic information
Stores objects, interactions, and semantic labels in memory
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json

from backend.core.logging import get_logger
from backend.services.search.query_executor import InteractionResult

logger = get_logger(__name__)


class EnhancedKnowledgeGraph:
    """
    In-memory knowledge graph with semantic labels
    Optimized for the test pipeline (no database required)
    """

    def __init__(self, video_id: int = 0, fps: float = 2.0):
        """
        Initialize knowledge graph

        Args:
            video_id: Video identifier
            fps: Frames per second for timestamp conversion
        """
        self.video_id = video_id
        self.fps = fps

        # Storage
        self.objects: Dict[int, Dict] = {}  # object_id -> object data
        self.interactions: List[Dict] = []  # List of interactions
        self.semantic_labels: Dict[int, Dict] = {}  # object_id -> semantic label

        # Indexes for fast lookup
        self.label_to_objects: Dict[str, List[int]] = defaultdict(list)
        self.category_to_objects: Dict[str, List[int]] = defaultdict(list)
        self.object_interactions: Dict[int, List[int]] = defaultdict(list)  # object_id -> interaction indices

        logger.info(f"EnhancedKnowledgeGraph initialized for video {video_id}")

    def add_object(
        self,
        object_id: int,
        frame_number: int,
        timestamp: float,
        bbox: List[float],
        semantic_label: Dict[str, Any]
    ):
        """
        Add object to graph

        Args:
            object_id: Unique object ID
            frame_number: Frame where object appears
            timestamp: Timestamp in seconds
            bbox: Bounding box [x1, y1, x2, y2]
            semantic_label: Semantic label dictionary from SemanticLabeler
        """
        if object_id not in self.objects:
            self.objects[object_id] = {
                'object_id': object_id,
                'label': semantic_label.get('primary_label', 'unknown'),
                'confidence': semantic_label.get('confidence', 0.0),
                'category': semantic_label.get('category', 'unknown'),
                'super_category': semantic_label.get('super_category', 'unknown'),
                'first_seen_frame': frame_number,
                'first_seen_time': timestamp,
                'last_seen_frame': frame_number,
                'last_seen_time': timestamp,
                'appearances': [],
                'total_appearances': 0
            }

            # Store full semantic label
            self.semantic_labels[object_id] = semantic_label

            # Update indexes
            label = semantic_label.get('primary_label', 'unknown')
            category = semantic_label.get('category', 'unknown')
            self.label_to_objects[label].append(object_id)
            self.category_to_objects[category].append(object_id)

        # Update appearance info
        obj = self.objects[object_id]
        obj['appearances'].append({
            'frame': frame_number,
            'timestamp': timestamp,
            'bbox': bbox
        })
        obj['last_seen_frame'] = frame_number
        obj['last_seen_time'] = timestamp
        obj['total_appearances'] += 1

    def add_interaction(
        self,
        object_id_1: int,
        object_id_2: int,
        interaction_type: str,
        frame: int,
        timestamp: float,
        confidence: float,
        metadata: Dict = None
    ):
        """
        Add interaction between objects

        Args:
            object_id_1: First object ID
            object_id_2: Second object ID
            interaction_type: Type of interaction
            frame: Frame number
            timestamp: Timestamp in seconds
            confidence: Confidence score
            metadata: Additional metadata
        """
        interaction = {
            'object_id_1': object_id_1,
            'object_id_2': object_id_2,
            'type': interaction_type,
            'frame': frame,
            'timestamp': timestamp,
            'start_frame': frame,
            'end_frame': frame,
            'start_time': timestamp,
            'end_time': timestamp,
            'confidence': confidence,
            'metadata': metadata or {}
        }

        interaction_idx = len(self.interactions)
        self.interactions.append(interaction)

        # Update indexes
        self.object_interactions[object_id_1].append(interaction_idx)
        self.object_interactions[object_id_2].append(interaction_idx)

    def get_object(self, object_id: int) -> Optional[Dict]:
        """Get object by ID"""
        return self.objects.get(object_id)

    def get_objects_by_label(self, label: str) -> List[Dict]:
        """Get all objects with specific label"""
        object_ids = self.label_to_objects.get(label.lower(), [])
        return [self.objects[oid] for oid in object_ids if oid in self.objects]

    def get_objects_by_category(self, category: str) -> List[Dict]:
        """Get all objects in a category"""
        object_ids = self.category_to_objects.get(category.lower(), [])
        return [self.objects[oid] for oid in object_ids if oid in self.objects]

    def get_all_objects(self) -> List[Dict]:
        """Get all objects"""
        return list(self.objects.values())

    def get_all_interactions(self) -> List[Dict]:
        """Get all interactions"""
        return self.interactions

    def get_interactions_for_object(
        self,
        object_id: int,
        interaction_types: List[str] = None
    ) -> List[InteractionResult]:
        """
        Get all interactions involving an object

        Args:
            object_id: Object ID
            interaction_types: Optional filter by interaction types

        Returns:
            List of InteractionResults
        """
        interaction_indices = self.object_interactions.get(object_id, [])
        results = []

        for idx in interaction_indices:
            interaction = self.interactions[idx]

            # Filter by type if specified
            if interaction_types and interaction['type'] not in interaction_types:
                continue

            # Get labels for both objects
            obj1_id = interaction['object_id_1']
            obj2_id = interaction['object_id_2']

            obj1 = self.objects.get(obj1_id, {})
            obj2 = self.objects.get(obj2_id, {})

            result = InteractionResult(
                object_a_id=obj1_id,
                object_a_label=obj1.get('label', f'object_{obj1_id}'),
                object_b_id=obj2_id,
                object_b_label=obj2.get('label', f'object_{obj2_id}'),
                interaction_type=interaction['type'],
                start_frame=interaction['start_frame'],
                end_frame=interaction['end_frame'],
                start_time=interaction['start_time'],
                end_time=interaction['end_time'],
                confidence=interaction['confidence']
            )
            results.append(result)

        return results

    def get_interactions_between(
        self,
        object_id_1: int,
        object_id_2: int,
        interaction_types: List[str] = None
    ) -> List[InteractionResult]:
        """
        Get interactions between two specific objects

        Args:
            object_id_1: First object ID
            object_id_2: Second object ID
            interaction_types: Optional filter by types

        Returns:
            List of InteractionResults
        """
        # Get interactions for first object
        all_interactions = self.get_interactions_for_object(object_id_1, interaction_types)

        # Filter to only those involving second object
        results = [
            i for i in all_interactions
            if (i.object_b_id == object_id_2 or i.object_a_id == object_id_2)
        ]

        return results

    def get_neighbors(self, object_id: int, max_distance: int = 1) -> List[int]:
        """
        Get neighboring objects (objects that interact with this one)

        Args:
            object_id: Object ID
            max_distance: Maximum graph distance (1 = direct neighbors)

        Returns:
            List of neighboring object IDs
        """
        if max_distance == 0:
            return [object_id]

        neighbors = set()
        interaction_indices = self.object_interactions.get(object_id, [])

        for idx in interaction_indices:
            interaction = self.interactions[idx]
            obj1_id = interaction['object_id_1']
            obj2_id = interaction['object_id_2']

            # Add the other object
            if obj1_id == object_id:
                neighbors.add(obj2_id)
            else:
                neighbors.add(obj1_id)

        # For distance > 1, recursively get neighbors
        if max_distance > 1:
            extended = set(neighbors)
            for neighbor_id in list(neighbors):
                extended.update(self.get_neighbors(neighbor_id, max_distance - 1))
            neighbors = extended

        return list(neighbors)

    def get_subgraph(
        self,
        object_ids: List[int],
        include_neighbors: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract subgraph containing specific objects

        Args:
            object_ids: List of object IDs to include
            include_neighbors: Whether to include direct neighbors

        Returns:
            Tuple of (objects, interactions) in subgraph
        """
        # Expand to include neighbors if requested
        if include_neighbors:
            expanded = set(object_ids)
            for oid in object_ids:
                expanded.update(self.get_neighbors(oid, max_distance=1))
            object_ids = list(expanded)

        # Get objects
        objects = [self.objects[oid] for oid in object_ids if oid in self.objects]

        # Get interactions between these objects
        object_id_set = set(object_ids)
        interactions = [
            i for i in self.interactions
            if i['object_id_1'] in object_id_set and i['object_id_2'] in object_id_set
        ]

        return objects, interactions

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        # Label distribution
        label_counts = defaultdict(int)
        for obj in self.objects.values():
            label_counts[obj['label']] += 1

        # Interaction type distribution
        interaction_type_counts = defaultdict(int)
        for interaction in self.interactions:
            interaction_type_counts[interaction['type']] += 1

        # Temporal range
        timestamps = [obj['first_seen_time'] for obj in self.objects.values()]
        if timestamps:
            duration = max(timestamps) - min(timestamps)
        else:
            duration = 0.0

        return {
            'total_objects': len(self.objects),
            'total_interactions': len(self.interactions),
            'unique_labels': len(label_counts),
            'label_distribution': dict(label_counts),
            'interaction_types': dict(interaction_type_counts),
            'duration': duration,
            'fps': self.fps
        }

    def export_to_json(self) -> Dict:
        """Export graph to JSON format"""
        # Convert objects to serializable format
        objects_json = []
        for obj in self.objects.values():
            obj_copy = obj.copy()
            # Get semantic label
            sem_label = self.semantic_labels.get(obj['object_id'], {})
            obj_copy['semantic_label'] = {
                k: v for k, v in sem_label.items()
                if k != 'embedding'  # Exclude embedding (too large)
            }
            objects_json.append(obj_copy)

        return {
            'video_id': self.video_id,
            'fps': self.fps,
            'objects': objects_json,
            'interactions': self.interactions,
            'statistics': self.get_statistics()
        }

    def save_to_file(self, filepath: str):
        """Save graph to JSON file"""
        data = self.export_to_json()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Knowledge graph saved to {filepath}")
