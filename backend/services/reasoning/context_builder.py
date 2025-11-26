"""
Graph context builder
Compresses knowledge graph into LLM-friendly context
"""

import numpy as np
from typing import Dict, List, Any, Optional

from backend.core.logging import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """
    Builds compressed context from knowledge graph for LLM reasoning
    """

    def __init__(self, max_tokens: int = 4000):
        """
        Initialize context builder

        Args:
            max_tokens: Maximum tokens for context (approximate)
        """
        self.max_tokens = max_tokens

    def build_context(
        self,
        objects: List[Dict],
        interactions: List[Dict],
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Build compressed graph context

        Args:
            objects: List of object dictionaries with labels and embeddings
            interactions: List of interaction dictionaries
            query: Optional query to focus context on
            query_embedding: Optional query embedding for relevance ranking

        Returns:
            Formatted context string
        """
        # If query embedding provided, rank objects by relevance
        if query_embedding is not None and len(objects) > 0:
            objects = self._rank_by_relevance(objects, query_embedding)

        # Limit objects
        max_objects = min(50, len(objects))
        objects = objects[:max_objects]

        # Build context sections
        context_parts = []

        # Objects section
        context_parts.append("## Detected Objects\n")
        for i, obj in enumerate(objects[:30], 1):  # Top 30 objects
            context_parts.append(
                f"{i}. Object {obj['object_id']}: **{obj['label']}** "
                f"(confidence: {obj['confidence']:.2f}, "
                f"category: {obj.get('category', 'unknown')}, "
                f"frames: {obj.get('frame_count', 1)})"
            )

        if len(objects) > 30:
            context_parts.append(f"... and {len(objects) - 30} more objects")

        # Interactions section
        context_parts.append("\n## Interactions & Events\n")

        # Group interactions by type
        interactions_by_type = {}
        for interaction in interactions:
            itype = interaction.get('type', 'unknown')
            if itype not in interactions_by_type:
                interactions_by_type[itype] = []
            interactions_by_type[itype].append(interaction)

        # Show top interactions of each type
        interaction_count = 0
        for itype, ilist in sorted(interactions_by_type.items()):
            context_parts.append(f"\n### {itype.capitalize()} Interactions:")

            for interaction in sorted(ilist, key=lambda x: x.get('timestamp', 0))[:10]:
                obj1_id = interaction.get('object_id_1', interaction.get('object_a_id'))
                obj2_id = interaction.get('object_id_2', interaction.get('object_b_id'))

                # Find object labels
                obj1_label = self._get_object_label(obj1_id, objects)
                obj2_label = self._get_object_label(obj2_id, objects)

                timestamp = interaction.get('timestamp', interaction.get('start_time', 0))

                context_parts.append(
                    f"- t={timestamp:.1f}s: {obj1_label} {itype} {obj2_label} "
                    f"(confidence: {interaction.get('confidence', 0):.2f})"
                )

                interaction_count += 1
                if interaction_count >= 40:  # Max 40 interactions
                    break

            if interaction_count >= 40:
                context_parts.append(f"... and {len(interactions) - 40} more interactions")
                break

        # Temporal summary
        context_parts.append("\n## Temporal Summary\n")
        if interactions:
            timestamps = [i.get('timestamp', i.get('start_time', 0)) for i in interactions]
            context_parts.append(
                f"- Activity spans from {min(timestamps):.1f}s to {max(timestamps):.1f}s"
            )
            context_parts.append(f"- Total of {len(interactions)} interactions detected")

        # Join all parts
        context = "\n".join(context_parts)

        # Truncate if too long (rough estimate: 1 token ≈ 4 characters)
        max_chars = self.max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... [context truncated due to length]"
            logger.warning(f"Context truncated from {len(context)} to {max_chars} chars")

        return context

    def _rank_by_relevance(
        self,
        objects: List[Dict],
        query_embedding: np.ndarray
    ) -> List[Dict]:
        """Rank objects by relevance to query embedding"""
        scores = []

        for obj in objects:
            if 'embedding' in obj and obj['embedding'] is not None:
                emb = np.array(obj['embedding'])
                # Cosine similarity
                similarity = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
                )
                scores.append(similarity)
            else:
                scores.append(0.0)

        # Sort by score
        ranked_indices = np.argsort(scores)[::-1]
        return [objects[i] for i in ranked_indices]

    def _get_object_label(self, object_id: int, objects: List[Dict]) -> str:
        """Get label for an object ID"""
        for obj in objects:
            if obj.get('object_id') == object_id:
                return obj.get('label', f'object_{object_id}')
        return f'object_{object_id}'

    def build_summary_statistics(
        self,
        objects: List[Dict],
        interactions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Build summary statistics for the graph

        Args:
            objects: List of objects
            interactions: List of interactions

        Returns:
            Dictionary of summary statistics
        """
        # Count labels
        label_counts = {}
        for obj in objects:
            label = obj.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

        # Top objects
        top_objects = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Interaction types
        interaction_types = {}
        for interaction in interactions:
            itype = interaction.get('type', 'unknown')
            interaction_types[itype] = interaction_types.get(itype, 0) + 1

        # Duration
        timestamps = []
        for interaction in interactions:
            timestamps.append(interaction.get('timestamp', interaction.get('start_time', 0)))
        duration = max(timestamps) if timestamps else 0.0

        return {
            'total_objects': len(objects),
            'total_interactions': len(interactions),
            'unique_labels': list(label_counts.keys()),
            'top_objects': top_objects,
            'top_objects_text': '\n'.join([f"- {label}: {count} instances" for label, count in top_objects]),
            'interaction_types': interaction_types,
            'key_interactions_text': '\n'.join([f"- {itype}: {count} occurrences" for itype, count in interaction_types.items()]),
            'duration': duration
        }
