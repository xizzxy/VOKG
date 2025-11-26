"""
Query execution engine
Executes structured queries against the knowledge graph and embedding store
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .query_parser import StructuredQuery, QueryType
from .embedding_store import EmbeddingStore, SearchResult
from backend.services.semantic.clip_encoder import CLIPEncoder
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InteractionResult:
    """Result of an interaction query"""
    object_a_id: int
    object_a_label: str
    object_b_id: int
    object_b_label: str
    interaction_type: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float

    def to_dict(self) -> Dict:
        return {
            'object_a': {'id': self.object_a_id, 'label': self.object_a_label},
            'object_b': {'id': self.object_b_id, 'label': self.object_b_label},
            'interaction_type': self.interaction_type,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence
        }


@dataclass
class QueryResult:
    """Unified query result"""
    query_type: str
    objects: List[SearchResult] = None
    interactions: List[InteractionResult] = None
    timeline: List[Dict] = None
    reasoning_answer: str = None
    summary: str = None

    def to_dict(self) -> Dict:
        result = {
            'query_type': self.query_type,
            'summary': self.summary or ''
        }

        if self.objects:
            result['objects'] = [obj.to_dict() for obj in self.objects]
        if self.interactions:
            result['interactions'] = [inter.to_dict() for inter in self.interactions]
        if self.timeline:
            result['timeline'] = self.timeline
        if self.reasoning_answer:
            result['reasoning_answer'] = self.reasoning_answer

        return result


class QueryExecutor:
    """
    Executes structured queries against knowledge graph and embedding store
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        clip_encoder: CLIPEncoder,
        knowledge_graph: Any = None,  # Will be the enhanced KG
        fps: float = 2.0
    ):
        """
        Initialize query executor

        Args:
            embedding_store: Embedding store for similarity search
            clip_encoder: CLIP encoder for text queries
            knowledge_graph: Knowledge graph instance
            fps: Video frame rate for timestamp conversion
        """
        self.embedding_store = embedding_store
        self.clip_encoder = clip_encoder
        self.knowledge_graph = knowledge_graph
        self.fps = fps

    def execute(self, structured_query: StructuredQuery) -> QueryResult:
        """
        Execute a structured query

        Args:
            structured_query: Parsed query

        Returns:
            QueryResult with search results
        """
        logger.info(f"Executing {structured_query.query_type.value} query: {structured_query.original_query}")

        if structured_query.query_type == QueryType.SEARCH:
            return self._execute_search(structured_query)
        elif structured_query.query_type == QueryType.INTERACTION:
            return self._execute_interaction(structured_query)
        elif structured_query.query_type == QueryType.TIMELINE:
            return self._execute_timeline(structured_query)
        elif structured_query.query_type == QueryType.REASONING:
            return self._execute_reasoning(structured_query)
        else:
            raise ValueError(f"Unknown query type: {structured_query.query_type}")

    def _execute_search(self, query: StructuredQuery) -> QueryResult:
        """Execute search query to find objects"""
        all_results = []

        for target in query.target_objects:
            # Encode target as text
            text_embedding = self.clip_encoder.encode_text(target)[0]

            # Search embedding store
            results = self.embedding_store.search_by_text(
                text_embedding,
                k=50,
                min_similarity=0.2
            )

            # Apply temporal filter
            if query.temporal_constraint:
                results = self._filter_temporal(results, query.temporal_constraint)

            all_results.extend(results)

        # Remove duplicates and sort by similarity
        seen = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.similarity_score, reverse=True):
            if r.object_id not in seen:
                seen.add(r.object_id)
                unique_results.append(r)

        # Generate summary
        if unique_results:
            summary = self._generate_search_summary(query, unique_results)
        else:
            summary = f"No objects matching '{', '.join(query.target_objects)}' found in the video."

        return QueryResult(
            query_type="search",
            objects=unique_results,
            summary=summary
        )

    def _execute_interaction(self, query: StructuredQuery) -> QueryResult:
        """Execute interaction query"""
        # Find target objects first
        target_objects_map = {}

        for target in query.target_objects:
            text_embedding = self.clip_encoder.encode_text(target)[0]
            results = self.embedding_store.search_by_text(text_embedding, k=10, min_similarity=0.3)

            if results:
                target_objects_map[target] = results[0]  # Take best match

        if len(target_objects_map) < len(query.target_objects):
            missing = set(query.target_objects) - set(target_objects_map.keys())
            return QueryResult(
                query_type="interaction",
                interactions=[],
                summary=f"Could not find objects: {', '.join(missing)}"
            )

        # Query knowledge graph for interactions
        interactions = []
        if self.knowledge_graph:
            # Get all combinations of target objects
            objects = list(target_objects_map.values())

            if len(objects) == 1:
                # Find all interactions involving this object
                interactions = self.knowledge_graph.get_interactions_for_object(
                    objects[0].object_id,
                    interaction_types=query.interaction_types
                )
            else:
                # Find interactions between specific objects
                for i in range(len(objects)):
                    for j in range(i + 1, len(objects)):
                        obj_interactions = self.knowledge_graph.get_interactions_between(
                            objects[i].object_id,
                            objects[j].object_id,
                            interaction_types=query.interaction_types
                        )
                        interactions.extend(obj_interactions)

            # Apply temporal filter
            if query.temporal_constraint:
                interactions = self._filter_interactions_temporal(interactions, query.temporal_constraint)

        summary = self._generate_interaction_summary(query, target_objects_map, interactions)

        return QueryResult(
            query_type="interaction",
            objects=list(target_objects_map.values()),
            interactions=interactions,
            summary=summary
        )

    def _execute_timeline(self, query: StructuredQuery) -> QueryResult:
        """Execute timeline query to track object over time"""
        # Find target object
        if not query.target_objects:
            return QueryResult(
                query_type="timeline",
                timeline=[],
                summary="No target object specified for timeline query."
            )

        target = query.target_objects[0]
        text_embedding = self.clip_encoder.encode_text(target)[0]
        results = self.embedding_store.search_by_text(text_embedding, k=100, min_similarity=0.3)

        if not results:
            return QueryResult(
                query_type="timeline",
                timeline=[],
                summary=f"No instances of '{target}' found in the video."
            )

        # Build timeline (group by object_id, sort by time)
        timeline_by_object = {}
        for result in results:
            if result.object_id not in timeline_by_object:
                timeline_by_object[result.object_id] = []
            timeline_by_object[result.object_id].append(result)

        # Sort each object's timeline
        for obj_id in timeline_by_object:
            timeline_by_object[obj_id].sort(key=lambda x: x.timestamp)

        # Find the most prominent object (most appearances)
        best_obj_id = max(timeline_by_object.keys(), key=lambda oid: len(timeline_by_object[oid]))
        timeline_events = timeline_by_object[best_obj_id]

        timeline = [
            {
                'timestamp': event.timestamp,
                'frame_number': event.frame_number,
                'label': event.label,
                'confidence': event.confidence
            }
            for event in timeline_events
        ]

        # Apply temporal filter
        if query.temporal_constraint:
            timeline = self._filter_timeline_temporal(timeline, query.temporal_constraint)

        summary = self._generate_timeline_summary(target, timeline)

        return QueryResult(
            query_type="timeline",
            timeline=timeline,
            summary=summary
        )

    def _execute_reasoning(self, query: StructuredQuery) -> QueryResult:
        """Execute reasoning query (delegated to reasoning engine)"""
        # This will be implemented by the reasoning engine
        return QueryResult(
            query_type="reasoning",
            reasoning_answer="Reasoning queries require the reasoning engine module.",
            summary="Reasoning not yet implemented in executor."
        )

    def _filter_temporal(self, results: List[SearchResult], constraint) -> List[SearchResult]:
        """Filter results by temporal constraint"""
        filtered = []
        for result in results:
            if constraint.start_time is not None and result.timestamp < constraint.start_time:
                continue
            if constraint.end_time is not None and result.timestamp > constraint.end_time:
                continue
            filtered.append(result)
        return filtered

    def _filter_interactions_temporal(self, interactions: List[InteractionResult], constraint) -> List[InteractionResult]:
        """Filter interactions by temporal constraint"""
        filtered = []
        for interaction in interactions:
            if constraint.start_time is not None and interaction.end_time < constraint.start_time:
                continue
            if constraint.end_time is not None and interaction.start_time > constraint.end_time:
                continue
            filtered.append(interaction)
        return filtered

    def _filter_timeline_temporal(self, timeline: List[Dict], constraint) -> List[Dict]:
        """Filter timeline events by temporal constraint"""
        filtered = []
        for event in timeline:
            if constraint.start_time is not None and event['timestamp'] < constraint.start_time:
                continue
            if constraint.end_time is not None and event['timestamp'] > constraint.end_time:
                continue
            filtered.append(event)
        return filtered

    def _generate_search_summary(self, query: StructuredQuery, results: List[SearchResult]) -> str:
        """Generate natural language summary for search results"""
        target = ', '.join(query.target_objects)
        count = len(results)

        if count == 0:
            return f"No instances of '{target}' found."

        # Count unique objects
        unique_ids = set(r.object_id for r in results)
        unique_count = len(unique_ids)

        # Get time range
        timestamps = [r.timestamp for r in results]
        first_seen = min(timestamps)
        last_seen = max(timestamps)

        summary = f"Found {count} instance(s) of '{target}' across {unique_count} unique object(s). "
        summary += f"First seen at {first_seen:.1f}s, last seen at {last_seen:.1f}s."

        return summary

    def _generate_interaction_summary(
        self,
        query: StructuredQuery,
        objects_map: Dict,
        interactions: List[InteractionResult]
    ) -> str:
        """Generate summary for interaction query"""
        if not interactions:
            targets = ', '.join(query.target_objects)
            return f"No interactions found involving {targets}."

        count = len(interactions)
        interaction_types = set(i.interaction_type for i in interactions)

        summary = f"Found {count} interaction(s) of type: {', '.join(interaction_types)}. "

        if interactions:
            first = min(interactions, key=lambda x: x.start_time)
            last = max(interactions, key=lambda x: x.end_time)
            summary += f"Interactions span from {first.start_time:.1f}s to {last.end_time:.1f}s."

        return summary

    def _generate_timeline_summary(self, target: str, timeline: List[Dict]) -> str:
        """Generate summary for timeline query"""
        if not timeline:
            return f"'{target}' was not detected in the video."

        count = len(timeline)
        first = timeline[0]['timestamp']
        last = timeline[-1]['timestamp']

        summary = f"'{target}' appeared {count} time(s). "
        summary += f"First appearance at {first:.1f}s, last appearance at {last:.1f}s."

        return summary
