"""
Neo4j client for knowledge graph storage and querying
"""

from typing import Optional, Dict, List, Any
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Neo4j database client for knowledge graph operations
    """

    def __init__(self):
        """
        Initialize Neo4j driver
        """
        self.driver: Driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=120,
        )
        self._verify_connectivity()
        self._create_constraints()

    def _verify_connectivity(self):
        """
        Verify Neo4j connection
        """
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connection established")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _create_constraints(self):
        """
        Create database constraints and indexes
        """
        constraints = [
            # Unique constraints
            "CREATE CONSTRAINT video_id_unique IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
            "CREATE CONSTRAINT object_unique IF NOT EXISTS FOR (o:Object) REQUIRE (o.video_id, o.object_id) IS UNIQUE",
            "CREATE CONSTRAINT event_unique IF NOT EXISTS FOR (e:Event) REQUIRE (e.video_id, e.event_id) IS UNIQUE",
            # Indexes for performance
            "CREATE INDEX object_frame_idx IF NOT EXISTS FOR (o:Object) ON (o.frame_number)",
            "CREATE INDEX event_time_idx IF NOT EXISTS FOR (e:Event) ON (e.start_frame, e.end_frame)",
            "CREATE INDEX interaction_type_idx IF NOT EXISTS FOR ()-[r:INTERACTS_WITH]-() ON (r.interaction_type)",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint/index: {constraint[:50]}...")
                except Neo4jError as e:
                    # Ignore if already exists
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create constraint: {e}")

    def close(self):
        """
        Close Neo4j driver
        """
        self.driver.close()
        logger.info("Neo4j connection closed")

    # Video graph operations

    def create_video_node(self, video_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Create video node in graph

        Args:
            video_id: Video ID
            metadata: Video metadata

        Returns:
            True if successful
        """
        query = """
        MERGE (v:Video {video_id: $video_id})
        SET v += $metadata
        RETURN v
        """
        try:
            with self.driver.session() as session:
                session.run(query, video_id=video_id, metadata=metadata)
            logger.debug(f"Created video node: {video_id}")
            return True
        except Neo4jError as e:
            logger.error(f"Failed to create video node: {e}", video_id=video_id)
            return False

    def create_object_node(
        self, video_id: int, object_id: int, frame_number: int, properties: Dict[str, Any]
    ) -> bool:
        """
        Create object node in graph

        Args:
            video_id: Video ID
            object_id: Object tracking ID
            frame_number: Frame number where object appears
            properties: Object properties (bbox, embedding, label, etc.)

        Returns:
            True if successful
        """
        query = """
        MATCH (v:Video {video_id: $video_id})
        MERGE (o:Object {video_id: $video_id, object_id: $object_id})
        SET o += $properties
        SET o.frame_number = $frame_number
        MERGE (v)-[:CONTAINS]->(o)
        RETURN o
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    video_id=video_id,
                    object_id=object_id,
                    frame_number=frame_number,
                    properties=properties,
                )
            return True
        except Neo4jError as e:
            logger.error(f"Failed to create object node: {e}", object_id=object_id)
            return False

    def create_interaction_edge(
        self,
        video_id: int,
        object_id_1: int,
        object_id_2: int,
        interaction_type: str,
        properties: Dict[str, Any],
    ) -> bool:
        """
        Create interaction relationship between objects

        Args:
            video_id: Video ID
            object_id_1: First object ID
            object_id_2: Second object ID
            interaction_type: Type of interaction
            properties: Interaction properties (confidence, frames, etc.)

        Returns:
            True if successful
        """
        query = """
        MATCH (o1:Object {video_id: $video_id, object_id: $object_id_1})
        MATCH (o2:Object {video_id: $video_id, object_id: $object_id_2})
        MERGE (o1)-[r:INTERACTS_WITH]->(o2)
        SET r.interaction_type = $interaction_type
        SET r += $properties
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    video_id=video_id,
                    object_id_1=object_id_1,
                    object_id_2=object_id_2,
                    interaction_type=interaction_type,
                    properties=properties,
                )
            return True
        except Neo4jError as e:
            logger.error(
                f"Failed to create interaction edge: {e}",
                object_1=object_id_1,
                object_2=object_id_2,
            )
            return False

    def create_temporal_edge(
        self, video_id: int, object_id_1: int, object_id_2: int, properties: Dict[str, Any]
    ) -> bool:
        """
        Create temporal relationship (before/after)

        Args:
            video_id: Video ID
            object_id_1: Earlier object ID
            object_id_2: Later object ID
            properties: Temporal properties

        Returns:
            True if successful
        """
        query = """
        MATCH (o1:Object {video_id: $video_id, object_id: $object_id_1})
        MATCH (o2:Object {video_id: $video_id, object_id: $object_id_2})
        MERGE (o1)-[r:BEFORE]->(o2)
        SET r += $properties
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    video_id=video_id,
                    object_id_1=object_id_1,
                    object_id_2=object_id_2,
                    properties=properties,
                )
            return True
        except Neo4jError as e:
            logger.error(f"Failed to create temporal edge: {e}")
            return False

    def create_causal_edge(
        self, video_id: int, cause_id: int, effect_id: int, properties: Dict[str, Any]
    ) -> bool:
        """
        Create causal relationship

        Args:
            video_id: Video ID
            cause_id: Cause object/event ID
            effect_id: Effect object/event ID
            properties: Causal properties

        Returns:
            True if successful
        """
        query = """
        MATCH (c:Object {video_id: $video_id, object_id: $cause_id})
        MATCH (e:Object {video_id: $video_id, object_id: $effect_id})
        MERGE (c)-[r:CAUSES]->(e)
        SET r += $properties
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    video_id=video_id,
                    cause_id=cause_id,
                    effect_id=effect_id,
                    properties=properties,
                )
            return True
        except Neo4jError as e:
            logger.error(f"Failed to create causal edge: {e}")
            return False

    # Query operations

    def get_video_graph(self, video_id: int) -> Dict[str, Any]:
        """
        Get complete knowledge graph for video

        Args:
            video_id: Video ID

        Returns:
            Graph data with nodes and relationships
        """
        query = """
        MATCH (v:Video {video_id: $video_id})-[:CONTAINS]->(o:Object)
        OPTIONAL MATCH (o)-[r]->(o2:Object)
        RETURN v, collect(DISTINCT o) as objects, collect(DISTINCT r) as relationships
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, video_id=video_id)
                record = result.single()
                if record:
                    return {
                        "video": dict(record["v"]),
                        "objects": [dict(obj) for obj in record["objects"]],
                        "relationships": [dict(rel) for rel in record["relationships"] if rel],
                    }
                return {"video": None, "objects": [], "relationships": []}
        except Neo4jError as e:
            logger.error(f"Failed to get video graph: {e}", video_id=video_id)
            return {"video": None, "objects": [], "relationships": []}

    def get_object_interactions(self, video_id: int, object_id: int) -> List[Dict[str, Any]]:
        """
        Get all interactions for an object

        Args:
            video_id: Video ID
            object_id: Object ID

        Returns:
            List of interactions
        """
        query = """
        MATCH (o:Object {video_id: $video_id, object_id: $object_id})-[r:INTERACTS_WITH]-(o2:Object)
        RETURN o2, r, type(r) as rel_type
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, video_id=video_id, object_id=object_id)
                return [
                    {
                        "object": dict(record["o2"]),
                        "relationship": dict(record["r"]),
                        "type": record["rel_type"],
                    }
                    for record in result
                ]
        except Neo4jError as e:
            logger.error(f"Failed to get object interactions: {e}", object_id=object_id)
            return []

    def find_objects_by_label(self, video_id: int, label: str) -> List[Dict[str, Any]]:
        """
        Find objects by CLIP label

        Args:
            video_id: Video ID
            label: Object label

        Returns:
            List of matching objects
        """
        query = """
        MATCH (o:Object {video_id: $video_id})
        WHERE o.clip_label CONTAINS $label
        RETURN o
        ORDER BY o.frame_number
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, video_id=video_id, label=label)
                return [dict(record["o"]) for record in result]
        except Neo4jError as e:
            logger.error(f"Failed to find objects by label: {e}", label=label)
            return []

    def get_temporal_sequence(self, video_id: int) -> List[Dict[str, Any]]:
        """
        Get temporal sequence of events

        Args:
            video_id: Video ID

        Returns:
            Ordered list of temporal relationships
        """
        query = """
        MATCH (o1:Object {video_id: $video_id})-[r:BEFORE]->(o2:Object)
        RETURN o1, r, o2
        ORDER BY r.timestamp
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, video_id=video_id)
                return [
                    {
                        "from": dict(record["o1"]),
                        "relationship": dict(record["r"]),
                        "to": dict(record["o2"]),
                    }
                    for record in result
                ]
        except Neo4jError as e:
            logger.error(f"Failed to get temporal sequence: {e}")
            return []

    def compute_graph_metrics(self, video_id: int) -> Dict[str, Any]:
        """
        Compute graph metrics (centrality, density, etc.)

        Args:
            video_id: Video ID

        Returns:
            Dictionary of metrics
        """
        queries = {
            "node_count": """
                MATCH (o:Object {video_id: $video_id})
                RETURN count(o) as count
            """,
            "edge_count": """
                MATCH (o1:Object {video_id: $video_id})-[r]-(o2:Object)
                RETURN count(r) as count
            """,
            "avg_degree": """
                MATCH (o:Object {video_id: $video_id})
                OPTIONAL MATCH (o)-[r]-()
                WITH o, count(r) as degree
                RETURN avg(degree) as avg_degree
            """,
        }

        metrics = {}
        try:
            with self.driver.session() as session:
                for metric_name, query in queries.items():
                    result = session.run(query, video_id=video_id)
                    record = result.single()
                    if record:
                        metrics[metric_name] = record[0]
            return metrics
        except Neo4jError as e:
            logger.error(f"Failed to compute graph metrics: {e}")
            return {}

    def delete_video_graph(self, video_id: int) -> bool:
        """
        Delete entire video graph

        Args:
            video_id: Video ID

        Returns:
            True if successful
        """
        query = """
        MATCH (v:Video {video_id: $video_id})
        OPTIONAL MATCH (v)-[:CONTAINS]->(o:Object)
        DETACH DELETE v, o
        """
        try:
            with self.driver.session() as session:
                session.run(query, video_id=video_id)
            logger.info(f"Deleted video graph: {video_id}")
            return True
        except Neo4jError as e:
            logger.error(f"Failed to delete video graph: {e}", video_id=video_id)
            return False


# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """
    Get singleton Neo4j client instance
    """
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
