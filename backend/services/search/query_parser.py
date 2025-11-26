"""
Natural language query parser
Converts user queries into structured search operations
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from backend.core.logging import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries supported"""
    SEARCH = "search"              # Find objects by label
    INTERACTION = "interaction"    # Find interactions between objects
    TIMELINE = "timeline"          # Track object over time
    REASONING = "reasoning"        # Complex reasoning query


@dataclass
class TemporalConstraint:
    """Temporal constraints for queries"""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SpatialConstraint:
    """Spatial constraints for queries"""
    proximity_threshold: Optional[float] = None  # Max distance in pixels
    region: Optional[Dict] = None  # Bounding box region

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StructuredQuery:
    """Structured representation of a natural language query"""
    query_type: QueryType
    target_objects: List[str]
    interaction_types: List[str] = None
    temporal_constraint: TemporalConstraint = None
    spatial_constraint: SpatialConstraint = None
    original_query: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'query_type': self.query_type.value,
            'target_objects': self.target_objects,
            'interaction_types': self.interaction_types or [],
            'temporal_constraint': self.temporal_constraint.to_dict() if self.temporal_constraint else {},
            'spatial_constraint': self.spatial_constraint.to_dict() if self.spatial_constraint else {},
            'original_query': self.original_query,
            'confidence': self.confidence
        }


class QueryParser:
    """
    Parses natural language queries using hybrid LLM + rule-based approach
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize query parser

        Args:
            use_llm: Whether to use LLM for parsing (falls back to rules if False)
        """
        self.use_llm = use_llm
        self.llm_client = None

        if use_llm:
            self.llm_client = self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client for query parsing"""
        try:
            provider = os.getenv("LLM_PROVIDER", "openai")

            if provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set, falling back to rule-based parsing")
                    return None
                return openai.OpenAI(api_key=api_key)

            elif provider == "gemini":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("GEMINI_API_KEY not set, falling back to rule-based parsing")
                    return None
                genai.configure(api_key=api_key)
                return genai.GenerativeModel('gemini-2.0-flash-exp')

            else:
                logger.warning(f"Unknown LLM provider: {provider}, falling back to rules")
                return None

        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}, falling back to rules")
            return None

    def parse(self, query: str) -> StructuredQuery:
        """
        Parse natural language query into structured format

        Args:
            query: Natural language query

        Returns:
            StructuredQuery object
        """
        logger.info(f"Parsing query: {query}")

        # Try LLM-based parsing first
        if self.use_llm and self.llm_client:
            try:
                structured = self._parse_with_llm(query)
                if structured:
                    return structured
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, falling back to rules")

        # Fallback to rule-based parsing
        return self._parse_with_rules(query)

    def _parse_with_llm(self, query: str) -> Optional[StructuredQuery]:
        """Parse query using LLM"""
        system_prompt = """You are a query parser for a video knowledge graph system.
Parse user queries into structured JSON format.

Output ONLY valid JSON with this exact structure:
{
  "query_type": "search" | "interaction" | "timeline" | "reasoning",
  "target_objects": ["object_name", ...],
  "interaction_types": ["proximity", "occlusion", "following", ...],
  "temporal_constraint": {
    "start_time": <seconds or null>,
    "end_time": <seconds or null>
  },
  "spatial_constraint": {
    "proximity_threshold": <pixels or null>
  }
}

Query types:
- "search": Find objects by name/type (e.g., "find all cats")
- "interaction": Find interactions between objects (e.g., "when does person pick up cup")
- "timeline": Track object over time (e.g., "where was the red bag last seen")
- "reasoning": Complex questions requiring reasoning (e.g., "what happens in this scene")

Examples:
Input: "find every cat in the video"
Output: {"query_type": "search", "target_objects": ["cat"], "interaction_types": [], "temporal_constraint": {}, "spatial_constraint": {}}

Input: "when does the person pick up the cup?"
Output: {"query_type": "interaction", "target_objects": ["person", "cup"], "interaction_types": ["proximity", "occlusion"], "temporal_constraint": {}, "spatial_constraint": {"proximity_threshold": 50}}

Input: "show all interactions involving a phone"
Output: {"query_type": "interaction", "target_objects": ["phone"], "interaction_types": [], "temporal_constraint": {}, "spatial_constraint": {}}
"""

        user_prompt = f"Parse this query: {query}"

        try:
            provider = os.getenv("LLM_PROVIDER", "openai")

            if provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast model for parsing
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content

            else:  # gemini
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.llm_client.generate_content(
                    full_prompt,
                    generation_config={"temperature": 0.0, "max_output_tokens": 500}
                )
                result_text = response.text

            # Extract JSON from response
            result_text = result_text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(result_text)

            # Convert to StructuredQuery
            query_type = QueryType(parsed.get('query_type', 'search'))
            target_objects = parsed.get('target_objects', [])
            interaction_types = parsed.get('interaction_types', [])

            temporal = parsed.get('temporal_constraint', {})
            temporal_constraint = TemporalConstraint(
                start_time=temporal.get('start_time'),
                end_time=temporal.get('end_time')
            ) if temporal else None

            spatial = parsed.get('spatial_constraint', {})
            spatial_constraint = SpatialConstraint(
                proximity_threshold=spatial.get('proximity_threshold')
            ) if spatial else None

            return StructuredQuery(
                query_type=query_type,
                target_objects=target_objects,
                interaction_types=interaction_types,
                temporal_constraint=temporal_constraint,
                spatial_constraint=spatial_constraint,
                original_query=query,
                confidence=0.9
            )

        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
            return None

    def _parse_with_rules(self, query: str) -> StructuredQuery:
        """Parse query using rule-based patterns"""
        query_lower = query.lower()

        # Detect query type
        query_type = QueryType.SEARCH

        if any(word in query_lower for word in ['interact', 'pick up', 'touch', 'near', 'with']):
            query_type = QueryType.INTERACTION
        elif any(word in query_lower for word in ['track', 'follow', 'timeline', 'where was', 'last seen']):
            query_type = QueryType.TIMELINE
        elif any(word in query_lower for word in ['what happens', 'explain', 'describe', 'analyze']):
            query_type = QueryType.REASONING

        # Extract object names (common nouns)
        common_objects = [
            'person', 'people', 'man', 'woman', 'child', 'cat', 'dog', 'car', 'phone',
            'cup', 'bottle', 'chair', 'table', 'laptop', 'bag', 'book', 'ball',
            'bird', 'bicycle', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'backpack', 'umbrella', 'handbag', 'suitcase', 'sports ball', 'skateboard'
        ]

        target_objects = []
        for obj in common_objects:
            if obj in query_lower:
                target_objects.append(obj)

        # If no objects found, extract nouns (simple heuristic)
        if not target_objects:
            # Look for words after "find", "show", "track"
            patterns = [
                r'find (?:all |every |the )?(\w+)',
                r'show (?:all |every |the )?(\w+)',
                r'track (?:the )?(\w+)',
                r'where (?:is |was )?(?:the )?(\w+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    target_objects.append(match.group(1))
                    break

        # Extract interaction types
        interaction_types = []
        if 'near' in query_lower or 'close' in query_lower or 'proximity' in query_lower:
            interaction_types.append('proximity')
        if 'pick up' in query_lower or 'touch' in query_lower or 'grab' in query_lower:
            interaction_types.extend(['proximity', 'occlusion'])
        if 'follow' in query_lower or 'chase' in query_lower:
            interaction_types.append('following')
        if 'overlap' in query_lower or 'occlude' in query_lower:
            interaction_types.append('occlusion')

        # Extract temporal constraints
        temporal_constraint = None
        time_patterns = [
            (r'at (\d+(?:\.\d+)?) seconds?', 'exact'),
            (r'after (\d+(?:\.\d+)?) seconds?', 'after'),
            (r'before (\d+(?:\.\d+)?) seconds?', 'before'),
            (r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)', 'range')
        ]

        for pattern, constraint_type in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if constraint_type == 'exact':
                    t = float(match.group(1))
                    temporal_constraint = TemporalConstraint(start_time=t, end_time=t + 1)
                elif constraint_type == 'after':
                    temporal_constraint = TemporalConstraint(start_time=float(match.group(1)))
                elif constraint_type == 'before':
                    temporal_constraint = TemporalConstraint(end_time=float(match.group(1)))
                elif constraint_type == 'range':
                    temporal_constraint = TemporalConstraint(
                        start_time=float(match.group(1)),
                        end_time=float(match.group(2))
                    )
                break

        return StructuredQuery(
            query_type=query_type,
            target_objects=target_objects or ['unknown'],
            interaction_types=interaction_types,
            temporal_constraint=temporal_constraint,
            spatial_constraint=None,
            original_query=query,
            confidence=0.7  # Lower confidence for rule-based
        )
