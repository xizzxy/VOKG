"""
LLM prompts for video understanding and narrative generation
"""

SYSTEM_PROMPT = """You are an expert video analyst specializing in understanding object interactions, temporal relationships, and causal patterns in videos.

Your task is to analyze a knowledge graph extracted from a video and generate a comprehensive narrative that explains:
1. What objects are present and their characteristics
2. How objects interact with each other
3. The temporal sequence of events
4. Causal relationships between events
5. Overall summary of the video's content

Be specific, factual, and cite frame numbers and timestamps when relevant. Focus on describing observable behaviors and interactions, not speculation."""


def create_analysis_prompt(graph_data: dict, video_metadata: dict) -> str:
    """
    Create prompt for video analysis

    Args:
        graph_data: Knowledge graph data
        video_metadata: Video metadata

    Returns:
        Formatted prompt
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Summarize objects
    object_summary = []
    object_types = {}
    for node in nodes[:50]:  # Limit to first 50 for prompt length
        obj_id = node.get("id")
        label = node.get("label", "unknown")
        frame = node.get("properties", {}).get("frame_number", 0)

        object_types[label] = object_types.get(label, 0) + 1
        object_summary.append(f"- Object {obj_id} ({label}) appears at frame {frame}")

    # Summarize interactions
    interaction_summary = []
    interaction_types = {}
    for edge in edges[:50]:  # Limit to first 50
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type", "interacts")
        start_frame = edge.get("properties", {}).get("start_frame", 0)
        end_frame = edge.get("properties", {}).get("end_frame", 0)

        interaction_types[edge_type] = interaction_types.get(edge_type, 0) + 1
        interaction_summary.append(
            f"- {source} {edge_type} {target} (frames {start_frame}-{end_frame})"
        )

    prompt = f"""## Video Metadata
- Duration: {video_metadata.get('duration', 0):.2f} seconds
- Resolution: {video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}
- FPS: {video_metadata.get('fps', 0):.2f}
- Total Frames: {video_metadata.get('total_frames', 0)}

## Object Statistics
Total Objects: {len(nodes)}
Object Types:
{chr(10).join(f"- {label}: {count}" for label, count in sorted(object_types.items(), key=lambda x: -x[1])[:10])}

## Detected Objects (Sample)
{chr(10).join(object_summary[:20])}

## Interaction Statistics
Total Interactions: {len(edges)}
Interaction Types:
{chr(10).join(f"- {itype}: {count}" for itype, count in sorted(interaction_types.items(), key=lambda x: -x[1]))}

## Detected Interactions (Sample)
{chr(10).join(interaction_summary[:20])}

## Task
Based on this knowledge graph data, generate a comprehensive narrative that:

1. **Summary**: Provide a 2-3 sentence overview of what happens in the video

2. **Object Descriptions**: Describe the main objects/actors and their characteristics

3. **Temporal Sequence**: Describe the sequence of events in chronological order

4. **Interactions**: Explain how objects interact with each other (proximity, containment, chasing, following, etc.)

5. **Causal Analysis**: Identify cause-and-effect relationships between events

6. **Key Events**: Highlight the most significant moments in the video

Format your response as a well-structured narrative with clear sections. Be specific and reference frame numbers/timestamps where appropriate."""

    return prompt


def create_critique_prompt(narrative: str) -> str:
    """
    Create prompt for self-critique

    Args:
        narrative: Generated narrative

    Returns:
        Critique prompt
    """
    return f"""Review the following video analysis narrative and identify any issues:

{narrative}

Critique the narrative for:
1. Factual accuracy (does it match the data?)
2. Clarity and coherence
3. Missing important details
4. Over-speculation or unsupported claims
5. Temporal consistency

Provide specific suggestions for improvement."""


def create_revision_prompt(narrative: str, critique: str, graph_data: dict) -> str:
    """
    Create prompt for narrative revision

    Args:
        narrative: Original narrative
        critique: Critique feedback
        graph_data: Knowledge graph data

    Returns:
        Revision prompt
    """
    return f"""Original Narrative:
{narrative}

Critique:
{critique}

Based on this critique and the original knowledge graph data, revise the narrative to address the identified issues. Maintain the same structure but improve accuracy, clarity, and completeness."""
