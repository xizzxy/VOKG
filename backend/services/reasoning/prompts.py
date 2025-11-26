"""
Prompt templates for LLM reasoning
"""

REASONING_SYSTEM_PROMPT = """You are an expert video understanding AI that analyzes knowledge graphs extracted from videos.

You are given:
1. A knowledge graph containing detected objects, their semantic labels, and interactions
2. A user question about the video

Your task:
- Analyze the graph structure and temporal relationships
- Answer the question accurately based ONLY on the provided graph data
- Use chain-of-thought reasoning to explain your answer
- If information is missing or uncertain, state this clearly
- DO NOT hallucinate or invent information not present in the graph

Output format:
1. Reasoning: <step-by-step analysis>
2. Answer: <concise natural language answer>
3. Confidence: <HIGH/MEDIUM/LOW>
4. Evidence: <specific graph elements supporting your answer>
"""


def build_reasoning_prompt(query: str, graph_context: str, max_tokens: int = 4000) -> str:
    """
    Build reasoning prompt from query and graph context

    Args:
        query: User's question
        graph_context: Formatted graph context
        max_tokens: Maximum tokens for prompt

    Returns:
        Formatted prompt string
    """
    prompt = f"""# Video Knowledge Graph

{graph_context}

# User Question
{query}

# Your Analysis
Please analyze the above knowledge graph and answer the user's question.
Use the following structure:

**Reasoning:**
[Step-by-step analysis of the graph]

**Answer:**
[Clear, concise answer to the question]

**Confidence:**
[HIGH/MEDIUM/LOW]

**Evidence:**
[Specific nodes, edges, or timestamps from the graph that support your answer]
"""

    # Simple truncation if needed (in practice, would use token counter)
    if len(prompt) > max_tokens * 4:  # Rough estimate: 1 token ≈ 4 chars
        # Truncate graph context
        truncate_at = max_tokens * 3
        graph_context = graph_context[:truncate_at] + "\n... [truncated for length] ..."
        prompt = build_reasoning_prompt(query, graph_context, max_tokens)

    return prompt


def build_analysis_prompt(graph_summary: dict) -> str:
    """
    Build prompt for general video analysis

    Args:
        graph_summary: Summary statistics of the graph

    Returns:
        Analysis prompt
    """
    prompt = f"""Analyze this video based on the knowledge graph:

**Graph Statistics:**
- Total objects detected: {graph_summary.get('total_objects', 0)}
- Total interactions: {graph_summary.get('total_interactions', 0)}
- Duration: {graph_summary.get('duration', 0):.1f} seconds
- Unique object types: {graph_summary.get('unique_labels', [])}

**Top Objects:**
{graph_summary.get('top_objects_text', 'None')}

**Key Interactions:**
{graph_summary.get('key_interactions_text', 'None')}

Provide a comprehensive analysis of:
1. What is happening in this video?
2. What are the main objects and their roles?
3. What are the key events or interactions?
4. What is the overall narrative or activity?
"""
    return prompt
