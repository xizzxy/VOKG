"""
Example usage of VOKG v2 system
Demonstrates key features and API
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables (for demonstration)
os.environ.setdefault("LLM_PROVIDER", "openai")

print("="*70)
print("VOKG v2 - Example Usage")
print("="*70)
print()

# ============================================================================
# Example 1: Semantic Labeling with CLIP
# ============================================================================
print("Example 1: Semantic Object Labeling")
print("-" * 70)

try:
    from backend.services.semantic import CLIPEncoder, SemanticLabeler
    from PIL import Image
    import numpy as np

    # Initialize CLIP
    print("Initializing CLIP encoder...")
    clip_encoder = CLIPEncoder(model_name="ViT-B/32")
    labeler = SemanticLabeler(clip_encoder)

    # Create a dummy image (in practice, this would be from a video frame)
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    print("Labeling object...")
    label = labeler.label_object(
        image_crop=dummy_image,
        object_id=1,
        frame_number=0,
        timestamp=0.0
    )

    print(f"✓ Primary Label: {label.primary_label}")
    print(f"  Confidence: {label.confidence:.2f}")
    print(f"  Category: {label.category}")
    print(f"  Super-category: {label.super_category}")
    print(f"  Alternative labels: {[l for l, _ in label.alternative_labels[:3]]}")
    print()

except ImportError as e:
    print(f"⚠ CLIP not available: {e}")
    print("  Install with: pip install git+https://github.com/openai/CLIP.git")
    print()

# ============================================================================
# Example 2: Embedding Search
# ============================================================================
print("Example 2: Embedding-based Similarity Search")
print("-" * 70)

try:
    from backend.services.search import EmbeddingStore

    # Initialize embedding store
    print("Initializing FAISS embedding store...")
    store = EmbeddingStore(embedding_dim=512, index_type="flat", metric="cosine")

    # Add some dummy embeddings
    embeddings = np.random.randn(10, 512).astype(np.float32)
    metadata = [
        {
            'object_id': i,
            'label': f'object_{i}',
            'confidence': 0.9,
            'category': 'test',
            'frame_number': i,
            'timestamp': i * 0.5
        }
        for i in range(10)
    ]

    store.add(embeddings, metadata)
    print(f"✓ Added {len(embeddings)} embeddings to index")

    # Search for similar objects
    query_embedding = np.random.randn(512).astype(np.float32)
    results = store.search(query_embedding, k=3)

    print(f"✓ Top 3 similar objects:")
    for r in results:
        print(f"  - {r.label} (similarity: {r.similarity_score:.3f})")
    print()

except ImportError as e:
    print(f"⚠ FAISS not available: {e}")
    print("  Install with: pip install faiss-cpu")
    print()

# ============================================================================
# Example 3: Natural Language Query Parsing
# ============================================================================
print("Example 3: Natural Language Query Parsing")
print("-" * 70)

from backend.services.search import QueryParser

# Initialize parser (will work without LLM using rule-based fallback)
print("Initializing query parser...")
parser = QueryParser(use_llm=False)  # Use rules for this example

# Test various queries
test_queries = [
    "find all cats in the video",
    "when does the person pick up the cup?",
    "show interactions involving a phone",
    "track the red car"
]

for query in test_queries:
    structured = parser.parse(query)
    print(f"\nQuery: '{query}'")
    print(f"  Type: {structured.query_type.value}")
    print(f"  Target objects: {structured.target_objects}")
    if structured.interaction_types:
        print(f"  Interaction types: {structured.interaction_types}")

print()

# ============================================================================
# Example 4: Knowledge Graph Construction
# ============================================================================
print("Example 4: Enhanced Knowledge Graph")
print("-" * 70)

from backend.graph import EnhancedKnowledgeGraph

# Create graph
print("Building knowledge graph...")
graph = EnhancedKnowledgeGraph(video_id=1, fps=30.0)

# Add some objects with semantic labels
for i in range(5):
    graph.add_object(
        object_id=i,
        frame_number=i * 10,
        timestamp=i * 0.33,
        bbox=[10 + i*10, 10, 50 + i*10, 50],
        semantic_label={
            'primary_label': ['person', 'car', 'phone', 'cup', 'chair'][i],
            'confidence': 0.85 + i * 0.02,
            'category': ['human', 'vehicle', 'electronics', 'container', 'furniture'][i],
            'super_category': ['living thing', 'transportation', 'object', 'object', 'indoor'][i]
        }
    )

# Add some interactions
graph.add_interaction(0, 2, 'proximity', 10, 0.33, 0.9)
graph.add_interaction(0, 3, 'occlusion', 20, 0.66, 0.7)

print(f"✓ Graph created:")
stats = graph.get_statistics()
print(f"  Objects: {stats['total_objects']}")
print(f"  Interactions: {stats['total_interactions']}")
print(f"  Unique labels: {stats['unique_labels']}")
print(f"  Duration: {stats['duration']:.2f}s")

# Query the graph
print(f"\n✓ Querying graph:")
person_objects = graph.get_objects_by_label('person')
print(f"  Found {len(person_objects)} person object(s)")

interactions = graph.get_interactions_for_object(0)
print(f"  Object 0 has {len(interactions)} interaction(s)")

print()

# ============================================================================
# Example 5: LLM Reasoning (if API key available)
# ============================================================================
print("Example 5: LLM-Powered Reasoning")
print("-" * 70)

has_openai = bool(os.getenv("OPENAI_API_KEY"))
has_gemini = bool(os.getenv("GEMINI_API_KEY"))

if has_openai or has_gemini:
    try:
        from backend.services.reasoning import ReasoningEngine

        provider = "openai" if has_openai else "gemini"
        print(f"Initializing reasoning engine ({provider})...")

        engine = ReasoningEngine(provider=provider)

        # Prepare data
        objects = graph.get_all_objects()
        interactions = graph.get_all_interactions()

        print("Analyzing video content...")
        result = engine.analyze_video(objects, interactions)

        print(f"\n✓ Analysis complete:")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Confidence: {result.confidence}")
        print()

    except Exception as e:
        print(f"⚠ Reasoning failed: {e}")
        print()
else:
    print("⚠ No LLM API key found")
    print("  Set OPENAI_API_KEY or GEMINI_API_KEY to enable reasoning")
    print()

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("Summary")
print("="*70)
print()
print("VOKG v2 provides:")
print("  ✓ Semantic object labeling with CLIP")
print("  ✓ Fast embedding-based search with FAISS")
print("  ✓ Natural language query understanding")
print("  ✓ Enhanced knowledge graph with semantics")
print("  ✓ LLM-powered reasoning (with API key)")
print()
print("For full pipeline demonstration:")
print("  python scripts/test_vokg_v2_pipeline.py --num-clips 2")
print()
print("For documentation:")
print("  See VOKG_V2_README.md")
print()
