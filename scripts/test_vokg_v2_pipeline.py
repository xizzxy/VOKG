"""
VOKG v2 Testing Pipeline
Upgraded with semantic labeling, embedding search, and LLM reasoning
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import VOKG v2 modules
from backend.services.semantic import CLIPEncoder, SemanticLabeler, LabelHierarchy, COCO_CATEGORIES
from backend.services.search import EmbeddingStore, QueryParser, QueryExecutor
from backend.services.reasoning import ReasoningEngine, ContextBuilder
from backend.graph import EnhancedKnowledgeGraph


class SimpleObjectTracker:
    """IoU-based object tracker for consistent IDs across frames"""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}
        self.max_age = 10

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]

        if not detections:
            return []

        # Match detections to existing tracks
        if self.tracks:
            matches, unmatched = self._match_detections(detections)
        else:
            matches = []
            unmatched = list(range(len(detections)))

        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id]["bbox"] = detections[det_idx]["bbox"]
            self.tracks[track_id]["age"] = 0
            detections[det_idx]["object_id"] = track_id

        # Create new tracks
        for det_idx in unmatched:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "bbox": detections[det_idx]["bbox"],
                "age": 0
            }
            detections[det_idx]["object_id"] = track_id

        return detections

    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple], List[int]]:
        """Match detections to tracks using IoU"""
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        for i, det in enumerate(detections):
            for j, track_bbox in enumerate(track_bboxes):
                iou_matrix[i, j] = self._compute_iou(det["bbox"], track_bbox)

        # Greedy matching
        matches = []
        unmatched = set(range(len(detections)))

        for det_idx in range(len(detections)):
            best_track_idx = np.argmax(iou_matrix[det_idx])
            if iou_matrix[det_idx, best_track_idx] >= self.iou_threshold:
                matches.append((det_idx, track_ids[best_track_idx]))
                unmatched.discard(det_idx)
                iou_matrix[:, best_track_idx] = 0  # Prevent reuse

        return matches, list(unmatched)

    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IoU between two bboxes"""
        x1_i = max(bbox1[0], bbox2[0])
        y1_i = max(bbox1[1], bbox2[1])
        x2_i = min(bbox1[2], bbox2[2])
        y2_i = min(bbox1[3], bbox2[3])

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class VOKGv2TestEngine:
    """Enhanced VOKG pipeline with semantic understanding and reasoning"""

    def __init__(self, clips_dir: str, use_clip: bool = True, use_llm: bool = True):
        """
        Initialize VOKG v2 test engine

        Args:
            clips_dir: Directory containing video clips
            use_clip: Whether to use CLIP for semantic labeling
            use_llm: Whether to use LLM for reasoning
        """
        self.clips_dir = Path(clips_dir)
        self.results = []
        self.use_clip = use_clip
        self.use_llm = use_llm

        # Initialize CLIP encoder
        if use_clip:
            try:
                print("Initializing CLIP encoder...")
                self.clip_encoder = CLIPEncoder(model_name="ViT-B/32")
                self.semantic_labeler = SemanticLabeler(
                    clip_encoder=self.clip_encoder,
                    candidate_labels=COCO_CATEGORIES
                )
                print("✓ CLIP encoder initialized")
            except Exception as e:
                print(f"⚠ CLIP initialization failed: {e}")
                print("  Falling back to basic labeling")
                self.use_clip = False
                self.clip_encoder = None
                self.semantic_labeler = None
        else:
            self.clip_encoder = None
            self.semantic_labeler = None

        # Initialize embedding store
        self.embedding_store = None
        if use_clip:
            try:
                self.embedding_store = EmbeddingStore(
                    embedding_dim=512,
                    index_type="flat",
                    metric="cosine"
                )
                print("✓ Embedding store initialized")
            except Exception as e:
                print(f"⚠ Embedding store initialization failed: {e}")
                self.embedding_store = None

        # Initialize reasoning engine
        self.reasoning_engine = None
        if use_llm:
            try:
                # Check for API keys
                has_openai = bool(os.getenv("OPENAI_API_KEY"))
                has_gemini = bool(os.getenv("GEMINI_API_KEY"))

                if has_openai or has_gemini:
                    provider = "openai" if has_openai else "gemini"
                    print(f"Initializing reasoning engine with {provider}...")
                    self.reasoning_engine = ReasoningEngine(provider=provider)
                    print(f"✓ Reasoning engine initialized ({provider})")
                else:
                    print("⚠ No LLM API keys found (OPENAI_API_KEY or GEMINI_API_KEY)")
                    print("  Reasoning features will be disabled")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠ Reasoning engine initialization failed: {e}")
                self.use_llm = False
                self.reasoning_engine = None

        print()

    def select_clips(self, num_clips: int = None) -> List[Path]:
        """Select diverse clips for testing"""
        all_clips = list(self.clips_dir.glob("*.mp4"))

        if not all_clips:
            raise ValueError(f"No MP4 files found in {self.clips_dir}")

        # Select clips
        if num_clips is None:
            num_clips = min(3, len(all_clips))  # Default: 3 clips

        selected = random.sample(all_clips, min(num_clips, len(all_clips)))

        print(f"Selected {len(selected)} clips from {len(all_clips)} available:")
        for clip in selected:
            print(f"  - {clip.name}")
        print()

        return selected

    def extract_frames(self, video_path: Path, max_frames: int = 30, fps: float = 2.0) -> List[np.ndarray]:
        """Extract frames from video using OpenCV"""
        print(f"  [1/6] Extracting frames...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))

        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_idx += 1

        cap.release()

        print(f"    ✓ Extracted {len(frames)} frames")
        return frames

    def segment_objects_sam2(self, frame: np.ndarray) -> List[Dict]:
        """Segment objects (simplified for testing)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.001
        max_area = (h * w) * 0.8

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            detection = {
                "bbox": [x, y, x + bw, y + bh],
                "area": float(area),
                "confidence": min(1.0, area / (bw * bh))
            }
            detections.append(detection)

        detections = sorted(detections, key=lambda x: x["area"], reverse=True)[:20]
        return detections

    def label_objects_semantic(
        self,
        frames: List[np.ndarray],
        objects_by_frame: Dict[int, List[Dict]]
    ) -> Dict[int, Dict]:
        """Label objects using CLIP"""
        print(f"  [3/6] Semantic labeling with CLIP...")

        all_crops = []
        all_metadata = []

        for frame_idx, frame_objects in objects_by_frame.items():
            frame = frames[frame_idx]

            for obj in frame_objects:
                bbox = obj["bbox"]
                x1, y1, x2, y2 = map(int, bbox)

                # Crop object region
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                all_crops.append(crop)
                all_metadata.append({
                    'object_id': obj['object_id'],
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / 2.0
                })

        if not all_crops:
            print("    ⚠ No crops to label")
            return {}

        # Batch label
        print(f"    Labeling {len(all_crops)} object instances...")
        labels = self.semantic_labeler.label_objects_batch(
            all_crops,
            [m['object_id'] for m in all_metadata],
            [m['frame_number'] for m in all_metadata],
            [m['timestamp'] for m in all_metadata]
        )

        # Group by object_id (take most confident label)
        labels_by_object = {}
        for label in labels:
            oid = label.object_id
            if oid not in labels_by_object or label.confidence > labels_by_object[oid].confidence:
                labels_by_object[oid] = label

        print(f"    ✓ Labeled {len(labels_by_object)} unique objects")
        return labels_by_object

    def build_embedding_index(self, semantic_labels: Dict[int, any]):
        """Build FAISS embedding index"""
        print(f"  [4/6] Building embedding search index...")

        embeddings = []
        metadata = []

        for object_id, label in semantic_labels.items():
            embeddings.append(label.embedding)
            metadata.append({
                'object_id': object_id,
                'label': label.primary_label,
                'confidence': label.confidence,
                'category': label.category,
                'frame_number': label.frame_number,
                'timestamp': label.timestamp,
                'embedding': label.embedding
            })

        if embeddings:
            embeddings_array = np.array(embeddings).astype(np.float32)
            self.embedding_store.add(embeddings_array, metadata)
            print(f"    ✓ Indexed {len(embeddings)} embeddings")

    def detect_interactions(self, objects_by_frame: Dict[int, List[Dict]]) -> List[Dict]:
        """Detect spatial and temporal interactions"""
        interactions = []

        # Spatial interactions (per frame)
        for frame_num, frame_objects in objects_by_frame.items():
            for i, obj1 in enumerate(frame_objects):
                for obj2 in frame_objects[i+1:]:
                    # Proximity detection
                    distance = self._compute_distance(obj1["bbox"], obj2["bbox"])
                    if distance < 150:
                        interactions.append({
                            "type": "proximity",
                            "object_id_1": obj1["object_id"],
                            "object_id_2": obj2["object_id"],
                            "frame": frame_num,
                            "confidence": max(0, 1.0 - distance / 150),
                            "timestamp": frame_num / 2.0
                        })

                    # Occlusion detection
                    iou = SimpleObjectTracker._compute_iou(obj1["bbox"], obj2["bbox"])
                    if iou > 0.1:
                        interactions.append({
                            "type": "occlusion",
                            "object_id_1": obj1["object_id"],
                            "object_id_2": obj2["object_id"],
                            "frame": frame_num,
                            "confidence": iou,
                            "timestamp": frame_num / 2.0
                        })

        return interactions

    def _compute_distance(self, bbox1, bbox2):
        """Compute center-to-center distance"""
        c1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        c2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def build_knowledge_graph(
        self,
        objects_by_frame: Dict[int, List[Dict]],
        semantic_labels: Dict[int, any],
        interactions: List[Dict],
        fps: float = 2.0
    ) -> EnhancedKnowledgeGraph:
        """Build enhanced knowledge graph"""
        print(f"  [5/6] Building knowledge graph...")

        graph = EnhancedKnowledgeGraph(video_id=0, fps=fps)

        # Add objects
        for frame_num, frame_objects in objects_by_frame.items():
            for obj in frame_objects:
                obj_id = obj['object_id']
                semantic_label = semantic_labels.get(obj_id)

                if semantic_label:
                    graph.add_object(
                        object_id=obj_id,
                        frame_number=frame_num,
                        timestamp=frame_num / fps,
                        bbox=obj['bbox'],
                        semantic_label=semantic_label.to_dict_with_embedding()
                    )

        # Add interactions
        for interaction in interactions:
            graph.add_interaction(
                object_id_1=interaction['object_id_1'],
                object_id_2=interaction['object_id_2'],
                interaction_type=interaction['type'],
                frame=interaction['frame'],
                timestamp=interaction['timestamp'],
                confidence=interaction['confidence']
            )

        stats = graph.get_statistics()
        print(f"    ✓ Graph: {stats['total_objects']} objects, {stats['total_interactions']} interactions")

        return graph

    def process_clip(self, video_path: Path) -> Dict:
        """Process a single clip through VOKG v2 pipeline"""
        print(f"\n{'='*70}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*70}")

        # Step 1: Frame extraction
        frames = self.extract_frames(video_path, max_frames=30, fps=2.0)

        # Step 2: Object detection + tracking
        print(f"  [2/6] Object detection and tracking...")
        tracker = SimpleObjectTracker()
        all_objects = []
        objects_by_frame = {}

        for frame_idx, frame in enumerate(frames):
            detections = self.segment_objects_sam2(frame)
            tracked = tracker.update(detections)

            for obj in tracked:
                obj["frame"] = frame_idx

            all_objects.extend(tracked)
            objects_by_frame[frame_idx] = tracked

        print(f"    ✓ Tracked {tracker.next_id} unique objects")

        # Step 3: Semantic labeling (if CLIP available)
        semantic_labels = {}
        if self.use_clip and self.semantic_labeler:
            semantic_labels = self.label_objects_semantic(frames, objects_by_frame)
        else:
            print(f"  [3/6] Skipping semantic labeling (CLIP not available)")

        # Step 4: Build embedding index (if available)
        if self.embedding_store and semantic_labels:
            self.build_embedding_index(semantic_labels)
        else:
            print(f"  [4/6] Skipping embedding index (not available)")

        # Step 5: Interaction detection
        print(f"  [5/6] Detecting interactions...")
        interactions = self.detect_interactions(objects_by_frame)
        print(f"    ✓ Found {len(interactions)} interactions")

        # Step 6: Build knowledge graph
        knowledge_graph = self.build_knowledge_graph(
            objects_by_frame,
            semantic_labels,
            interactions,
            fps=2.0
        )

        result = {
            "filename": video_path.name,
            "num_frames": len(frames),
            "knowledge_graph": knowledge_graph,
            "semantic_labels": semantic_labels,
            "statistics": knowledge_graph.get_statistics()
        }

        return result

    def test_queries(self, result: Dict):
        """Test various natural language queries"""
        print(f"\n  [6/6] Testing natural language queries...")

        if not self.use_clip or not self.embedding_store:
            print("    ⚠ Skipping queries (CLIP not available)")
            return

        graph = result['knowledge_graph']

        # Initialize query system
        query_parser = QueryParser(use_llm=self.use_llm)
        query_executor = QueryExecutor(
            embedding_store=self.embedding_store,
            clip_encoder=self.clip_encoder,
            knowledge_graph=graph,
            fps=2.0
        )

        # Test queries
        test_queries = [
            "find all people in the video",
            "find all cars",
            "show interactions involving a person",
        ]

        query_results = []

        for query_text in test_queries:
            print(f"\n    Query: '{query_text}'")

            # Parse query
            structured_query = query_parser.parse(query_text)

            # Execute query
            query_result = query_executor.execute(structured_query)

            print(f"    Result: {query_result.summary}")

            query_results.append({
                'query': query_text,
                'result': query_result.to_dict()
            })

        result['query_results'] = query_results

        # Test reasoning (if available)
        if self.use_llm and self.reasoning_engine:
            print(f"\n    LLM Reasoning...")

            objects = graph.get_all_objects()
            interactions = graph.get_all_interactions()

            # Convert to format expected by reasoning engine
            objects_for_reasoning = []
            for obj in objects:
                obj_copy = obj.copy()
                sem_label = graph.semantic_labels.get(obj['object_id'], {})
                obj_copy['embedding'] = sem_label.get('embedding')
                objects_for_reasoning.append(obj_copy)

            reasoning_result = self.reasoning_engine.analyze_video(
                objects_for_reasoning,
                interactions
            )

            print(f"    Analysis: {reasoning_result.answer[:200]}...")

            result['reasoning_analysis'] = reasoning_result.to_dict()

    def run_tests(self, num_clips: int = None):
        """Run VOKG v2 pipeline on selected clips"""
        print("="*70)
        print("VOKG v2 TESTING ENGINE")
        print("Features: Semantic Labeling + Embedding Search + LLM Reasoning")
        print("="*70)
        print()

        # Select clips
        clips = self.select_clips(num_clips)

        # Process each clip
        for clip in clips:
            result = self.process_clip(clip)

            # Test queries on this clip
            self.test_queries(result)

            self.results.append(result)

        # Generate summary
        self.generate_summary()

        # Save results
        self.save_results()

    def generate_summary(self):
        """Generate comparative summary"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        for i, result in enumerate(self.results, 1):
            stats = result['statistics']
            print(f"\n{i}. {result['filename']}")
            print(f"   Objects: {stats['total_objects']}")
            print(f"   Interactions: {stats['total_interactions']}")
            print(f"   Unique labels: {stats['unique_labels']}")

            if 'label_distribution' in stats:
                top_labels = sorted(
                    stats['label_distribution'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                print(f"   Top objects: {', '.join([f'{label}({count})' for label, count in top_labels])}")

    def save_results(self):
        """Save results to files"""
        output_dir = Path("test_results_v2")
        output_dir.mkdir(exist_ok=True)

        # Save each result
        for result in self.results:
            filename = Path(result['filename']).stem

            # Save knowledge graph
            graph_file = output_dir / f"{filename}_graph.json"
            result['knowledge_graph'].save_to_file(str(graph_file))

            # Save full result
            result_file = output_dir / f"{filename}_result.json"
            result_data = {
                'filename': result['filename'],
                'num_frames': result['num_frames'],
                'statistics': result['statistics'],
                'query_results': result.get('query_results', []),
                'reasoning_analysis': result.get('reasoning_analysis', {})
            }

            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {output_dir}/")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="VOKG v2 Testing Engine")
    parser.add_argument("--clips-dir", default="clips for vdkg",
                       help="Directory containing video clips")
    parser.add_argument("--num-clips", type=int, default=2,
                       help="Number of clips to process")
    parser.add_argument("--no-clip", action="store_true",
                       help="Disable CLIP semantic labeling")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM reasoning")

    args = parser.parse_args()

    # Run tests
    engine = VOKGv2TestEngine(
        args.clips_dir,
        use_clip=not args.no_clip,
        use_llm=not args.no_llm
    )
    engine.run_tests(num_clips=args.num_clips)


if __name__ == "__main__":
    main()
