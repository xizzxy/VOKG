"""
VOKG Pipeline Testing Engine
Standalone script to test video clips through the complete VOKG pipeline
"""

import os
import sys
import json
import random
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


class VOKGTestEngine:
    """Standalone VOKG pipeline testing engine"""

    def __init__(self, clips_dir: str):
        self.clips_dir = Path(clips_dir)
        self.results = []

    def select_clips(self, num_clips: int = None) -> List[Path]:
        """Select diverse clips for testing"""
        all_clips = list(self.clips_dir.glob("*.mp4"))

        if not all_clips:
            raise ValueError(f"No MP4 files found in {self.clips_dir}")

        # Select 3-5 clips based on variety
        if num_clips is None:
            num_clips = min(4, len(all_clips))  # Choose 4 for good variety

        selected = random.sample(all_clips, min(num_clips, len(all_clips)))

        print(f"Selected {len(selected)} clips from {len(all_clips)} available:")
        for clip in selected:
            print(f"  - {clip.name}")

        return selected

    def extract_frames(self, video_path: Path, max_frames: int = 30, fps: float = 2.0) -> List[np.ndarray]:
        """Extract frames from video using OpenCV"""
        print(f"  [1/5] Extracting frames from {video_path.name}...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame interval
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

        print(f"    Extracted {len(frames)} frames from {total_frames} total")
        return frames

    def segment_objects_sam2(self, frame: np.ndarray) -> List[Dict]:
        """
        Simulate SAM 2 segmentation
        In production, this would use actual SAM 2 model
        For testing, we use simplified object detection
        """
        # Use OpenCV for basic object detection (simulating SAM 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.001  # 0.1% of frame
        max_area = (h * w) * 0.8    # 80% of frame

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            detection = {
                "bbox": [x, y, x + bw, y + bh],
                "mask": mask,
                "area": float(area),
                "confidence": min(1.0, area / (bw * bh))  # Simplified confidence
            }
            detections.append(detection)

        # Limit to top detections by area
        detections = sorted(detections, key=lambda x: x["area"], reverse=True)[:20]

        return detections

    def detect_interactions(self, objects_by_frame: Dict[int, List[Dict]]) -> List[Dict]:
        """Detect spatial and temporal interactions"""
        interactions = []

        # Spatial interactions (per frame)
        for frame_num, frame_objects in objects_by_frame.items():
            for i, obj1 in enumerate(frame_objects):
                for obj2 in frame_objects[i+1:]:
                    # Proximity detection
                    distance = self._compute_distance(obj1["bbox"], obj2["bbox"])
                    if distance < 150:  # pixels
                        interactions.append({
                            "type": "proximity",
                            "object_id_1": obj1["object_id"],
                            "object_id_2": obj2["object_id"],
                            "frame": frame_num,
                            "confidence": max(0, 1.0 - distance / 150),
                            "timestamp": frame_num / 2.0  # Assuming 2 fps
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

        # Temporal interactions (across frames)
        temporal = self._detect_temporal_interactions(objects_by_frame)
        interactions.extend(temporal)

        return interactions

    def _compute_distance(self, bbox1, bbox2):
        """Compute center-to-center distance"""
        c1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        c2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _detect_temporal_interactions(self, objects_by_frame: Dict) -> List[Dict]:
        """Detect temporal patterns like following, chasing"""
        interactions = []

        # Group by object ID
        objects_by_id = defaultdict(list)
        for frame_num, frame_objects in objects_by_frame.items():
            for obj in frame_objects:
                objects_by_id[obj["object_id"]].append({
                    "frame": frame_num,
                    "bbox": obj["bbox"]
                })

        # Analyze pairs
        object_ids = list(objects_by_id.keys())
        for i, oid1 in enumerate(object_ids):
            for oid2 in object_ids[i+1:]:
                track1 = objects_by_id[oid1]
                track2 = objects_by_id[oid2]

                # Find common frames
                frames1 = {t["frame"]: t for t in track1}
                frames2 = {t["frame"]: t for t in track2}
                common = sorted(set(frames1.keys()) & set(frames2.keys()))

                if len(common) < 5:
                    continue

                # Check for following pattern (constant distance)
                distances = [
                    self._compute_distance(frames1[f]["bbox"], frames2[f]["bbox"])
                    for f in common
                ]

                if len(distances) >= 5:
                    dist_std = np.std(distances)
                    dist_mean = np.mean(distances)

                    if dist_std < 0.2 * dist_mean and dist_mean < 300:
                        interactions.append({
                            "type": "following",
                            "object_id_1": oid1,
                            "object_id_2": oid2,
                            "frame": common[0],
                            "duration_frames": len(common),
                            "confidence": 0.7,
                            "timestamp": common[0] / 2.0
                        })

        return interactions

    def build_knowledge_graph(self, objects: List[Dict], interactions: List[Dict]) -> Dict:
        """Build knowledge graph structure"""
        # Create nodes (unique objects)
        unique_objects = {}
        for obj in objects:
            oid = obj["object_id"]
            if oid not in unique_objects:
                unique_objects[oid] = {
                    "id": f"obj_{oid}",
                    "type": "object",
                    "label": f"object_{oid}",
                    "properties": {
                        "first_appearance": obj["frame"],
                        "appearances": 1,
                        "avg_area": obj["area"]
                    }
                }
            else:
                unique_objects[oid]["properties"]["appearances"] += 1

        nodes = list(unique_objects.values())

        # Create edges (interactions)
        edges = []
        for interaction in interactions:
            edges.append({
                "source": f"obj_{interaction['object_id_1']}",
                "target": f"obj_{interaction['object_id_2']}",
                "type": interaction["type"],
                "properties": {
                    "frame": interaction.get("frame"),
                    "timestamp": interaction.get("timestamp"),
                    "confidence": interaction["confidence"]
                }
            })

        return {
            "nodes": nodes,
            "edges": edges
        }

    def generate_narrative(self, objects: List[Dict], interactions: List[Dict],
                          knowledge_graph: Dict, video_name: str) -> str:
        """Generate natural language explanation"""
        num_objects = len(knowledge_graph["nodes"])
        num_interactions = len(interactions)

        # Count interaction types
        interaction_types = defaultdict(int)
        for interaction in interactions:
            interaction_types[interaction["type"]] += 1

        narrative = f"Scene Analysis for '{video_name}':\n\n"

        if num_objects == 0:
            narrative += "LOW-INTERACTION FOOTAGE: No significant objects detected in this clip. "
            narrative += "The scene may be too dark, uniform, or lack distinct visual elements.\n"
            return narrative

        narrative += f"The scene contains {num_objects} distinct object(s) tracked across frames.\n\n"

        if num_interactions == 0:
            narrative += "LOW-INTERACTION FOOTAGE: Objects are present but show minimal interaction. "
            narrative += "They appear static or move independently without meaningful relationships.\n"
        else:
            narrative += f"Detected {num_interactions} interaction(s):\n"
            for itype, count in interaction_types.items():
                narrative += f"  - {count} {itype} interaction(s)\n"

            narrative += "\nSpatial relationships suggest "
            if "proximity" in interaction_types:
                narrative += "objects are frequently near each other, indicating possible shared context or activity. "
            if "occlusion" in interaction_types:
                narrative += "Some objects overlap, suggesting layered depth or one object passing in front of another. "
            if "following" in interaction_types:
                narrative += "Temporal analysis shows consistent movement patterns, indicating coordinated motion. "

            narrative += "\n\nThe objects maintain visual presence across multiple frames, "
            narrative += "demonstrating temporal consistency in the scene."

        return narrative

    def process_clip(self, video_path: Path) -> Dict:
        """Process a single clip through the full VOKG pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*60}")

        # Step 1: Frame extraction
        frames = self.extract_frames(video_path, max_frames=30, fps=2.0)

        # Step 2: SAM 2 segmentation + object tracking
        print(f"  [2/5] Running SAM 2 segmentation and object tracking...")
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

        print(f"    Detected {tracker.next_id} unique objects across {len(frames)} frames")

        # Step 3: Interaction detection
        print(f"  [3/5] Detecting interactions...")
        interactions = self.detect_interactions(objects_by_frame)
        print(f"    Found {len(interactions)} interactions")

        # Step 4: Knowledge graph generation
        print(f"  [4/5] Building knowledge graph...")
        knowledge_graph = self.build_knowledge_graph(all_objects, interactions)
        print(f"    Graph: {len(knowledge_graph['nodes'])} nodes, {len(knowledge_graph['edges'])} edges")

        # Step 5: Generate narrative
        print(f"  [5/5] Generating narrative explanation...")
        narrative = self.generate_narrative(all_objects, interactions, knowledge_graph, video_path.name)

        # Compile results
        result = {
            "filename": video_path.name,
            "num_frames": len(frames),
            "detected_objects": [
                {"object_id": node["id"], "label": node["label"],
                 "appearances": node["properties"]["appearances"]}
                for node in knowledge_graph["nodes"]
            ],
            "interactions": [
                {"type": interaction["type"],
                 "objects": (interaction["object_id_1"], interaction["object_id_2"]),
                 "timestamp": interaction.get("timestamp", "unknown"),
                 "confidence": interaction["confidence"]}
                for interaction in interactions
            ],
            "knowledge_graph": knowledge_graph,
            "narrative": narrative,
            "statistics": {
                "total_objects": len(knowledge_graph["nodes"]),
                "total_interactions": len(interactions),
                "interaction_density": len(interactions) / max(1, len(knowledge_graph["nodes"])),
                "is_low_interaction": len(interactions) < 5 or len(knowledge_graph["nodes"]) < 2
            }
        }

        return result

    def run_tests(self, num_clips: int = None):
        """Run VOKG pipeline on selected clips"""
        print("="*60)
        print("VOKG TESTING ENGINE")
        print("="*60)

        # Select clips
        clips = self.select_clips(num_clips)

        # Process each clip
        for clip in clips:
            result = self.process_clip(clip)
            self.results.append(result)

        # Generate comparative summary
        self.generate_summary()

        # Save results
        self.save_results()

    def generate_summary(self):
        """Generate comparative summary of all clips"""
        print("\n" + "="*60)
        print("COMPARATIVE SUMMARY")
        print("="*60)

        if not self.results:
            print("No results to summarize.")
            return

        # Find clips with most/least interactions
        sorted_by_interactions = sorted(self.results,
                                       key=lambda x: x["statistics"]["total_interactions"],
                                       reverse=True)

        most_interactions = sorted_by_interactions[0]
        least_interactions = sorted_by_interactions[-1]

        # Find clip with clearest relationships
        sorted_by_density = sorted(self.results,
                                  key=lambda x: x["statistics"]["interaction_density"],
                                  reverse=True)
        clearest = sorted_by_density[0]

        # Determine easiest/hardest
        easiest = min(self.results, key=lambda x: (1 if x["statistics"]["is_low_interaction"] else 0,
                                                   -x["statistics"]["total_objects"]))
        hardest = max(self.results, key=lambda x: (1 if x["statistics"]["is_low_interaction"] else 0,
                                                   x["statistics"]["total_objects"]))

        print(f"\n1. MOST INTERACTIONS:")
        print(f"   {most_interactions['filename']}")
        print(f"   {most_interactions['statistics']['total_interactions']} interactions detected")

        print(f"\n2. CLEAREST OBJECT RELATIONSHIPS:")
        print(f"   {clearest['filename']}")
        print(f"   Interaction density: {clearest['statistics']['interaction_density']:.2f}")

        print(f"\n3. EASIEST FOR SYSTEM:")
        print(f"   {easiest['filename']}")
        print(f"   Well-defined objects and clear interactions")

        print(f"\n4. HARDEST FOR SYSTEM:")
        print(f"   {hardest['filename']}")
        if hardest["statistics"]["is_low_interaction"]:
            print(f"   Low-interaction footage with minimal detectable structure")
        else:
            print(f"   Complex scene with {hardest['statistics']['total_objects']} objects")

        print(f"\n5. SUGGESTED IMPROVEMENTS FOR NEXT VERSION:")
        print(f"   • Integrate actual SAM 2 model for better segmentation quality")
        print(f"   • Add CLIP-based semantic labeling for object recognition")
        print(f"   • Implement more sophisticated temporal reasoning (LSTM/Transformer)")
        print(f"   • Add causal inference for action-reaction detection")
        print(f"   • Improve low-light and motion blur handling")
        print(f"   • Add multi-object tracking with appearance features")
        print(f"   • Implement graph neural networks for relationship learning")

    def save_results(self):
        """Save results to JSON file"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "vokg_test_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        # Also save individual reports
        for result in self.results:
            report_file = output_dir / f"{Path(result['filename']).stem}_report.txt"
            with open(report_file, 'w') as f:
                f.write(f"VOKG Analysis Report\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"File: {result['filename']}\n\n")
                f.write(f"Detected Objects ({len(result['detected_objects'])}):\n")
                for obj in result['detected_objects']:
                    f.write(f"  - {obj['label']} (appeared {obj['appearances']} times)\n")
                f.write(f"\nInteractions ({len(result['interactions'])}):\n")
                for interaction in result['interactions']:
                    f.write(f"  - {interaction['type']} between obj_{interaction['objects'][0]} "
                           f"and obj_{interaction['objects'][1]} at t={interaction['timestamp']:.1f}s "
                           f"(confidence: {interaction['confidence']:.2f})\n")
                f.write(f"\nKnowledge Graph:\n")
                f.write(f"  Nodes: {len(result['knowledge_graph']['nodes'])}\n")
                f.write(f"  Edges: {len(result['knowledge_graph']['edges'])}\n\n")
                f.write(f"Narrative:\n{result['narrative']}\n")

            print(f"  Individual report: {report_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="VOKG Pipeline Testing Engine")
    parser.add_argument("--clips-dir", default="clips for vdkg",
                       help="Directory containing video clips")
    parser.add_argument("--num-clips", type=int, default=None,
                       help="Number of clips to process (default: 3-5)")

    args = parser.parse_args()

    # Run tests
    engine = VOKGTestEngine(args.clips_dir)
    engine.run_tests(num_clips=args.num_clips)


if __name__ == "__main__":
    main()