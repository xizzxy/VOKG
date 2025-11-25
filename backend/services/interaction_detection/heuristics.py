"""
Interaction detection heuristics
Spatial, temporal, and causal interaction detection
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def compute_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute distance between bbox centers

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        Euclidean distance
    """
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
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

    if union == 0:
        return 0.0

    return intersection / union


def detect_proximity_interaction(
    obj1: Dict, obj2: Dict, threshold: Optional[float] = None
) -> Optional[Dict]:
    """
    Detect proximity interaction between two objects

    Args:
        obj1: Object 1 data (bbox, frame, etc.)
        obj2: Object 2 data
        threshold: Distance threshold (pixels)

    Returns:
        Interaction dict if detected, None otherwise
    """
    threshold = threshold or settings.INTERACTION_PROXIMITY_THRESHOLD

    distance = compute_distance(obj1["bbox"], obj2["bbox"])

    if distance < threshold:
        return {
            "type": "proximity",
            "confidence": max(0.0, 1.0 - (distance / threshold)),
            "metadata": {"distance": distance},
        }

    return None


def detect_containment_interaction(obj1: Dict, obj2: Dict) -> Optional[Dict]:
    """
    Detect if one object contains another

    Args:
        obj1: Object 1 (potential container)
        obj2: Object 2 (potential contained)

    Returns:
        Interaction dict if detected
    """
    # Check if obj2 is inside obj1
    if (
        obj1["bbox"][0] <= obj2["bbox"][0]
        and obj1["bbox"][1] <= obj2["bbox"][1]
        and obj1["bbox"][2] >= obj2["bbox"][2]
        and obj1["bbox"][3] >= obj2["bbox"][3]
    ):
        # Compute containment ratio
        area1 = (obj1["bbox"][2] - obj1["bbox"][0]) * (obj1["bbox"][3] - obj1["bbox"][1])
        area2 = (obj2["bbox"][2] - obj2["bbox"][0]) * (obj2["bbox"][3] - obj2["bbox"][1])

        return {
            "type": "containment",
            "confidence": 0.9,
            "metadata": {"container_id": obj1["object_id"], "contained_id": obj2["object_id"]},
        }

    return None


def detect_occlusion_interaction(obj1: Dict, obj2: Dict) -> Optional[Dict]:
    """
    Detect occlusion between objects

    Args:
        obj1: Object 1
        obj2: Object 2

    Returns:
        Interaction dict if detected
    """
    iou = compute_iou(obj1["bbox"], obj2["bbox"])

    if iou > settings.INTERACTION_IOU_THRESHOLD:
        return {
            "type": "occlusion",
            "confidence": iou,
            "metadata": {"iou": iou},
        }

    return None


def detect_temporal_interactions(
    object_tracks: Dict[int, List[Dict]], window_size: Optional[int] = None
) -> List[Dict]:
    """
    Detect temporal interactions (chasing, following, etc.)

    Args:
        object_tracks: Dict of object_id -> list of temporal observations
        window_size: Temporal window size (frames)

    Returns:
        List of temporal interactions
    """
    window_size = window_size or settings.INTERACTION_TEMPORAL_WINDOW
    interactions = []

    # Get all object IDs
    object_ids = list(object_tracks.keys())

    # For each pair of objects
    for i, obj_id1 in enumerate(object_ids):
        for obj_id2 in object_ids[i + 1 :]:
            track1 = object_tracks[obj_id1]
            track2 = object_tracks[obj_id2]

            # Find overlapping temporal windows
            frames1 = {obs["frame"]: obs for obs in track1}
            frames2 = {obs["frame"]: obs for obs in track2}
            common_frames = sorted(set(frames1.keys()) & set(frames2.keys()))

            if len(common_frames) < window_size:
                continue

            # Analyze movement patterns
            for start_idx in range(len(common_frames) - window_size + 1):
                window_frames = common_frames[start_idx : start_idx + window_size]

                # Get positions in window
                positions1 = [frames1[f]["bbox"] for f in window_frames]
                positions2 = [frames2[f]["bbox"] for f in window_frames]

                # Detect chasing pattern
                chase_interaction = detect_chase_pattern(
                    positions1, positions2, window_frames, obj_id1, obj_id2
                )
                if chase_interaction:
                    interactions.append(chase_interaction)

                # Detect following pattern
                follow_interaction = detect_follow_pattern(
                    positions1, positions2, window_frames, obj_id1, obj_id2
                )
                if follow_interaction:
                    interactions.append(follow_interaction)

    return interactions


def detect_chase_pattern(
    positions1: List[List[float]],
    positions2: List[List[float]],
    frames: List[int],
    obj_id1: int,
    obj_id2: int,
) -> Optional[Dict]:
    """
    Detect chasing pattern (obj1 chasing obj2)

    Args:
        positions1: Object 1 positions over time
        positions2: Object 2 positions over time
        frames: Frame numbers
        obj_id1: Object 1 ID
        obj_id2: Object 2 ID

    Returns:
        Chase interaction if detected
    """
    # Compute velocities
    velocities1 = compute_velocities(positions1)
    velocities2 = compute_velocities(positions2)

    # Compute distances over time
    distances = [compute_distance(p1, p2) for p1, p2 in zip(positions1, positions2)]

    # Check if distance is decreasing and obj1 is moving toward obj2
    distance_decreasing = all(
        distances[i + 1] <= distances[i] for i in range(len(distances) - 1)
    )

    if distance_decreasing and np.mean(velocities1) > settings.INTERACTION_VELOCITY_THRESHOLD:
        return {
            "type": "chase",
            "object_id_1": obj_id1,
            "object_id_2": obj_id2,
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "confidence": 0.8,
            "metadata": {
                "avg_velocity_1": np.mean(velocities1),
                "distance_change": distances[0] - distances[-1],
            },
        }

    return None


def detect_follow_pattern(
    positions1: List[List[float]],
    positions2: List[List[float]],
    frames: List[int],
    obj_id1: int,
    obj_id2: int,
) -> Optional[Dict]:
    """
    Detect following pattern (obj1 following obj2 at constant distance)

    Args:
        positions1: Object 1 positions
        positions2: Object 2 positions
        frames: Frame numbers
        obj_id1: Object 1 ID
        obj_id2: Object 2 ID

    Returns:
        Follow interaction if detected
    """
    # Compute distances over time
    distances = [compute_distance(p1, p2) for p1, p2 in zip(positions1, positions2)]

    # Check if distance is relatively constant
    distance_std = np.std(distances)
    distance_mean = np.mean(distances)

    if distance_std < 0.2 * distance_mean and distance_mean < settings.INTERACTION_PROXIMITY_THRESHOLD * 2:
        return {
            "type": "follow",
            "object_id_1": obj_id1,
            "object_id_2": obj_id2,
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "confidence": 0.75,
            "metadata": {
                "avg_distance": distance_mean,
                "distance_std": distance_std,
            },
        }

    return None


def compute_velocities(positions: List[List[float]]) -> List[float]:
    """
    Compute velocities from positions

    Args:
        positions: List of bboxes over time

    Returns:
        List of velocities (magnitude)
    """
    velocities = []
    for i in range(len(positions) - 1):
        center1 = [(positions[i][0] + positions[i][2]) / 2, (positions[i][1] + positions[i][3]) / 2]
        center2 = [(positions[i + 1][0] + positions[i + 1][2]) / 2, (positions[i + 1][1] + positions[i + 1][3]) / 2]

        velocity = np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)
        velocities.append(velocity)

    return velocities


def detect_causal_interactions(
    events: List[Dict], velocity_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Detect causal interactions (one event causes another)

    Args:
        events: List of detected events
        velocity_threshold: Velocity change threshold

    Returns:
        List of causal interactions
    """
    velocity_threshold = velocity_threshold or settings.INTERACTION_VELOCITY_THRESHOLD
    causal_interactions = []

    # Simple heuristic: if object A's velocity changes suddenly near object B,
    # there may be a causal relationship

    # This is a placeholder for more sophisticated causal inference
    # Would use techniques like Granger causality in production

    return causal_interactions
