"""
Object tracking module
Tracks object IDs across frames using IoU matching
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ObjectTracker:
    """
    Simple IoU-based object tracker
    Assigns consistent IDs to objects across frames
    """

    def __init__(self, iou_threshold: float = None):
        """
        Initialize tracker

        Args:
            iou_threshold: IoU threshold for matching (default from settings)
        """
        self.iou_threshold = iou_threshold or settings.TRACKING_IOU_THRESHOLD
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}  # track_id -> track_info
        self.max_age = settings.TRACKING_MAX_AGE
        self.min_hits = settings.TRACKING_MIN_HITS

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections

        Args:
            detections: List of detections with bbox and confidence

        Returns:
            List of detections with assigned track IDs
        """
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]

        if not detections:
            return []

        # Match detections to existing tracks
        if self.tracks:
            matches, unmatched_dets = self._match_detections(detections)
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))

        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id]["bbox"] = detections[det_idx]["bbox"]
            self.tracks[track_id]["age"] = 0
            self.tracks[track_id]["hits"] += 1
            detections[det_idx]["object_id"] = track_id

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "bbox": detections[det_idx]["bbox"],
                "age": 0,
                "hits": 1,
            }
            detections[det_idx]["object_id"] = track_id

        return detections

    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Match detections to existing tracks using IoU

        Args:
            detections: List of detections

        Returns:
            Tuple of (matches, unmatched_detections)
            matches: List of (detection_idx, track_id) tuples
            unmatched_detections: List of detection indices
        """
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        for i, det in enumerate(detections):
            for j, track_bbox in enumerate(track_bboxes):
                iou_matrix[i, j] = compute_iou(det["bbox"], track_bbox)

        # Hungarian algorithm for optimal matching
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)

        # Filter matches by IoU threshold
        matches = []
        unmatched_dets = set(range(len(detections)))

        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                matches.append((det_idx, track_ids[track_idx]))
                unmatched_dets.discard(det_idx)

        return matches, list(unmatched_dets)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def bbox_from_mask(mask: np.ndarray) -> List[float]:
    """
    Extract bounding box from binary mask

    Args:
        mask: Binary mask (H, W)

    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [float(x1), float(y1), float(x2), float(y2)]
