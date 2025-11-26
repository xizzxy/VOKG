"""
Semantic labeling for detected objects
Combines CLIP encoding with hierarchical classification
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image

from .clip_encoder import CLIPEncoder
from .label_hierarchy import LabelHierarchy, COCO_CATEGORIES
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticLabel:
    """
    Semantic label for an object with hierarchical classification
    """
    object_id: int
    primary_label: str
    confidence: float
    category: str
    super_category: str
    alternative_labels: List[Tuple[str, float]]
    embedding: np.ndarray
    frame_number: int
    timestamp: float

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding embedding for serialization)"""
        data = asdict(self)
        data.pop('embedding')  # Too large for JSON
        return data

    def to_dict_with_embedding(self) -> Dict:
        """Convert to dictionary including embedding"""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        return data


class SemanticLabeler:
    """
    Labels objects using CLIP and hierarchical taxonomy
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder = None,
        label_hierarchy: LabelHierarchy = None,
        candidate_labels: List[str] = None,
        top_k_alternatives: int = 5
    ):
        """
        Initialize semantic labeler

        Args:
            clip_encoder: CLIP encoder instance (creates new if None)
            label_hierarchy: Label hierarchy (creates new if None)
            candidate_labels: List of possible labels (uses COCO if None)
            top_k_alternatives: Number of alternative labels to keep
        """
        self.clip_encoder = clip_encoder or CLIPEncoder()
        self.label_hierarchy = label_hierarchy or LabelHierarchy()
        self.candidate_labels = candidate_labels or COCO_CATEGORIES
        self.top_k_alternatives = top_k_alternatives

        logger.info(f"SemanticLabeler initialized with {len(self.candidate_labels)} candidate labels")

    def label_object(
        self,
        image_crop: np.ndarray,
        object_id: int,
        frame_number: int,
        timestamp: float,
        custom_candidates: List[str] = None
    ) -> SemanticLabel:
        """
        Label a single object

        Args:
            image_crop: Cropped region containing the object (RGB numpy array)
            object_id: Unique object ID
            frame_number: Frame number where object appears
            timestamp: Timestamp in seconds
            custom_candidates: Optional custom candidate labels

        Returns:
            SemanticLabel with predictions and embedding
        """
        candidates = custom_candidates or self.candidate_labels

        try:
            # Get predictions
            predictions = self.clip_encoder.predict_labels(
                image_crop,
                candidates,
                top_k=self.top_k_alternatives + 1
            )

            primary_label, confidence = predictions[0]
            alternative_labels = predictions[1:]

            # Get hierarchy
            hierarchy = self.label_hierarchy.get_hierarchy(primary_label)

            # Get embedding
            embedding = self.clip_encoder.encode_image(image_crop)[0]  # (512,)

            return SemanticLabel(
                object_id=object_id,
                primary_label=primary_label,
                confidence=confidence,
                category=hierarchy["category"],
                super_category=hierarchy["super_category"],
                alternative_labels=alternative_labels,
                embedding=embedding,
                frame_number=frame_number,
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"Failed to label object {object_id}: {e}")
            # Return default unknown label
            return SemanticLabel(
                object_id=object_id,
                primary_label="unknown",
                confidence=0.0,
                category="unknown",
                super_category="unknown",
                alternative_labels=[],
                embedding=np.zeros(512, dtype=np.float32),
                frame_number=frame_number,
                timestamp=timestamp
            )

    def label_objects_batch(
        self,
        image_crops: List[np.ndarray],
        object_ids: List[int],
        frame_numbers: List[int],
        timestamps: List[float]
    ) -> List[SemanticLabel]:
        """
        Label multiple objects in batch (more efficient)

        Args:
            image_crops: List of cropped object images
            object_ids: List of object IDs
            frame_numbers: List of frame numbers
            timestamps: List of timestamps

        Returns:
            List of SemanticLabels
        """
        if not image_crops:
            return []

        try:
            # Batch encode all images
            embeddings = self.clip_encoder.encode_image(image_crops)  # (N, 512)

            # Encode all candidate labels once
            text_embeddings = self.clip_encoder.encode_text(self.candidate_labels)  # (M, 512)

            # Compute all similarities at once
            similarities = self.clip_encoder.compute_similarity(embeddings, text_embeddings)  # (N, M)

            # Convert to probabilities
            exp_sim = np.exp(similarities * 100)
            probabilities = exp_sim / exp_sim.sum(axis=1, keepdims=True)

            # Generate labels
            labels = []
            for i in range(len(image_crops)):
                # Get top-k predictions
                top_indices = np.argsort(probabilities[i])[::-1][:self.top_k_alternatives + 1]
                predictions = [
                    (self.candidate_labels[idx], float(probabilities[i, idx]))
                    for idx in top_indices
                ]

                primary_label, confidence = predictions[0]
                alternative_labels = predictions[1:]

                # Get hierarchy
                hierarchy = self.label_hierarchy.get_hierarchy(primary_label)

                label = SemanticLabel(
                    object_id=object_ids[i],
                    primary_label=primary_label,
                    confidence=confidence,
                    category=hierarchy["category"],
                    super_category=hierarchy["super_category"],
                    alternative_labels=alternative_labels,
                    embedding=embeddings[i],
                    frame_number=frame_numbers[i],
                    timestamp=timestamps[i]
                )
                labels.append(label)

            logger.info(f"Batch labeled {len(labels)} objects")
            return labels

        except Exception as e:
            logger.error(f"Batch labeling failed: {e}")
            # Fallback to individual labeling
            return [
                self.label_object(crop, oid, fn, ts)
                for crop, oid, fn, ts in zip(image_crops, object_ids, frame_numbers, timestamps)
            ]

    def refine_label_with_context(
        self,
        label: SemanticLabel,
        context_labels: List[SemanticLabel],
        image_crop: np.ndarray = None
    ) -> SemanticLabel:
        """
        Refine label using context from nearby objects

        Args:
            label: Label to refine
            context_labels: Labels of nearby/related objects
            image_crop: Optional image crop for re-evaluation

        Returns:
            Refined SemanticLabel
        """
        # Get related categories from context
        context_categories = [l.category for l in context_labels]
        context_labels_text = [l.primary_label for l in context_labels]

        # If surrounded by similar objects, boost confidence
        related_count = sum(
            1 for ctx_label in context_labels
            if self.label_hierarchy.are_related(label.primary_label, ctx_label.primary_label)
        )

        if related_count > 0:
            # Context supports this label
            boost_factor = min(0.2, 0.05 * related_count)
            label.confidence = min(1.0, label.confidence + boost_factor)
            logger.debug(
                f"Boosted confidence for {label.primary_label} "
                f"from context (related objects: {related_count})"
            )

        return label