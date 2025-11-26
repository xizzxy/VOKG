"""
Label taxonomy and hierarchical classification
Defines category hierarchies for semantic labels
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


# COCO dataset categories (80 classes)
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# Hierarchical taxonomy: label → category → super-category
LABEL_TAXONOMY = {
    # Living things
    "person": {"category": "human", "super_category": "living thing"},
    "bird": {"category": "animal", "super_category": "living thing"},
    "cat": {"category": "animal", "super_category": "living thing"},
    "dog": {"category": "animal", "super_category": "living thing"},
    "horse": {"category": "animal", "super_category": "living thing"},
    "sheep": {"category": "animal", "super_category": "living thing"},
    "cow": {"category": "animal", "super_category": "living thing"},
    "elephant": {"category": "animal", "super_category": "living thing"},
    "bear": {"category": "animal", "super_category": "living thing"},
    "zebra": {"category": "animal", "super_category": "living thing"},
    "giraffe": {"category": "animal", "super_category": "living thing"},

    # Vehicles
    "bicycle": {"category": "vehicle", "super_category": "transportation"},
    "car": {"category": "vehicle", "super_category": "transportation"},
    "motorcycle": {"category": "vehicle", "super_category": "transportation"},
    "airplane": {"category": "vehicle", "super_category": "transportation"},
    "bus": {"category": "vehicle", "super_category": "transportation"},
    "train": {"category": "vehicle", "super_category": "transportation"},
    "truck": {"category": "vehicle", "super_category": "transportation"},
    "boat": {"category": "vehicle", "super_category": "transportation"},

    # Outdoor objects
    "traffic light": {"category": "traffic", "super_category": "outdoor"},
    "fire hydrant": {"category": "street furniture", "super_category": "outdoor"},
    "stop sign": {"category": "traffic", "super_category": "outdoor"},
    "parking meter": {"category": "street furniture", "super_category": "outdoor"},
    "bench": {"category": "furniture", "super_category": "outdoor"},

    # Accessories
    "backpack": {"category": "accessory", "super_category": "object"},
    "umbrella": {"category": "accessory", "super_category": "object"},
    "handbag": {"category": "accessory", "super_category": "object"},
    "tie": {"category": "accessory", "super_category": "object"},
    "suitcase": {"category": "accessory", "super_category": "object"},

    # Sports equipment
    "frisbee": {"category": "sports equipment", "super_category": "object"},
    "skis": {"category": "sports equipment", "super_category": "object"},
    "snowboard": {"category": "sports equipment", "super_category": "object"},
    "sports ball": {"category": "sports equipment", "super_category": "object"},
    "kite": {"category": "sports equipment", "super_category": "object"},
    "baseball bat": {"category": "sports equipment", "super_category": "object"},
    "baseball glove": {"category": "sports equipment", "super_category": "object"},
    "skateboard": {"category": "sports equipment", "super_category": "object"},
    "surfboard": {"category": "sports equipment", "super_category": "object"},
    "tennis racket": {"category": "sports equipment", "super_category": "object"},

    # Kitchen items
    "bottle": {"category": "container", "super_category": "object"},
    "wine glass": {"category": "container", "super_category": "object"},
    "cup": {"category": "container", "super_category": "object"},
    "fork": {"category": "utensil", "super_category": "object"},
    "knife": {"category": "utensil", "super_category": "object"},
    "spoon": {"category": "utensil", "super_category": "object"},
    "bowl": {"category": "container", "super_category": "object"},

    # Food
    "banana": {"category": "food", "super_category": "object"},
    "apple": {"category": "food", "super_category": "object"},
    "sandwich": {"category": "food", "super_category": "object"},
    "orange": {"category": "food", "super_category": "object"},
    "broccoli": {"category": "food", "super_category": "object"},
    "carrot": {"category": "food", "super_category": "object"},
    "hot dog": {"category": "food", "super_category": "object"},
    "pizza": {"category": "food", "super_category": "object"},
    "donut": {"category": "food", "super_category": "object"},
    "cake": {"category": "food", "super_category": "object"},

    # Furniture
    "chair": {"category": "furniture", "super_category": "indoor"},
    "couch": {"category": "furniture", "super_category": "indoor"},
    "potted plant": {"category": "furniture", "super_category": "indoor"},
    "bed": {"category": "furniture", "super_category": "indoor"},
    "dining table": {"category": "furniture", "super_category": "indoor"},
    "toilet": {"category": "appliance", "super_category": "indoor"},

    # Electronics
    "tv": {"category": "electronics", "super_category": "object"},
    "laptop": {"category": "electronics", "super_category": "object"},
    "mouse": {"category": "electronics", "super_category": "object"},
    "remote": {"category": "electronics", "super_category": "object"},
    "keyboard": {"category": "electronics", "super_category": "object"},
    "cell phone": {"category": "electronics", "super_category": "object"},

    # Appliances
    "microwave": {"category": "appliance", "super_category": "object"},
    "oven": {"category": "appliance", "super_category": "object"},
    "toaster": {"category": "appliance", "super_category": "object"},
    "sink": {"category": "appliance", "super_category": "object"},
    "refrigerator": {"category": "appliance", "super_category": "object"},

    # Other objects
    "book": {"category": "item", "super_category": "object"},
    "clock": {"category": "item", "super_category": "object"},
    "vase": {"category": "item", "super_category": "object"},
    "scissors": {"category": "tool", "super_category": "object"},
    "teddy bear": {"category": "toy", "super_category": "object"},
    "hair drier": {"category": "appliance", "super_category": "object"},
    "toothbrush": {"category": "item", "super_category": "object"},
}


class LabelHierarchy:
    """
    Manages hierarchical label taxonomy
    """

    def __init__(self, taxonomy: Dict = None):
        """
        Initialize label hierarchy

        Args:
            taxonomy: Custom taxonomy dict (uses LABEL_TAXONOMY if None)
        """
        self.taxonomy = taxonomy or LABEL_TAXONOMY

    def get_category(self, label: str) -> Optional[str]:
        """Get category for a label"""
        if label in self.taxonomy:
            return self.taxonomy[label]["category"]
        return "unknown"

    def get_super_category(self, label: str) -> Optional[str]:
        """Get super-category for a label"""
        if label in self.taxonomy:
            return self.taxonomy[label]["super_category"]
        return "unknown"

    def get_hierarchy(self, label: str) -> Dict[str, str]:
        """
        Get full hierarchy for a label

        Returns:
            Dict with label, category, super_category
        """
        if label in self.taxonomy:
            return {
                "label": label,
                "category": self.taxonomy[label]["category"],
                "super_category": self.taxonomy[label]["super_category"]
            }
        return {
            "label": label,
            "category": "unknown",
            "super_category": "unknown"
        }

    def get_labels_by_category(self, category: str) -> List[str]:
        """Get all labels in a category"""
        return [
            label for label, info in self.taxonomy.items()
            if info["category"] == category
        ]

    def get_labels_by_super_category(self, super_category: str) -> List[str]:
        """Get all labels in a super-category"""
        return [
            label for label, info in self.taxonomy.items()
            if info["super_category"] == super_category
        ]

    def are_related(self, label1: str, label2: str) -> bool:
        """Check if two labels share category or super-category"""
        if label1 not in self.taxonomy or label2 not in self.taxonomy:
            return False

        info1 = self.taxonomy[label1]
        info2 = self.taxonomy[label2]

        return (
            info1["category"] == info2["category"] or
            info1["super_category"] == info2["super_category"]
        )