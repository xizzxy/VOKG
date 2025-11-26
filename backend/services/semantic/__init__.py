"""
Semantic labeling module for VOKG
Provides CLIP-based object labeling and hierarchical classification
"""

from .clip_encoder import CLIPEncoder
from .label_hierarchy import LabelHierarchy, COCO_CATEGORIES
from .semantic_labeler import SemanticLabeler, SemanticLabel

__all__ = [
    'CLIPEncoder',
    'LabelHierarchy',
    'COCO_CATEGORIES',
    'SemanticLabeler',
    'SemanticLabel'
]