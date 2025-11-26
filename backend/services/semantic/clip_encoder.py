"""
CLIP model wrapper for vision-language encoding
Provides efficient batched encoding for images and text
"""

import torch
import numpy as np
from typing import List, Union
from PIL import Image
import clip

from backend.core.logging import get_logger

logger = get_logger(__name__)


class CLIPEncoder:
    """
    Wrapper for OpenAI CLIP model
    Handles encoding of images and text into 512-dim embeddings
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP encoder

        Args:
            model_name: CLIP model variant ("ViT-B/32", "ViT-B/16", "ViT-L/14")
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        self.embedding_dim = 512  # CLIP ViT-B/32 embedding dimension

    @torch.no_grad()
    def encode_image(
        self,
        images: Union[np.ndarray, Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """
        Encode image(s) into CLIP embeddings

        Args:
            images: Single image (PIL or numpy) or list of PIL images

        Returns:
            np.ndarray of shape (N, 512) where N is number of images
        """
        # Convert to list if single image
        if isinstance(images, (np.ndarray, Image.Image)):
            images = [images]

        # Convert numpy to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert from RGB numpy array to PIL
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                pil_images.append(img)

        # Preprocess and stack
        try:
            image_tensors = torch.stack([
                self.preprocess(img) for img in pil_images
            ]).to(self.device)

            # Encode
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embeddings = image_features.cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into CLIP embeddings

        Args:
            texts: Single string or list of strings

        Returns:
            np.ndarray of shape (N, 512) where N is number of texts
        """
        # Convert to list if single string
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Tokenize
            text_tokens = clip.tokenize(texts).to(self.device)

            # Encode
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embeddings = text_features.cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise

    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings

        Args:
            image_embeddings: (M, 512) array of image embeddings
            text_embeddings: (N, 512) array of text embeddings

        Returns:
            (M, N) array of similarity scores
        """
        # Ensure 2D arrays
        if image_embeddings.ndim == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.reshape(1, -1)

        # Cosine similarity (embeddings are already normalized)
        similarity = np.matmul(image_embeddings, text_embeddings.T)

        return similarity

    def predict_labels(
        self,
        image: Union[np.ndarray, Image.Image],
        candidate_labels: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Predict labels for an image using zero-shot classification

        Args:
            image: Input image
            candidate_labels: List of possible labels
            top_k: Number of top predictions to return

        Returns:
            List of (label, confidence) tuples, sorted by confidence
        """
        # Encode image
        image_embedding = self.encode_image(image)  # (1, 512)

        # Encode labels
        text_embeddings = self.encode_text(candidate_labels)  # (N, 512)

        # Compute similarities
        similarities = self.compute_similarity(image_embedding, text_embeddings)[0]

        # Convert to probabilities using softmax
        exp_sim = np.exp(similarities * 100)  # Temperature scaling
        probabilities = exp_sim / exp_sim.sum()

        # Get top-k
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = [
            (candidate_labels[i], float(probabilities[i]))
            for i in top_indices
        ]

        return predictions