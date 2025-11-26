"""
FAISS-based embedding storage and similarity search
Efficient vector similarity search for semantic queries
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result from embedding similarity search"""
    object_id: int
    label: str
    confidence: float
    category: str
    frame_number: int
    timestamp: float
    similarity_score: float
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'object_id': self.object_id,
            'label': self.label,
            'confidence': self.confidence,
            'category': self.category,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'similarity_score': self.similarity_score
        }


class EmbeddingStore:
    """
    Vector database for semantic embeddings
    Uses FAISS for fast similarity search
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize embedding store

        Args:
            embedding_dim: Dimension of embeddings (512 for CLIP)
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2")
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric

        # Initialize FAISS index
        self.index = self._create_index()

        # Metadata storage (FAISS only stores vectors, not metadata)
        self.id_to_metadata: Dict[int, Dict] = {}
        self.next_id = 0

        logger.info(f"EmbeddingStore initialized with {index_type} index, {metric} metric")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.metric == "cosine":
            # For cosine similarity, normalize vectors and use inner product
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "ivf":
                # IVF with 100 clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        else:  # L2
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100, faiss.METRIC_L2)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_L2)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[int]:
        """
        Add embeddings to the index

        Args:
            embeddings: (N, embedding_dim) array of embeddings
            metadata: List of N metadata dictionaries

        Returns:
            List of assigned IDs
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        # Train index if needed (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings.astype(np.float32))

        # Add to index
        ids = list(range(self.next_id, self.next_id + len(embeddings)))
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        for i, meta in enumerate(metadata):
            self.id_to_metadata[ids[i]] = meta

        self.next_id += len(embeddings)

        logger.info(f"Added {len(embeddings)} embeddings to index (total: {self.index.ntotal})")
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[SearchResult]:
        """
        Search for similar embeddings

        Args:
            query_embedding: (embedding_dim,) query vector
            k: Number of results to return
            filter_fn: Optional function to filter results (takes metadata, returns bool)

        Returns:
            List of SearchResults sorted by similarity
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / (norm + 1e-8)

        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), min(k * 2, self.index.ntotal))

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata = self.id_to_metadata.get(idx)
            if metadata is None:
                continue

            # Apply filter
            if filter_fn and not filter_fn(metadata):
                continue

            # Convert distance to similarity score
            if self.metric == "cosine":
                similarity = float(dist)  # Inner product (already similarity)
            else:
                similarity = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity

            result = SearchResult(
                object_id=metadata['object_id'],
                label=metadata['label'],
                confidence=metadata['confidence'],
                category=metadata.get('category', 'unknown'),
                frame_number=metadata['frame_number'],
                timestamp=metadata['timestamp'],
                similarity_score=similarity,
                embedding=metadata.get('embedding')
            )
            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_by_text(
        self,
        text_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search using text embedding (from CLIP text encoder)

        Args:
            text_embedding: (embedding_dim,) text embedding
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResults
        """
        results = self.search(text_embedding, k=k)

        # Filter by minimum similarity
        results = [r for r in results if r.similarity_score >= min_similarity]

        return results

    def search_by_label(
        self,
        label: str,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search for objects with specific label

        Args:
            label: Label to search for (e.g., "person", "cat")
            k: Number of results

        Returns:
            List of SearchResults
        """
        # Filter function
        def label_filter(meta):
            return meta['label'].lower() == label.lower()

        # Use a generic query (center of embedding space) and filter
        query = np.zeros(self.embedding_dim, dtype=np.float32)
        all_results = self.search(query, k=self.index.ntotal)

        # Filter and limit
        filtered = [r for r in all_results if label_filter(self.id_to_metadata.get(r.object_id, {}))]
        return filtered[:k]

    def get_metadata(self, idx: int) -> Optional[Dict]:
        """Get metadata for a specific index"""
        return self.id_to_metadata.get(idx)

    def save(self, directory: Path):
        """
        Save index and metadata to disk

        Args:
            directory: Directory to save to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = directory / "embeddings.index"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = directory / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_to_metadata': self.id_to_metadata,
                'next_id': self.next_id,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)

        logger.info(f"Saved embedding store to {directory}")

    def load(self, directory: Path):
        """
        Load index and metadata from disk

        Args:
            directory: Directory to load from
        """
        directory = Path(directory)

        # Load FAISS index
        index_path = directory / "embeddings.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = directory / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.id_to_metadata = data['id_to_metadata']
            self.next_id = data['next_id']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
            self.metric = data['metric']

        logger.info(f"Loaded embedding store from {directory} ({self.index.ntotal} embeddings)")

    def __len__(self) -> int:
        """Get number of embeddings in index"""
        return self.index.ntotal