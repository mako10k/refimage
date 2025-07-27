"""
Vector search engine using FAISS for efficient similarity search.

This module provides vector indexing and search capabilities
for image embeddings using Facebook's FAISS library.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from ..config import Settings
from ..models.schemas import ImageEmbedding

logger = logging.getLogger(__name__)


class VectorSearchError(Exception):
    """Vector search related errors."""


class VectorSearchEngine:
    """
    FAISS-based vector search engine for image embeddings.

    Provides efficient k-nearest neighbor search on high-dimensional
    embedding vectors with persistence capabilities.
    """

    def __init__(self, settings: Settings):
        """
        Initialize vector search engine.

        Args:
            settings: Application settings

        Raises:
            VectorSearchError: If FAISS is not available or
                              initialization fails
        """
        assert settings is not None, "Settings object is required"

        if faiss is None:
            error_msg = (
                "FAISS library not available. " "Install with: pip install faiss-cpu"
            )
            raise VectorSearchError(error_msg)

        self.settings = settings
        self.index = None
        self.embedding_ids = []  # Track embedding IDs in index order
        self.embedding_dimension = None
        self._setup_index()

    def _setup_index(self) -> None:
        """Setup FAISS index based on configuration."""
        # Will be initialized when first embedding is added
        self.index = None
        self.embedding_ids = []
        self.embedding_dimension = None
        logger.info("Vector search engine initialized")

    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index for given dimension.

        Args:
            dimension: Embedding vector dimension

        Returns:
            Configured FAISS index
        """
        assert dimension > 0, f"Invalid dimension: {dimension}"

        try:
            # Use IndexFlatIP (Inner Product) for cosine similarity
            # Embeddings should be normalized before adding
            index = faiss.IndexFlatIP(dimension)

            # For large datasets, consider using IndexIVFFlat
            # quantizer = faiss.IndexFlatIP(dimension)
            # index = faiss.IndexIVFFlat(quantizer, dimension, 100)

            logger.info(f"Created FAISS index with dimension {dimension}")
            return index

        except Exception as e:
            error_msg = f"Failed to create FAISS index: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def add_embedding(self, embedding: ImageEmbedding) -> None:
        """
        Add embedding to the search index.

        Args:
            embedding: Image embedding to add

        Raises:
            VectorSearchError: If adding embedding fails
        """
        assert embedding is not None, "Embedding object is required"
        assert embedding.embedding is not None, "Embedding vector is required"
        assert len(embedding.embedding) > 0, "Embedding vector cannot be empty"

        try:
            embedding_vector = np.array(embedding.embedding, dtype=np.float32)

            # Initialize index on first embedding
            if self.index is None:
                self.embedding_dimension = len(embedding_vector)
                self.index = self._create_index(self.embedding_dimension)
            else:
                # Validate dimension consistency
                expected_dim = self.embedding_dimension
                actual_dim = len(embedding_vector)
                assert actual_dim == expected_dim, (
                    f"Embedding dimension mismatch: "
                    f"expected {expected_dim}, got {actual_dim}"
                )

            # Normalize vector for cosine similarity
            norm = np.linalg.norm(embedding_vector)
            if norm > 0:
                embedding_vector = embedding_vector / norm

            # Add to index
            embedding_vector = embedding_vector.reshape(1, -1)
            self.index.add(embedding_vector)
            self.embedding_ids.append(str(embedding.image_id))

            logger.debug(f"Added embedding for image {embedding.image_id}")

        except Exception as e:
            error_msg = f"Failed to add embedding: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def add_embeddings_batch(self, embeddings: List[ImageEmbedding]) -> None:
        """
        Add multiple embeddings in batch for efficiency.

        Args:
            embeddings: List of embeddings to add

        Raises:
            VectorSearchError: If batch addition fails
        """
        assert embeddings is not None, "Embeddings list is required"
        assert len(embeddings) > 0, "Embeddings list cannot be empty"

        try:
            # Convert to numpy arrays
            vectors = []
            ids = []

            for embedding in embeddings:
                assert embedding.embedding is not None, "Embedding vector required"
                vector = np.array(embedding.embedding, dtype=np.float32)

                # Initialize index on first embedding
                if self.index is None:
                    self.embedding_dimension = len(vector)
                    self.index = self._create_index(self.embedding_dimension)

                # Validate dimension
                assert len(vector) == self.embedding_dimension, (
                    f"Embedding dimension mismatch: "
                    f"expected {self.embedding_dimension}, got {len(vector)}"
                )

                # Normalize vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                vectors.append(vector)
                ids.append(str(embedding.image_id))

            # Convert to batch matrix
            batch_vectors = np.vstack(vectors)

            # Add to index
            self.index.add(batch_vectors)
            self.embedding_ids.extend(ids)

            logger.info(f"Added {len(embeddings)} embeddings in batch")

        except Exception as e:
            error_msg = f"Failed to add embeddings batch: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def search(
        self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (image_id, similarity_score) tuples

        Raises:
            VectorSearchError: If search fails
        """
        assert query_embedding is not None, "Query embedding is required"
        assert k > 0, f"Invalid k value: {k}"
        assert 0.0 <= threshold <= 1.0, f"Invalid threshold: {threshold}"

        if self.index is None or self.index.ntotal == 0:
            logger.warning("No embeddings in index")
            return []

        try:
            # Prepare query vector
            query_vector = np.array(query_embedding, dtype=np.float32)

            # Validate dimension
            assert len(query_vector) == self.embedding_dimension, (
                f"Query dimension mismatch: "
                f"expected {self.embedding_dimension}, got {len(query_vector)}"
            )

            # Normalize query vector
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            query_vector = query_vector.reshape(1, -1)

            # Perform search
            scores, indices = self.index.search(query_vector, k)

            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break

                # Convert inner product to cosine similarity [0, 1]
                similarity = (score + 1) / 2

                if similarity >= threshold:
                    image_id = self.embedding_ids[idx]
                    results.append((image_id, float(similarity)))

            logger.debug(f"Found {len(results)} similar embeddings")
            return results

        except Exception as e:
            error_msg = f"Failed to search embeddings: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def remove_embedding(self, image_id: str) -> bool:
        """
        Remove embedding by image ID.

        Note: FAISS doesn't support efficient removal.
        This method rebuilds the index without the specified embedding.

        Args:
            image_id: Image ID to remove

        Returns:
            True if embedding was found and removed

        Raises:
            VectorSearchError: If removal fails
        """
        assert image_id is not None, "Image ID is required"

        if self.index is None or image_id not in self.embedding_ids:
            return False

        try:
            # Find index of the embedding to remove
            idx_to_remove = self.embedding_ids.index(image_id)

            # Get all vectors except the one to remove
            all_vectors = []
            remaining_ids = []

            for i in range(self.index.ntotal):
                if i != idx_to_remove:
                    vector = self.index.reconstruct(i)
                    all_vectors.append(vector)
                    remaining_ids.append(self.embedding_ids[i])

            # Rebuild index
            if all_vectors:
                batch_vectors = np.vstack(all_vectors)
                self.index = self._create_index(self.embedding_dimension)
                self.index.add(batch_vectors)
                self.embedding_ids = remaining_ids
            else:
                # No embeddings left
                self.index = self._create_index(self.embedding_dimension)
                self.embedding_ids = []

            logger.info(f"Removed embedding for image {image_id}")
            return True

        except Exception as e:
            error_msg = f"Failed to remove embedding: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def save_index(self, index_path: Path) -> None:
        """
        Save index to disk.

        Args:
            index_path: Path to save index

        Raises:
            VectorSearchError: If saving fails
        """
        assert index_path is not None, "Index path is required"

        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            # Create directory if needed
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss_path = index_path.with_suffix(".faiss")
            faiss.write_index(self.index, str(faiss_path))

            # Save metadata
            metadata = {
                "embedding_ids": self.embedding_ids,
                "embedding_dimension": self.embedding_dimension,
            }
            metadata_path = index_path.with_suffix(".pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved index to {index_path}")

        except Exception as e:
            error_msg = f"Failed to save index: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def load_index(self, index_path: Path) -> None:
        """
        Load index from disk.

        Args:
            index_path: Path to load index from

        Raises:
            VectorSearchError: If loading fails
        """
        assert index_path is not None, "Index path is required"

        try:
            faiss_path = index_path.with_suffix(".faiss")
            metadata_path = index_path.with_suffix(".pkl")

            if not faiss_path.exists() or not metadata_path.exists():
                raise VectorSearchError(f"Index files not found at {index_path}")

            # Load FAISS index
            self.index = faiss.read_index(str(faiss_path))

            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.embedding_ids = metadata["embedding_ids"]
            self.embedding_dimension = metadata["embedding_dimension"]

            logger.info(f"Loaded index from {index_path}")

        except Exception as e:
            error_msg = f"Failed to load index: {e}"
            logger.error(error_msg)
            raise VectorSearchError(error_msg) from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary containing index statistics
        """
        return {
            "total_embeddings": len(self.embedding_ids),
            "embedding_dimension": self.embedding_dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "is_trained": self.index.is_trained if self.index else False,
        }
