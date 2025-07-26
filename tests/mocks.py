"""
Mock implementations for RefImage testing.

This module provides lightweight mock implementations for heavy dependencies
to prevent Test Fallback and Test Avoidance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from src.refimage.models.schemas import ImageEmbedding, ImageMetadata


class MockCLIPModel:
    """Mock CLIP model that generates deterministic dummy embeddings."""

    def __init__(self, model_name: str = "mock-clip", embedding_dim: int = 512):
        """Initialize mock CLIP model."""
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = "cpu"
        self._load_success = True

    def encode_image(self, image) -> np.ndarray:
        """Generate deterministic dummy embedding for image."""
        assert image is not None, "Image cannot be None"

        if not self._load_success:
            raise RuntimeError("Mock CLIP model failed to load")

        # Generate deterministic embedding based on image properties
        if hasattr(image, "size"):
            seed = hash((image.size[0], image.size[1])) % 2**32
        else:
            seed = hash(str(image)) % 2**32

        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """Generate deterministic dummy embedding for text."""
        assert text is not None, "Text cannot be None"
        assert len(text.strip()) > 0, "Text cannot be empty"

        if not self._load_success:
            raise RuntimeError("Mock CLIP model failed to load")

        # Generate deterministic embedding based on text hash
        seed = hash(text.lower().strip()) % 2**32
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate batch embeddings for texts."""
        return [self.encode_text(text) for text in texts]

    def encode_images_batch(self, images: List) -> List[np.ndarray]:
        """Generate batch embeddings for images."""
        return [self.encode_image(image) for image in images]

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "mock": True,
        }

    def simulate_load_failure(self):
        """Simulate model loading failure for error testing."""
        self._load_success = False

    def restore_normal_operation(self):
        """Restore normal operation after simulated failure."""
        self._load_success = True


class MockFAISSIndex:
    """Mock FAISS index for vector similarity search."""

    def __init__(self, dimension: int):
        """Initialize mock FAISS index."""
        self.dimension = dimension
        self.vectors = []
        self.vector_ids = []
        self.ntotal = 0
        self.is_trained = True
        self._index_corrupted = False

    def add(self, vectors: np.ndarray):
        """Add vectors to mock index."""
        if self._index_corrupted:
            raise RuntimeError("Mock FAISS index is corrupted")

        assert vectors.ndim == 2, "Vectors must be 2D array"
        assert (
            vectors.shape[1] == self.dimension
        ), f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"

        for vector in vectors:
            self.vectors.append(vector.copy())
        self.ntotal = len(self.vectors)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple:
        """Search for similar vectors."""
        if self._index_corrupted:
            raise RuntimeError("Mock FAISS index is corrupted")

        assert query_vectors.ndim == 2, "Query vectors must be 2D array"
        assert (
            query_vectors.shape[1] == self.dimension
        ), "Query vector dimension mismatch"
        assert k > 0, "k must be positive"

        if self.ntotal == 0:
            # Return empty results
            return np.array([[-1] * k]), np.array([[0.0] * k])

        scores = []
        indices = []

        for query_vector in query_vectors:
            # Normalize query vector for proper cosine similarity
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                query_normalized = query_vector
            else:
                query_normalized = query_vector / query_norm

            # Calculate cosine similarity with all stored vectors
            similarities = []
            for i, stored_vector in enumerate(self.vectors):
                # Normalize stored vector
                stored_norm = np.linalg.norm(stored_vector)
                if stored_norm == 0:
                    stored_normalized = stored_vector
                else:
                    stored_normalized = stored_vector / stored_norm

                # Proper cosine similarity calculation
                similarity = float(np.dot(query_normalized, stored_normalized))
                # Ensure similarity is within [-1, 1] range
                similarity = max(min(similarity, 1.0), -1.0)
                similarities.append((similarity, i))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Get top k results
            top_k = similarities[: min(k, len(similarities))]

            # Extract scores and indices
            k_scores = [sim for sim, _ in top_k]
            k_indices = [idx for _, idx in top_k]

            # Pad with -1 if needed
            while len(k_scores) < k:
                k_scores.append(0.0)
                k_indices.append(-1)

            scores.append(k_scores)
            indices.append(k_indices)

        return np.array(indices), np.array(scores)

    def reconstruct(self, index: int) -> np.ndarray:
        """Reconstruct vector by index."""
        if self._index_corrupted:
            raise RuntimeError("Mock FAISS index is corrupted")

        assert 0 <= index < self.ntotal, f"Index out of range: {index}"
        return self.vectors[index].copy()

    def simulate_corruption(self):
        """Simulate index corruption for error testing."""
        self._index_corrupted = True

    def restore_normal_operation(self):
        """Restore normal operation after simulated corruption."""
        self._index_corrupted = False


class MockStorageManager:
    """Mock storage manager for testing without file system dependencies."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize mock storage manager."""
        self.temp_dir = temp_dir or Path("/tmp/mock_storage")
        self.images = {}  # UUID -> ImageMetadata
        self.embeddings = {}  # UUID -> ImageEmbedding
        self._storage_corrupted = False

    def store_image(
        self,
        image_data: bytes,
        original_filename: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ImageMetadata:
        """Store image with mock file system."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        assert image_data is not None, "Image data cannot be None"
        assert original_filename is not None, "Filename cannot be None"

        # Create mock metadata
        image_id = uuid4()
        metadata = ImageMetadata(
            id=image_id,
            filename=original_filename,
            file_path=self.temp_dir / f"{image_id}.jpg",
            file_size=len(image_data),
            mime_type="image/jpeg",
            width=100,  # Mock dimensions
            height=100,
            description=description,
            tags=tags or [],
        )

        self.images[image_id] = metadata
        return metadata

    def get_image_metadata(self, image_id: UUID) -> Optional[ImageMetadata]:
        """Get image metadata by ID."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        return self.images.get(image_id)

    def store_embedding(self, embedding: ImageEmbedding) -> None:
        """Store image embedding."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        assert embedding is not None, "Embedding cannot be None"
        self.embeddings[embedding.image_id] = embedding

    def get_embedding(self, image_id: UUID) -> Optional[ImageEmbedding]:
        """Get embedding by image ID."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        return self.embeddings.get(image_id)

    def get_all_embeddings(self) -> List[ImageEmbedding]:
        """Get all embeddings."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        return list(self.embeddings.values())

    def list_images(
        self, limit: int = 10, offset: int = 0, tags_filter: Optional[List[str]] = None
    ) -> List[ImageMetadata]:
        """List images with pagination and filtering."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        images = list(self.images.values())

        # Apply tag filter
        if tags_filter:
            filtered_images = []
            for image in images:
                if any(tag in image.tags for tag in tags_filter):
                    filtered_images.append(image)
            images = filtered_images

        # Apply pagination
        return images[offset : offset + limit]

    def delete_image(self, image_id: UUID) -> bool:
        """Delete image and its embedding."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        if image_id in self.images:
            del self.images[image_id]
            if image_id in self.embeddings:
                del self.embeddings[image_id]
            return True
        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if self._storage_corrupted:
            raise RuntimeError("Mock storage is corrupted")

        return {
            "total_images": len(self.images),
            "total_embeddings": len(self.embeddings),
            "mock": True,
        }

    def simulate_corruption(self):
        """Simulate storage corruption for error testing."""
        self._storage_corrupted = True

    def restore_normal_operation(self):
        """Restore normal operation after simulated corruption."""
        self._storage_corrupted = False


def create_mock_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create mock PIL Image for testing."""
    return Image.new("RGB", (width, height), color="red")
