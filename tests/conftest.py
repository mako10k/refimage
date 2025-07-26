"""
Test configuration and shared fixtures for RefImage test suite.

This module provides common test fixtures, utilities, and configuration
for maintaining consistency across all test categories.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import Mock
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.refimage.config import Settings
from src.refimage.models.schemas import ImageEmbedding, ImageMetadata


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test-specific settings configuration."""
    return Settings(
        # Use in-memory database for tests
        database_url="sqlite:///:memory:",
        # Use temporary directory for file storage
        image_storage_path="/tmp/refimage_test_images",
        # Disable external API calls in tests
        clip_model_name="ViT-B/32",
        # Test-specific logging
        log_level="DEBUG",
        # Disable caching for predictable tests
        enable_caching=False,
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_image() -> Generator[bytes, None, None]:
    """Generate test image data."""
    # Create a simple 100x100 red image
    image = Image.new("RGB", (100, 100), color="red")
    temp_path = Path(tempfile.mktemp(suffix=".png"))
    try:
        image.save(temp_path)
        with open(temp_path, "rb") as f:
            yield f.read()
    finally:
        temp_path.unlink(missing_ok=True)


@pytest.fixture
def test_image_metadata() -> ImageMetadata:
    """Sample image metadata for testing."""
    return ImageMetadata(
        id=uuid4(),
        filename="test_image.png",
        file_path=Path("/tmp/test_image.png"),
        file_size=1024,
        mime_type="image/png",
        width=100,
        height=100,
        tags=["test", "red"],
        description="Test image for unit tests",
    )


@pytest.fixture
def test_embedding() -> ImageEmbedding:
    """Sample CLIP embedding for testing."""
    # Generate a 512-dimensional random embedding (typical for CLIP ViT-B/32)
    embedding_vector = np.random.rand(512).tolist()

    return ImageEmbedding(
        image_id=uuid4(),
        embedding=embedding_vector,
        model_name="ViT-B/32",
    )


@pytest.fixture
def mock_clip_model() -> Mock:
    """Mock CLIP model for testing without loading real model."""
    mock_model = Mock()
    mock_model.encode_image.return_value = np.random.rand(512)
    mock_model.encode_text.return_value = np.random.rand(512)
    mock_model.model_name = "ViT-B/32"
    mock_model.device = "cpu"
    mock_model.dimension = 512
    return mock_model


@pytest.fixture
def mock_storage_manager() -> Mock:
    """Mock storage manager for testing without file I/O."""
    mock_storage = Mock()
    mock_storage.store_image.return_value = uuid4()
    mock_storage.get_image_metadata.return_value = None
    mock_storage.delete_image.return_value = True
    mock_storage.list_images.return_value = []
    return mock_storage


@pytest.fixture
def mock_vector_search() -> Mock:
    """Mock vector search engine for testing without FAISS."""
    mock_search = Mock()
    mock_search.add_embedding.return_value = True
    mock_search.search.return_value = []
    mock_search.get_index_size.return_value = 0
    return mock_search


@pytest.fixture
def api_client(test_settings: Settings) -> TestClient:
    """FastAPI test client with test configuration."""
    from src.refimage.api import create_app

    app = create_app(test_settings)
    return TestClient(app)


@pytest.fixture
def sample_images() -> Dict[str, bytes]:
    """Collection of sample images for testing different scenarios."""
    images = {}

    # Red image
    red_image = Image.new("RGB", (100, 100), color="red")
    red_path = Path(tempfile.mktemp(suffix=".png"))
    red_image.save(red_path)
    with open(red_path, "rb") as f:
        images["red"] = f.read()
    red_path.unlink()

    # Blue image
    blue_image = Image.new("RGB", (100, 100), color="blue")
    blue_path = Path(tempfile.mktemp(suffix=".png"))
    blue_image.save(blue_path)
    with open(blue_path, "rb") as f:
        images["blue"] = f.read()
    blue_path.unlink()

    # Green image
    green_image = Image.new("RGB", (200, 150), color="green")
    green_path = Path(tempfile.mktemp(suffix=".jpg"))
    green_image.save(green_path, format="JPEG")
    with open(green_path, "rb") as f:
        images["green"] = f.read()
    green_path.unlink()

    return images


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup logic here if needed
    # For now, using in-memory database and temp directories
    # which are automatically cleaned up
    pass


# Test utilities
class TestDataBuilder:
    """Builder pattern for creating test data consistently."""

    @staticmethod
    def create_test_embedding(
        image_id: Optional[UUID] = None, dimension: int = 512
    ) -> ImageEmbedding:
        """Create test embedding with specified parameters."""
        return ImageEmbedding(
            image_id=uuid4() if image_id is None else image_id,
            embedding=np.random.rand(dimension).tolist(),
            model_name="ViT-B/32",
        )

    @staticmethod
    def create_test_metadata(
        filename: str = "test.png", **kwargs: Any
    ) -> ImageMetadata:
        """Create test metadata with specified parameters."""
        defaults: Dict[str, Any] = {
            "id": uuid4(),
            "filename": filename,
            "file_path": Path(f"/tmp/{filename}"),
            "file_size": 1024,
            "mime_type": "image/png",
            "width": 100,
            "height": 100,
            "tags": [],
            "description": None,
        }
        defaults.update(kwargs)
        return ImageMetadata(**defaults)


# Test markers for categorizing tests
pytest_marks = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
    "deduplication": pytest.mark.deduplication,
    "driver": pytest.mark.driver,
    "slow": pytest.mark.slow,
    "requires_gpu": pytest.mark.requires_gpu,
    "requires_internet": pytest.mark.requires_internet,
}


# Custom assertions for RefImage-specific testing
def assert_embedding_valid(embedding: ImageEmbedding):
    """Assert that an embedding is valid according to RefImage specs."""
    assert embedding.image_id is not None
    assert len(embedding.embedding) > 0
    assert len(embedding.embedding) <= 2048
    assert all(isinstance(x, (int, float)) for x in embedding.embedding)
    assert embedding.model_name is not None


def assert_metadata_valid(metadata: ImageMetadata):
    """Assert that image metadata is valid according to RefImage specs."""
    assert metadata.id is not None
    assert metadata.filename is not None
    assert metadata.file_path is not None
    assert metadata.file_size > 0
    assert metadata.mime_type is not None
    assert metadata.width > 0
    assert metadata.height > 0
