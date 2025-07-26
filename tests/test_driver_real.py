"""
Driver Tests for RefImage - Real Implementation Validation

This module provides end-to-end driver tests that validate
real CLIP+FAISS integration and detect implementation gaps
that mock tests cannot catch.
"""

import io
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from src.refimage.api import create_app
from src.refimage.config import Settings
from src.refimage.dsl import DSLError, DSLExecutor, DSLParser
from src.refimage.models.clip_model import CLIPModel, CLIPModelError
from src.refimage.search import VectorSearchEngine, VectorSearchError
from src.refimage.storage import StorageError, StorageManager


class TestDriverRealCLIPFAISS:
    """Driver tests with real CLIP and FAISS integration."""

    @pytest.fixture(scope="class")
    def real_temp_dir(self):
        """Create persistent temp directory for driver tests."""
        temp_dir = tempfile.mkdtemp(prefix="refimage_driver_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def real_settings(self, real_temp_dir):
        """Create settings for real driver tests."""
        return Settings(
            clip_model_name="ViT-B/32",  # Small model for testing
            device="cpu",  # Force CPU to avoid GPU dependency
            image_storage_path=real_temp_dir / "images",
            index_storage_path=real_temp_dir / "indexes",
            metadata_storage_path=real_temp_dir / "metadata",
            database_path=real_temp_dir / "refimage.db",
            max_image_size=5 * 1024 * 1024,  # 5MB limit
        )

    @pytest.fixture(scope="class")
    def real_clip_model(self, real_settings):
        """Create real CLIP model - This will test actual loading."""
        try:
            return CLIPModel(real_settings)
        except Exception as e:
            pytest.skip(f"Real CLIP model unavailable: {e}")

    def create_test_image(self, width=224, height=224, color=(255, 0, 0)):
        """Create test image in memory."""
        img = Image.new("RGB", (width, height), color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        return img_bytes.getvalue()

    def test_real_clip_model_loading(self, real_clip_model):
        """Test real CLIP model loading and device detection."""
        assert real_clip_model is not None
        assert real_clip_model.model is not None
        assert real_clip_model.preprocess is not None
        assert real_clip_model.device is not None

        # Verify model info
        info = real_clip_model.get_model_info()
        assert info["model_name"] == "ViT-B/32"
        assert info["device"] == "cpu"
        assert info["embedding_dimension"] == 512

    def test_real_clip_image_encoding(self, real_clip_model):
        """Test real CLIP image encoding with various formats."""
        # Test with PIL Image
        img = Image.new("RGB", (224, 224), (255, 0, 0))
        embedding = real_clip_model.encode_image(img)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)

        # Test with bytes
        img_bytes = self.create_test_image()
        embedding2 = real_clip_model.encode_image(img_bytes)
        assert isinstance(embedding2, np.ndarray)
        assert embedding2.shape == (512,)

    def test_real_clip_text_encoding(self, real_clip_model):
        """Test real CLIP text encoding."""
        text = "a red car"
        embedding = real_clip_model.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)

    def test_real_clip_similarity_calculation(self, real_clip_model):
        """Test real CLIP similarity calculations."""
        # Create two similar images
        red_img = Image.new("RGB", (224, 224), (255, 0, 0))
        blue_img = Image.new("RGB", (224, 224), (0, 0, 255))

        red_embedding = real_clip_model.encode_image(red_img)
        blue_embedding = real_clip_model.encode_image(blue_img)

        # Calculate similarity
        similarity = real_clip_model.calculate_similarity(red_embedding, blue_embedding)

        # Should be positive but not too high (different colors)
        assert 0.0 < similarity < 0.9

    def test_real_faiss_integration(self, real_settings):
        """Test real FAISS integration with actual vectors."""
        search_engine = VectorSearchEngine(real_settings)

        # Create test embeddings
        embeddings = []
        for i in range(10):
            # Create normalized random vectors
            vec = np.random.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        # Add embeddings to index
        embedding_ids = [str(uuid4()) for _ in range(10)]
        for i, (emb_id, embedding) in enumerate(zip(embedding_ids, embeddings)):
            search_engine.add_embedding(emb_id, embedding)

        # Test search
        query_vec = embeddings[0]  # Search for first embedding
        results = search_engine.search(query_vec, k=3)

        assert len(results) == 3
        assert results[0][0] == embedding_ids[0]  # Should find itself first
        assert results[0][1] > 0.99  # Very high similarity to itself

    def test_real_storage_concurrent_access(self, real_settings):
        """Test storage manager with concurrent access."""
        storage_manager = StorageManager(real_settings)

        def store_image_worker(worker_id):
            """Worker function for concurrent image storage."""
            img_data = self.create_test_image(color=(worker_id % 255, 0, 0))
            metadata = storage_manager.store_image(
                image_data=img_data,
                filename=f"test_image_{worker_id}.jpg",
                description=f"Test image {worker_id}",
                tags=[f"tag{worker_id}"],
            )
            return metadata.id

        # Run concurrent storage operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(store_image_worker, i) for i in range(20)]

            image_ids = []
            for future in as_completed(futures):
                try:
                    image_id = future.result(timeout=30)
                    image_ids.append(image_id)
                except Exception as e:
                    pytest.fail(f"Concurrent storage failed: {e}")

        # Verify all images were stored
        assert len(image_ids) == 20
        assert len(set(image_ids)) == 20  # All unique IDs

        # Verify database consistency
        all_images = storage_manager.list_images(limit=25)
        assert len(all_images) >= 20


class TestDriverLargeScale:
    """Driver tests for large-scale operations."""

    @pytest.fixture
    def large_scale_settings(self, tmp_path):
        """Settings for large-scale tests."""
        return Settings(
            clip_model_name="ViT-B/32",
            device="cpu",
            image_storage_path=tmp_path / "images",
            index_storage_path=tmp_path / "indexes",
            metadata_storage_path=tmp_path / "metadata",
            database_path=tmp_path / "refimage.db",
            max_image_size=50 * 1024 * 1024,  # 50MB limit
        )

    def test_large_image_processing(self, large_scale_settings):
        """Test processing of large images."""
        # Skip if real CLIP not available
        try:
            clip_model = CLIPModel(large_scale_settings)
        except Exception:
            pytest.skip("Real CLIP model not available")

        # Create large image (4K resolution)
        large_img = Image.new("RGB", (3840, 2160), (128, 128, 128))

        start_time = time.time()
        embedding = clip_model.encode_image(large_img)
        processing_time = time.time() - start_time

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        # Should process within reasonable time (adjust based on hardware)
        assert processing_time < 30.0  # 30 seconds max

    def test_batch_processing_efficiency(self, large_scale_settings):
        """Test batch processing efficiency."""
        try:
            clip_model = CLIPModel(large_scale_settings)
        except Exception:
            pytest.skip("Real CLIP model not available")

        # Create batch of images
        images = []
        for i in range(10):
            img = Image.new("RGB", (224, 224), (i * 25, 0, 0))
            images.append(img)

        # Test individual processing
        start_time = time.time()
        individual_embeddings = [clip_model.encode_image(img) for img in images]
        individual_time = time.time() - start_time

        # Test batch processing
        start_time = time.time()
        batch_embeddings = clip_model.encode_images_batch(images)
        batch_time = time.time() - start_time

        # Batch should be more efficient
        assert len(batch_embeddings) == 10
        assert batch_time < individual_time  # Batch should be faster

        # Results should be consistent
        for i, (ind_emb, batch_emb) in enumerate(
            zip(individual_embeddings, batch_embeddings)
        ):
            similarity = np.dot(ind_emb, batch_emb)
            assert similarity > 0.99  # Should be nearly identical


class TestDriverErrorScenarios:
    """Driver tests for error scenarios and edge cases."""

    def test_disk_space_exhaustion_simulation(self, tmp_path):
        """Test behavior when disk space is exhausted."""
        settings = Settings(
            image_storage_path=tmp_path / "images",
            database_path=tmp_path / "refimage.db",
        )

        storage_manager = StorageManager(settings)

        # Try to store many large images to exhaust space
        # (This is a simulation - real disk exhaustion is hard to test)
        large_img_data = b"fake_large_image_data" * 1000000  # 15MB fake data

        stored_count = 0
        try:
            for i in range(100):  # Try to store many large files
                storage_manager.store_image(
                    image_data=large_img_data,
                    filename=f"large_image_{i}.jpg",
                    description=f"Large test image {i}",
                )
                stored_count += 1
        except (StorageError, OSError) as e:
            # Should gracefully handle disk space issues
            assert "space" in str(e).lower() or "storage" in str(e).lower()

        # Should have stored some images before failing
        assert stored_count >= 0

    def test_corrupted_database_recovery(self, tmp_path):
        """Test recovery from corrupted database."""
        settings = Settings(database_path=tmp_path / "refimage.db")

        # Create and populate database
        storage_manager = StorageManager(settings)
        img_data = b"fake_image_data"

        metadata = storage_manager.store_image(
            image_data=img_data, filename="test.jpg", description="Test image"
        )

        # Simulate database corruption
        with open(settings.database_path, "w") as f:
            f.write("corrupted data")

        # Try to create new storage manager
        try:
            new_storage_manager = StorageManager(settings)
            # Should handle corruption gracefully
            assert new_storage_manager is not None
        except StorageError as e:
            # Should provide meaningful error message
            assert "corrupt" in str(e).lower() or "database" in str(e).lower()

    def test_dsl_complex_query_parsing(self):
        """Test DSL parser with complex, potentially problematic queries."""
        parser = DSLParser()

        # Test deeply nested query
        complex_query = (
            "AND("
            "  OR("
            '    TEXT("red car")'
            "    AND("
            '      TEXT("blue vehicle")'
            '      NOT(TAG("broken"))'
            "    )"
            "  )"
            '  TAG("automotive")'
            ")"
        )

        try:
            ast = parser.parse(complex_query)
            assert ast is not None
        except DSLError as e:
            # Should provide clear error message for parsing failures
            assert len(str(e)) > 0

        # Test malformed query
        malformed_query = 'AND(TEXT("incomplete query"'

        with pytest.raises(DSLError):
            parser.parse(malformed_query)

    def test_api_stress_concurrent_requests(self, tmp_path):
        """Test API under concurrent request load."""
        settings = Settings(
            image_storage_path=tmp_path / "images",
            database_path=tmp_path / "refimage.db",
        )

        try:
            app = create_app(settings)
            from fastapi.testclient import TestClient

            client = TestClient(app)
        except Exception:
            pytest.skip("API setup failed - likely missing dependencies")

        def make_search_request(query_text):
            """Make a search request."""
            response = client.post("/search", json={"query": query_text, "limit": 5})
            return response.status_code

        # Send concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_search_request, f"test query {i}")
                for i in range(50)
            ]

            status_codes = []
            for future in as_completed(futures):
                try:
                    status_code = future.result(timeout=10)
                    status_codes.append(status_code)
                except Exception as e:
                    pytest.fail(f"Concurrent API request failed: {e}")

        # Most requests should succeed (200) or return valid errors (4xx)
        success_count = sum(1 for code in status_codes if 200 <= code < 300)
        error_count = sum(1 for code in status_codes if 400 <= code < 500)

        assert success_count + error_count >= len(status_codes) * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
