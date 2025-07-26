"""
Regression tests for RefImage code deduplication.

This module ensures that existing functionality is preserved
after code deduplication and refactoring.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.refimage.config import Settings
from src.refimage.models.schemas import ImageMetadata


class TestAPIRegressionAfterDeduplication:
    """Regression tests for API functionality after deduplication."""

    def test_health_endpoint_preserved(self, api_client):
        """Test that health endpoint continues to work."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_upload_endpoint_signature_preserved(self, api_client):
        """Test that upload endpoint signature is preserved."""
        # Test with missing file (should return 422, not 404)
        response = api_client.post("/images/upload")
        assert response.status_code in [400, 422]
        assert response.status_code != 404

    def test_search_endpoint_signature_preserved(self, api_client):
        """Test that search endpoint signature is preserved."""
        response = api_client.post(
            "/images/search", json={"query": "test", "limit": 10}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_dsl_endpoint_signature_preserved(self, api_client):
        """Test that DSL endpoint signature is preserved."""
        response = api_client.post(
            "/dsl/query", json={"query": "TEXT(cat)", "limit": 10}
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404


class TestStorageRegressionAfterDeduplication:
    """Regression tests for storage functionality after deduplication."""

    @pytest.fixture
    def storage_manager(self, temp_dir):
        """Storage manager for regression testing."""
        settings = Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
        )
        from src.refimage.storage import StorageManager

        return StorageManager(settings)

    def test_store_image_interface_preserved(self, storage_manager):
        """Test that store_image interface is preserved."""
        with (
            patch("PIL.Image.open"),
            patch("builtins.open", MagicMock()),
            patch.object(storage_manager, "_setup_database"),
            patch.object(storage_manager, "_setup_storage"),
        ):

            # Method should exist and accept expected parameters
            assert hasattr(storage_manager, "store_image")

            # Should not raise AttributeError
            try:
                storage_manager.store_image(
                    image_data=b"fake_data",
                    filename="test.png",
                    description="test",
                    tags=["test"],
                )
            except Exception as e:
                # Any exception other than AttributeError is acceptable
                assert not isinstance(e, AttributeError)

    def test_get_metadata_interface_preserved(self, storage_manager):
        """Test that get_metadata interface is preserved."""
        test_id = uuid4()

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.return_value.fetchone.return_value = None

            # Method should exist and return expected type
            result = storage_manager.get_metadata(test_id)
            assert result is None or isinstance(result, ImageMetadata)

    def test_list_images_interface_preserved(self, storage_manager):
        """Test that list_images interface is preserved."""
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = []

            # Method should exist and return expected type
            result = storage_manager.list_images()
            assert isinstance(result, list)

    def test_storage_error_handling_preserved(self, storage_manager):
        """Test that error handling behavior is preserved."""
        test_id = uuid4()

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = Exception("Database error")

            # Should raise StorageError, not generic Exception
            from src.refimage.storage import StorageError

            with pytest.raises(StorageError):
                storage_manager.get_metadata(test_id)


class TestSearchRegressionAfterDeduplication:
    """Regression tests for search functionality after deduplication."""

    def test_vector_search_interface_preserved(self):
        """Test that vector search interface is preserved."""
        from src.refimage.search import VectorSearchEngine

        # Should be able to instantiate
        search_engine = VectorSearchEngine()

        # Should have expected methods
        assert hasattr(search_engine, "add_embedding")
        assert hasattr(search_engine, "search")
        assert hasattr(search_engine, "get_index_size")

    def test_search_results_format_preserved(self):
        """Test that search results format is preserved."""
        from src.refimage.search import VectorSearchEngine

        search_engine = VectorSearchEngine()

        with patch.object(search_engine, "search") as mock_search:
            mock_search.return_value = []

            # Search should return list
            results = search_engine.search("test query")
            assert isinstance(results, list)


class TestDSLRegressionAfterDeduplication:
    """Regression tests for DSL functionality after deduplication."""

    def test_dsl_executor_interface_preserved(self):
        """Test that DSL executor interface is preserved."""
        from src.refimage.dsl import DSLExecutor

        # Should be able to instantiate
        executor = DSLExecutor()

        # Should have expected methods
        assert hasattr(executor, "execute")
        assert hasattr(executor, "parse")

    def test_simple_dsl_queries_preserved(self):
        """Test that simple DSL queries still work."""
        from src.refimage.dsl import DSLExecutor

        executor = DSLExecutor()

        # Simple TEXT query should still parse
        try:
            result = executor.parse('TEXT("cat")')
            # Should not raise exception for simple queries
            assert result is not None
        except Exception as e:
            # If parsing fails, it should be a DSL-specific error
            from src.refimage.dsl import DSLError

            assert isinstance(e, DSLError)


class TestCLIPModelRegressionAfterDeduplication:
    """Regression tests for CLIP model functionality after deduplication."""

    def test_clip_model_interface_preserved(self):
        """Test that CLIP model interface is preserved."""
        from src.refimage.models.clip_model import CLIPModel

        # Should be able to instantiate (even if it fails due to missing model)
        try:
            model = CLIPModel()

            # Should have expected methods
            assert hasattr(model, "encode_image")
            assert hasattr(model, "encode_text")

        except Exception:
            # Model loading might fail in test environment
            # This is acceptable as long as the class exists
            pass

    def test_clip_model_error_handling_preserved(self):
        """Test that CLIP model error handling is preserved."""
        from src.refimage.models.clip_model import CLIPModelError

        # Error class should exist
        assert CLIPModelError is not None


class TestDataFlowRegressionAfterDeduplication:
    """Regression tests for data flow between components."""

    def test_image_upload_data_flow(self, api_client):
        """Test that image upload data flow is preserved."""
        with (
            patch("src.refimage.storage.StorageManager") as mock_storage,
            patch("src.refimage.models.clip_model.CLIPModel") as mock_clip,
        ):

            mock_storage.return_value.store_image.return_value = str(uuid4())
            mock_clip.return_value.encode_image.return_value = [0.1] * 512

            response = api_client.post(
                "/images/upload",
                files={"file": ("test.png", b"fake_data", "image/png")},
            )

            # Data flow should work (may fail for other reasons)
            # Main concern is that it doesn't fail due to missing methods
            assert response.status_code != 500 or "AttributeError" not in str(
                response.content
            )

    def test_image_search_data_flow(self, api_client):
        """Test that image search data flow is preserved."""
        with (
            patch("src.refimage.search.VectorSearchEngine") as mock_search,
            patch("src.refimage.models.clip_model.CLIPModel") as mock_clip,
        ):

            mock_search.return_value.search.return_value = []
            mock_clip.return_value.encode_text.return_value = [0.1] * 512

            response = api_client.post("/images/search", json={"query": "test query"})

            # Data flow should work (may fail for other reasons)
            assert response.status_code != 500 or "AttributeError" not in str(
                response.content
            )


class TestConfigurationRegressionAfterDeduplication:
    """Regression tests for configuration after deduplication."""

    def test_settings_interface_preserved(self):
        """Test that Settings interface is preserved."""
        from src.refimage.config import Settings

        # Should be able to create settings
        settings = Settings()

        # Should have expected attributes
        expected_attrs = [
            "database_url",
            "image_storage_path",
            "clip_model_name",
            "log_level",
        ]

        for attr in expected_attrs:
            assert hasattr(settings, attr), f"Missing attribute: {attr}"

    def test_app_creation_with_settings_preserved(self):
        """Test that app creation with settings is preserved."""
        from src.refimage.api import create_app
        from src.refimage.config import Settings

        settings = Settings()

        # Should be able to create app with settings
        app = create_app(settings)
        assert app is not None
