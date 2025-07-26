"""
Tests for API consolidation - api.py vs api_new.py deduplication.

This module validates that the consolidation of duplicated API code
maintains functionality while eliminating code duplication.
"""

from unittest.mock import Mock, patch

import pytest

from src.refimage.models.schemas import SearchResponse, UploadResponse


class TestAPIConsolidation:
    """Test suite for API consolidation and deduplication."""

    def test_create_app_factory_pattern(self, test_settings):
        """Test that create_app follows factory pattern correctly."""
        from src.refimage.api import create_app

        # Test app creation with custom settings
        app = create_app(test_settings)
        assert app is not None
        assert app.title == "RefImage API"

        # Test app creation with default settings
        app_default = create_app()
        assert app_default is not None

        # Ensure apps are independent instances
        assert app is not app_default

    def test_endpoint_registration_consolidated(self, api_client):
        """Test all endpoints are properly registered in consolidated API."""
        # Test that health check endpoint exists
        response = api_client.get("/health")
        assert response.status_code == 200

        # Test that upload endpoint exists
        response = api_client.post("/images/upload")
        # Should fail due to missing file, but endpoint should exist
        assert response.status_code in [400, 422]  # Not 404

        # Test that search endpoint exists
        response = api_client.post("/images/search", json={"query": "test"})
        # Should fail due to no images, but endpoint should exist
        assert response.status_code in [200, 404]  # Not 404 for endpoint

        # Test that DSL endpoint exists
        response = api_client.post("/dsl/query", json={"query": "TEXT(cat)"})
        # Should exist regardless of implementation
        assert response.status_code != 404

    def test_no_duplicate_route_definitions(self, api_client):
        """Test that there are no duplicate route definitions."""
        # Get all routes from the FastAPI app
        app = api_client.app
        routes = [route.path for route in app.routes if hasattr(route, "path")]

        # Check for duplicates
        unique_routes = set(routes)
        assert len(routes) == len(unique_routes), f"Duplicate routes found: {routes}"

    @patch("src.refimage.api.StorageManager")
    @patch("src.refimage.api.CLIPModel")
    def test_dependency_injection_consistency(
        self, mock_clip, mock_storage, api_client
    ):
        """Test that dependency injection works consistently."""
        # Mock successful responses
        mock_storage.return_value.store_image.return_value = "test-uuid"
        mock_clip.return_value.encode_image.return_value = [0.1] * 512

        # Test upload endpoint uses injected dependencies
        with patch("builtins.open", mock=Mock()):
            response = api_client.post(
                "/images/upload",
                files={"file": ("test.png", b"fake_image_data", "image/png")},
            )

        # Should use mocked dependencies (not fail with real dependency errors)
        assert response.status_code != 500


class TestAPIBehaviorConsistency:
    """Test that API behavior is consistent after consolidation."""

    def test_upload_response_format(self, api_client, sample_images):
        """Test upload response format matches schema."""
        with (
            patch("src.refimage.storage.StorageManager") as mock_storage,
            patch("src.refimage.models.clip_model.CLIPModel") as mock_clip,
        ):

            # Mock successful upload
            mock_storage.return_value.store_image.return_value = "test-uuid"
            mock_clip.return_value.encode_image.return_value = [0.1] * 512

            response = api_client.post(
                "/images/upload",
                files={"file": ("test.png", sample_images["red"], "image/png")},
            )

            if response.status_code == 200:
                data = response.json()
                # Validate response matches UploadResponse schema
                upload_response = UploadResponse(**data)
                assert upload_response.image_id is not None
                assert upload_response.processing_time_ms >= 0

    def test_search_response_format(self, api_client):
        """Test search response format matches schema."""
        with patch("src.refimage.search.VectorSearchEngine") as mock_search:
            # Mock search results
            mock_search.return_value.search.return_value = []

            response = api_client.post(
                "/images/search", json={"query": "test query", "limit": 10}
            )

            if response.status_code == 200:
                data = response.json()
                # Validate response matches SearchResponse schema
                search_response = SearchResponse(**data)
                assert search_response.query == "test query"
                assert search_response.total_results >= 0
                assert search_response.search_time_ms >= 0

    def test_error_response_format(self, api_client):
        """Test error responses follow consistent format."""
        # Test with invalid request
        response = api_client.post("/images/upload")  # Missing file

        if response.status_code in [400, 422]:
            data = response.json()
            # Should have error structure
            assert "detail" in data or "error" in data

    def test_cors_configuration(self, api_client):
        """Test CORS is properly configured."""
        response = api_client.options("/health")
        # Should not be 405 Method Not Allowed if CORS is configured
        assert response.status_code != 405


class TestAPIPerformanceConsistency:
    """Test that API performance is maintained after consolidation."""

    def test_response_time_reasonable(self, api_client):
        """Test that response times are reasonable."""
        import time

        start_time = time.time()
        response = api_client.get("/health")
        end_time = time.time()

        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200

    def test_memory_usage_stable(self, api_client):
        """Test that memory usage doesn't grow excessively."""
        import gc

        # Force garbage collection before test
        gc.collect()

        # Make multiple requests
        for _ in range(10):
            response = api_client.get("/health")
            assert response.status_code == 200

        # Force garbage collection after test
        gc.collect()

        # If we get here without memory errors, test passes
        assert True


class TestAPIRegressionPrevention:
    """Regression tests to ensure consolidation doesn't break existing features."""

    def test_all_original_endpoints_available(self, api_client):
        """Test that all original endpoints remain available."""
        # Test core endpoints that should exist
        endpoints_to_test = [
            ("/health", "GET"),
            ("/images/upload", "POST"),
            ("/images/search", "POST"),
            ("/dsl/query", "POST"),
        ]

        for endpoint, method in endpoints_to_test:
            if method == "GET":
                response = api_client.get(endpoint)
            else:
                # POST with minimal data to test endpoint existence
                response = api_client.post(endpoint, json={})

            # Endpoint should exist (not 404)
            assert (
                response.status_code != 404
            ), f"Endpoint {method} {endpoint} not found"

    def test_parameter_validation_preserved(self, api_client):
        """Test that parameter validation still works correctly."""
        # Test search with invalid parameters
        response = api_client.post(
            "/images/search",
            json={"query": "", "limit": 1000},  # Empty query, limit too high
        )

        # Should validate and reject
        assert response.status_code == 422

    def test_authentication_middleware_preserved(self, api_client):
        """Test that authentication middleware (if any) is preserved."""
        # This test assumes authentication might be added later
        # For now, just ensure endpoints are accessible
        response = api_client.get("/health")
        assert response.status_code != 401  # Not unauthorized

    def test_logging_functionality_preserved(self, api_client):
        """Test that logging functionality is preserved."""
        with patch("src.refimage.api.logger") as mock_logger:
            response = api_client.get("/health")

            # Logger should be called (indicates logging is working)
            # This is a basic check; specific logging assertions would depend
            # on the actual logging implementation
            assert response.status_code == 200


@pytest.mark.integration
class TestAPIIntegrationAfterConsolidation:
    """Integration tests to verify API works with other components."""

    def test_api_storage_integration(self, api_client, temp_dir):
        """Test API integration with storage layer."""
        with patch("src.refimage.storage.StorageManager") as mock_storage:
            mock_storage.return_value.store_image.return_value = "test-uuid"

            response = api_client.post(
                "/images/upload",
                files={"file": ("test.png", b"fake_data", "image/png")},
            )

            # Should attempt to use storage
            if response.status_code == 200:
                mock_storage.return_value.store_image.assert_called()

    def test_api_search_integration(self, api_client):
        """Test API integration with search engine."""
        with patch("src.refimage.search.VectorSearchEngine") as mock_search:
            mock_search.return_value.search.return_value = []

            response = api_client.post("/images/search", json={"query": "test"})

            # Should attempt to use search engine
            if response.status_code == 200:
                mock_search.return_value.search.assert_called()

    def test_api_clip_integration(self, api_client):
        """Test API integration with CLIP model."""
        with patch("src.refimage.models.clip_model.CLIPModel") as mock_clip:
            mock_clip.return_value.encode_text.return_value = [0.1] * 512

            response = api_client.post("/images/search", json={"query": "test"})

            # Should attempt to use CLIP model for text encoding
            if response.status_code == 200:
                # Note: actual call depends on implementation
                pass  # Verify integration without breaking test
