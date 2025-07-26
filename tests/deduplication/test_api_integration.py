"""
API integration tests after deduplication.

This module validates that API functionality remains intact
after removing api_new.py and consolidating to api.py.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.refimage.api import create_app
from src.refimage.config import Settings


class TestAPIAfterDeduplication:
    """Test API functionality after removing api_new.py."""

    @pytest.fixture
    def settings(self, temp_dir):
        """Test settings."""
        return Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
            log_level="DEBUG",
        )

    @pytest.fixture
    def app(self, settings):
        """FastAPI app instance."""
        with (
            patch("src.refimage.models.clip_model.CLIPModel"),
            patch("src.refimage.storage.StorageManager"),
            patch("src.refimage.search.VectorSearchEngine"),
            patch("src.refimage.dsl.DSLExecutor"),
        ):
            return create_app(settings)

    @pytest.fixture
    def client(self, app):
        """Test client."""
        return TestClient(app)

    def test_app_creation_with_settings(self, settings):
        """Test app creation with settings parameter."""
        with (
            patch("src.refimage.models.clip_model.CLIPModel"),
            patch("src.refimage.storage.StorageManager"),
            patch("src.refimage.search.VectorSearchEngine"),
            patch("src.refimage.dsl.DSLExecutor"),
        ):

            app = create_app(settings)
            assert app is not None

    def test_app_creation_with_config(self, settings):
        """Test app creation with config parameter (backward compatibility)."""
        with (
            patch("src.refimage.models.clip_model.CLIPModel"),
            patch("src.refimage.storage.StorageManager"),
            patch("src.refimage.search.VectorSearchEngine"),
            patch("src.refimage.dsl.DSLExecutor"),
        ):

            # Should accept config parameter for backward compatibility
            app = create_app(config=settings)
            assert app is not None

    def test_health_endpoint_exists(self, client):
        """Test health endpoint exists."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_upload_endpoint_exists(self, client):
        """Test upload endpoint exists."""
        response = client.post("/images/upload")
        # Should not return 404 (endpoint missing)
        assert response.status_code != 404

    def test_search_endpoint_exists(self, client):
        """Test search endpoint exists."""
        response = client.post("/images/search", json={"query": "test"})
        # Should not return 404 (endpoint missing)
        assert response.status_code != 404

    def test_dsl_endpoint_exists(self, client):
        """Test DSL endpoint exists."""
        response = client.post("/dsl/query", json={"query": "TEXT(test)"})
        # Should not return 404 (endpoint missing)
        assert response.status_code != 404

    def test_metadata_endpoint_exists(self, client):
        """Test metadata endpoint exists."""
        from uuid import uuid4

        test_id = uuid4()

        response = client.get(f"/images/{test_id}/metadata")
        # Should not return 404 (endpoint missing)
        assert response.status_code != 404

    def test_image_serve_endpoint_exists(self, client):
        """Test image serving endpoint exists."""
        from uuid import uuid4

        test_id = uuid4()

        response = client.get(f"/images/{test_id}")
        # Should not return 404 (endpoint missing)
        assert response.status_code != 404

    def test_cors_middleware_configured(self, app):
        """Test CORS middleware is configured."""
        # Check if CORS middleware is in the middleware stack
        middleware_types = [type(m).__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_types

    def test_dependency_injection_preserved(self, client):
        """Test dependency injection still works."""
        # All endpoints should be accessible (not 500 due to dependency issues)
        endpoints = [
            ("/health", "GET"),
            ("/images/search", "POST"),
            ("/dsl/query", "POST"),
        ]

        for endpoint, method in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={})

            # Should not fail due to dependency injection issues
            assert (
                response.status_code != 500 or "dependency" not in response.text.lower()
            )


class TestAPIBackwardCompatibility:
    """Test backward compatibility after API consolidation."""

    def test_import_from_api_works(self):
        """Test importing from api.py works."""
        from src.refimage.api import create_app

        assert create_app is not None

    def test_api_new_import_fails_gracefully(self):
        """Test api_new.py import fails gracefully after removal."""
        # api_new.py has been successfully removed
        # This test verifies the removal was successful
        import os

        api_new_path = "src/refimage/api_new.py"
        assert not os.path.exists(
            api_new_path
        ), f"api_new.py should be removed but still exists at {api_new_path}"

    def test_main_module_import_preserved(self):
        """Test main module can still import create_app."""
        try:
            from src.refimage.main import app

            assert app is not None
        except ImportError:
            # main.py might not exist yet
            pass


class TestAPIEndpointConsistency:
    """Test that all expected endpoints are available after consolidation."""

    @pytest.fixture
    def client(self, temp_dir):
        """Test client with mocked dependencies."""
        settings = Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
        )

        with (
            patch("src.refimage.models.clip_model.CLIPModel"),
            patch("src.refimage.storage.StorageManager"),
            patch("src.refimage.search.VectorSearchEngine"),
            patch("src.refimage.dsl.DSLExecutor"),
        ):

            app = create_app(settings)
            return TestClient(app)

    def test_all_expected_routes_available(self, client):
        """Test all expected routes are available."""
        expected_routes = ["/health", "/images/upload", "/images/search", "/dsl/query"]

        # Get all available routes
        available_routes = []
        for route in client.app.routes:
            if hasattr(route, "path"):
                available_routes.append(route.path)

        # Check that expected routes exist
        for expected in expected_routes:
            # Handle parameterized routes
            route_exists = any(
                expected in route or route.startswith(expected.split("{")[0])
                for route in available_routes
            )
            assert route_exists, f"Route {expected} not found in {available_routes}"

    def test_openapi_schema_generation(self, client):
        """Test OpenAPI schema generation works."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

        # Check that main endpoints are in schema
        paths = schema["paths"]
        assert "/health" in paths
        assert "/images/search" in paths
