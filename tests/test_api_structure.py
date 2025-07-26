"""
Test API structure and create_app() factory pattern.

Tests for ensuring proper API endpoint registration and
Swagger documentation generation after refactoring.
"""

import pytest
from fastapi.testclient import TestClient

from src.refimage.api import create_app
from src.refimage.config import get_config


class TestAPIStructure:
    """Test proper API structure and endpoint registration."""

    def test_create_app_returns_valid_fastapi_instance(self):
        """Test that create_app() returns a properly configured FastAPI instance."""
        app = create_app()

        # Contract Programming: Validate app instance
        assert app is not None, "create_app() must return an app instance"
        assert hasattr(app, "routes"), "App must have routes attribute"
        assert hasattr(app, "openapi"), "App must support OpenAPI generation"

        # Validate app configuration
        assert app.title == "RefImage API", "App title must be correctly set"
        assert app.version is not None, "App version must be set"

    def test_endpoints_registered_in_create_app(self):
        """Test that all endpoints are properly registered via create_app()."""
        app = create_app()
        client = TestClient(app)

        # Contract Programming: Essential endpoints must exist
        expected_endpoints = [
            "/health",  # Health check
            "/images/upload",  # Image upload
            "/images/search",  # Image search
            "/images/dsl",  # DSL query
            "/docs",  # Swagger UI
            "/openapi.json",  # OpenAPI spec
        ]

        # Get all registered routes
        registered_paths = {route.path for route in app.routes}

        for endpoint in expected_endpoints:
            assert endpoint in registered_paths or any(
                endpoint in path for path in registered_paths
            ), f"Endpoint {endpoint} must be registered in create_app()"

    def test_swagger_generation_works(self):
        """Test that Swagger/OpenAPI documentation is properly generated."""
        app = create_app()
        client = TestClient(app)

        # Test OpenAPI JSON generation
        response = client.get("/openapi.json")
        assert response.status_code == 200, "OpenAPI spec must be accessible"

        openapi_spec = response.json()

        # Contract Programming: Validate OpenAPI structure
        assert "openapi" in openapi_spec, "OpenAPI version must be specified"
        assert "info" in openapi_spec, "API info must be present"
        assert "paths" in openapi_spec, "API paths must be present"

        # Validate that endpoints are documented
        paths = openapi_spec["paths"]
        assert len(paths) > 0, "At least one endpoint must be documented"

        # Test Swagger UI accessibility
        response = client.get("/docs")
        assert response.status_code == 200, "Swagger UI must be accessible"

    def test_no_global_app_dependency(self):
        """Test that the app doesn't depend on global variables."""
        # Create multiple app instances to ensure independence
        app1 = create_app()
        app2 = create_app()

        # Contract Programming: Apps must be independent
        assert app1 is not app2, "create_app() must return new instances"

        # Both apps should have the same endpoint structure
        routes1 = {route.path for route in app1.routes}
        routes2 = {route.path for route in app2.routes}

        assert routes1 == routes2, "All app instances must have identical routes"

    def test_dependency_injection_support(self):
        """Test that the app supports dependency injection for testing."""
        from src.refimage.config import Config

        # Create test configuration
        test_config = get_config()
        test_config.database_url = "sqlite:///:memory:"

        # App should accept configuration injection
        app = create_app(config=test_config)

        # Contract Programming: Validate config injection
        assert app is not None, "App must support config injection"

        # Test with test client
        client = TestClient(app)
        response = client.get("/health")

        # Health endpoint should work with injected config
        assert (
            response.status_code == 200
        ), "Health check must work with injected config"


class TestAPIEndpointIntegration:
    """Test that endpoints work after structural refactoring."""

    @pytest.fixture
    def client(self):
        """Create test client with properly structured app."""
        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint functionality."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Contract Programming: Health response validation
        assert "status" in data, "Health response must include status"
        assert data["status"] == "healthy", "Health status must be 'healthy'"

    def test_upload_endpoint_structure(self, client):
        """Test upload endpoint is properly structured."""
        # Test OPTIONS request (CORS preflight)
        response = client.options("/images/upload")

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Upload endpoint must exist"

    def test_search_endpoint_structure(self, client):
        """Test search endpoint is properly structured."""
        response = client.options("/images/search")

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Search endpoint must exist"

    def test_dsl_endpoint_structure(self, client):
        """Test DSL query endpoint is properly structured."""
        response = client.options("/images/dsl")

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "DSL endpoint must exist"


class TestAPIErrorHandling:
    """Test proper error handling in restructured API."""

    @pytest.fixture
    def client(self):
        """Create test client for error testing."""
        app = create_app()
        return TestClient(app)

    def test_404_error_handling(self, client):
        """Test 404 error handling for non-existent endpoints."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

        # Validate error response structure
        data = response.json()
        assert "detail" in data, "404 response must include error detail"

    def test_method_not_allowed_handling(self, client):
        """Test 405 error handling for wrong HTTP methods."""
        # Try POST on GET-only endpoint
        response = client.post("/health")

        assert response.status_code == 405

    def test_internal_error_handling(self, client):
        """Test that internal errors are properly handled."""
        # This will be expanded once we identify error scenarios
        # For now, ensure error responses follow the ErrorResponse schema
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
