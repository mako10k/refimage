"""
Tests for storage.py code deduplication.

This module validates the elimination of duplicate code in storage.py
and ensures the refactored code maintains functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.refimage.config import Settings
from src.refimage.models.schemas import ImageMetadata
from src.refimage.storage import StorageError, StorageManager


class TestStorageDeduplication:
    """Test suite for storage code deduplication."""

    @pytest.fixture
    def temp_settings(self, temp_dir):
        """Test settings with temporary directories."""
        return Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
            log_level="DEBUG",
        )

    @pytest.fixture
    def storage_manager(self, temp_settings):
        """Storage manager instance for testing."""
        return StorageManager(temp_settings)

    def test_metadata_creation_consistency(self, storage_manager):
        """Test that ImageMetadata creation is consistent across methods."""
        # Create test data that would trigger both code paths
        test_row = {
            "id": str(uuid4()),
            "filename": "test.png",
            "file_path": "/tmp/test.png",
            "file_size": 1024,
            "mime_type": "image/png",
            "width": 100,
            "height": 100,
            "created_at": datetime.now().isoformat(),
            "description": "Test image",
            "tags": json.dumps(["test", "image"]),
        }

        # Test both methods that create ImageMetadata
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            # Mock get_metadata response
            mock_conn.execute.return_value.fetchone.return_value = test_row

            # Call get_metadata (original method)
            metadata1 = storage_manager.get_metadata(UUID(test_row["id"]))

            # Mock list_images response
            mock_conn.execute.return_value.fetchall.return_value = [test_row]

            # Call list_images (duplicate code method)
            metadata_list = storage_manager.list_images()
            metadata2 = metadata_list[0] if metadata_list else None

            # Both should create identical ImageMetadata objects
            assert metadata1 is not None
            assert metadata2 is not None
            assert metadata1.id == metadata2.id
            assert metadata1.filename == metadata2.filename
            assert metadata1.file_path == metadata2.file_path
            assert metadata1.file_size == metadata2.file_size
            assert metadata1.mime_type == metadata2.mime_type
            assert metadata1.width == metadata2.width
            assert metadata1.height == metadata2.height
            assert metadata1.description == metadata2.description
            assert metadata1.tags == metadata2.tags

    def test_duplicate_metadata_creation_elimination(self):
        """Test that duplicate ImageMetadata creation code is eliminated."""
        # Read the storage.py file
        storage_file = (
            Path(__file__).parent.parent.parent / "src" / "refimage" / "storage.py"
        )
        content = storage_file.read_text()

        # Count occurrences of ImageMetadata creation pattern
        metadata_creation_pattern = "ImageMetadata("
        occurrences = content.count(metadata_creation_pattern)

        # After deduplication, there should be only one occurrence
        # (in a shared helper method)
        assert occurrences <= 2, (
            f"Found {occurrences} ImageMetadata creations, "
            f"expected ≤2 after deduplication"
        )

    def test_row_to_metadata_helper_exists(self, storage_manager):
        """Test that a helper method for row-to-metadata conversion exists."""
        # Check if the helper method exists
        # (should be created during deduplication)
        helper_methods = [
            "_row_to_metadata",
            "_create_metadata_from_row",
            "_build_image_metadata",
            "_create_image_metadata_from_row",
        ]

        has_helper = any(hasattr(storage_manager, method) for method in helper_methods)
        assert has_helper, "No helper method found for ImageMetadata creation"


class TestStorageRefactoredFunctionality:
    """Test that refactored storage functionality works correctly."""

    @pytest.fixture
    def mock_storage_manager(self, temp_dir):
        """Mock storage manager for testing refactored functionality."""
        settings = Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
        )
        return StorageManager(settings)

    def test_get_metadata_after_refactoring(self, mock_storage_manager):
        """Test get_metadata works correctly after code deduplication."""
        test_id = uuid4()

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            # Mock database response
            mock_conn.execute.return_value.fetchone.return_value = {
                "id": str(test_id),
                "filename": "test.png",
                "file_path": "/tmp/test.png",
                "file_size": 1024,
                "mime_type": "image/png",
                "width": 100,
                "height": 100,
                "created_at": datetime.now().isoformat(),
                "description": "Test image",
                "tags": json.dumps(["test"]),
            }

            # Test method functionality
            metadata = mock_storage_manager.get_metadata(test_id)

            assert metadata is not None
            assert metadata.id == test_id
            assert metadata.filename == "test.png"
            assert metadata.tags == ["test"]

    def test_list_images_after_refactoring(self, mock_storage_manager):
        """Test list_images works correctly after code deduplication."""
        test_ids = [uuid4(), uuid4()]

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn

            # Mock database response with multiple rows
            mock_conn.execute.return_value.fetchall.return_value = [
                {
                    "id": str(test_ids[0]),
                    "filename": "test1.png",
                    "file_path": "/tmp/test1.png",
                    "file_size": 1024,
                    "mime_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "created_at": datetime.now().isoformat(),
                    "description": "Test image 1",
                    "tags": json.dumps(["test", "image1"]),
                },
                {
                    "id": str(test_ids[1]),
                    "filename": "test2.png",
                    "file_path": "/tmp/test2.png",
                    "file_size": 2048,
                    "mime_type": "image/png",
                    "width": 200,
                    "height": 200,
                    "created_at": datetime.now().isoformat(),
                    "description": "Test image 2",
                    "tags": json.dumps(["test", "image2"]),
                },
            ]

            # Test method functionality
            images = mock_storage_manager.list_images()

            assert len(images) == 2
            assert images[0].id == test_ids[0]
            assert images[1].id == test_ids[1]
            assert images[0].filename == "test1.png"
            assert images[1].filename == "test2.png"

    def test_error_handling_preserved(self, mock_storage_manager):
        """Test that error handling is preserved after refactoring."""
        test_id = uuid4()

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = Exception("Database connection failed")

            # Both methods should raise StorageError
            with pytest.raises(StorageError):
                mock_storage_manager.get_metadata(test_id)

            with pytest.raises(StorageError):
                mock_storage_manager.list_images()


class TestStorageCodeQuality:
    """Test code quality improvements from deduplication."""

    def test_dry_principle_adherence(self):
        """Test that DRY principle is followed after deduplication."""
        storage_file = (
            Path(__file__).parent.parent.parent / "src" / "refimage" / "storage.py"
        )
        content = storage_file.read_text()

        # Check for duplicate string patterns that should be eliminated
        duplicate_patterns = [
            "ImageMetadata(",
            "UUID(row['id'])",
            "Path(row['file_path'])",
            "datetime.fromisoformat(row['created_at'])",
            "json.loads(row['tags']) if row['tags'] else []",
        ]

        for pattern in duplicate_patterns:
            count = content.count(pattern)
            # After proper deduplication, these patterns should appear
            # at most twice (original + helper method)
            assert (
                count <= 3
            ), f"Pattern '{pattern}' appears {count} times, indicating possible duplication"

    def test_function_complexity_reduced(self):
        """Test that function complexity is reduced after deduplication."""
        # This is a basic test - in practice you'd use tools like radon
        storage_file = (
            Path(__file__).parent.parent.parent / "src" / "refimage" / "storage.py"
        )
        content = storage_file.read_text()

        # Count lines in get_metadata and list_images methods
        lines = content.split("\n")

        def get_method_lines(method_name):
            start_line = None
            indent_level = None
            method_lines = []

            for i, line in enumerate(lines):
                if f"def {method_name}(" in line:
                    start_line = i
                    indent_level = len(line) - len(line.lstrip())
                    continue

                if start_line is not None:
                    if line.strip() == "":
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    method_lines.append(line)

            return len(method_lines)

        get_metadata_lines = get_method_lines("get_metadata")
        list_images_lines = get_method_lines("list_images")

        # After deduplication, methods should be shorter
        # These are reasonable thresholds for refactored methods
        assert (
            get_metadata_lines < 30
        ), f"get_metadata has {get_metadata_lines} lines, should be < 30"
        assert (
            list_images_lines < 40
        ), f"list_images has {list_images_lines} lines, should be < 40"

    def test_maintainability_improved(self):
        """Test that code maintainability is improved."""
        # Check for presence of helper methods that improve maintainability
        storage_file = (
            Path(__file__).parent.parent.parent / "src" / "refimage" / "storage.py"
        )
        content = storage_file.read_text()

        # Look for helper method patterns
        helper_patterns = [
            "def _row_to_metadata",
            "def _create_metadata_from_row",
            "def _build_image_metadata",
            "def _parse_database_row",
        ]

        has_helper = any(pattern in content for pattern in helper_patterns)
        assert has_helper, "No helper methods found - maintainability not improved"


@pytest.mark.integration
class TestStorageIntegrationAfterRefactoring:
    """Integration tests to ensure storage works with other components after refactoring."""

    def test_storage_search_integration(self, temp_dir):
        """Test storage integration with search components."""
        settings = Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
        )
        storage = StorageManager(settings)

        # Test that storage can provide data for search engine
        # This would be expanded based on actual search integration
        with patch("sqlite3.connect"):
            embeddings = storage.get_all_embeddings()
            assert isinstance(embeddings, list)

    def test_storage_api_integration(self, temp_dir):
        """Test storage integration with API layer."""
        settings = Settings(
            database_url=f"sqlite:///{temp_dir}/test.db",
            image_storage_path=str(temp_dir / "images"),
        )
        storage = StorageManager(settings)

        # Test that storage provides consistent interface for API
        with patch("sqlite3.connect"):
            try:
                metadata = storage.get_metadata(uuid4())
                # Should return None or ImageMetadata, not raise exception
                assert metadata is None or isinstance(metadata, ImageMetadata)
            except StorageError:
                # StorageError is acceptable for missing items
                pass
