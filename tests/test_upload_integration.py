"""
Comprehensive upload functionality integration tests.

Tests the complete upload pipeline including:
- File validation
- Metadata creation
- Embedding generation
- Error handling
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.refimage.api import create_app


class TestUploadIntegration:
    """Upload integration test suite."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes

    def test_upload_basic_functionality(self, client, test_image):
        """Test basic upload functionality."""
        response = client.post(
            '/images',
            files={'file': ('test.jpg', test_image, 'image/jpeg')},
            data={'description': 'Test image', 'tags': 'test,upload'}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert 'image_id' in result
        assert 'upload_success' in result
        assert 'embedding_generated' in result
        assert 'metadata' in result
        
        # Verify upload success
        assert result['upload_success'] is True
        assert result['embedding_generated'] is True
        
        # Verify metadata
        metadata = result['metadata']
        assert metadata['filename'] == 'test.jpg'
        assert metadata['description'] == 'Test image'
        assert 'test' in metadata['tags']
        assert 'upload' in metadata['tags']
        
    def test_upload_large_image(self, client):
        """Test upload with larger image."""
        img = Image.new('RGB', (1000, 1000), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            '/images',
            files={'file': ('large_test.jpg', img_bytes, 'image/jpeg')},
            data={'description': 'Large test image'}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['upload_success'] is True
        
    def test_upload_different_formats(self, client):
        """Test upload with different image formats."""
        formats = ['JPEG', 'PNG']
        
        for fmt in formats:
            img = Image.new('RGB', (50, 50), color='green')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)
            
            ext = fmt.lower()
            response = client.post(
                '/images',
                files={'file': (f'test.{ext}', img_bytes, f'image/{ext}')},
                data={'description': f'{fmt} test image'}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['upload_success'] is True
            
    def test_upload_without_optional_fields(self, client, test_image):
        """Test upload without description and tags."""
        response = client.post(
            '/images',
            files={'file': ('minimal.jpg', test_image, 'image/jpeg')},
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['upload_success'] is True
        
        metadata = result['metadata']
        assert metadata['filename'] == 'minimal.jpg'
        assert metadata['description'] is None
        assert metadata['tags'] == []
        
    def test_upload_empty_tags(self, client, test_image):
        """Test upload with empty tags string."""
        response = client.post(
            '/images',
            files={'file': ('empty_tags.jpg', test_image, 'image/jpeg')},
            data={'tags': ''}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['metadata']['tags'] == []
        
    def test_upload_complex_tags(self, client, test_image):
        """Test upload with complex tags."""
        response = client.post(
            '/images',
            files={'file': ('complex_tags.jpg', test_image, 'image/jpeg')},
            data={'tags': ' tag1 , tag2, tag3 , ,tag4'}  # Test whitespace and empty tags
        )
        
        assert response.status_code == 200
        result = response.json()
        tags = result['metadata']['tags']
        
        # Should clean and deduplicate tags
        expected_tags = ['tag1', 'tag2', 'tag3', 'tag4']
        for tag in expected_tags:
            assert tag in tags
            
    def test_upload_error_handling_invalid_file(self, client):
        """Test error handling for invalid file."""
        # Test with non-image content
        invalid_content = io.BytesIO(b"Not an image")
        
        response = client.post(
            '/images',
            files={'file': ('invalid.jpg', invalid_content, 'image/jpeg')},
            data={'description': 'Invalid file test'}
        )
        
        # Should return error (exact status depends on implementation)
        assert response.status_code in [400, 500]
        
    def test_upload_missing_file(self, client):
        """Test error handling for missing file."""
        response = client.post(
            '/images',
            data={'description': 'No file test'}
        )
        
        # Should return validation error
        assert response.status_code == 422
