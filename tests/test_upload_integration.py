"""
Comprehensive upload functionality integration tests.

Tests the complete upload pipeline including:
- File validation
- Metadata creation
- Embedding generation
- Error handling
"""

import io
import random
import tempfile
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.refimage.api import create_app
from src.refimage.config import Settings


class TestUploadIntegration:
    """Upload integration test suite."""

    @pytest.fixture
    def test_settings(self):
        """Create test-specific settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            return Settings(
                image_storage_path=str(test_dir / "images"),
                database_path=str(test_dir / "test.db"),
                clip_model_name="ViT-B/32",
                device="cpu"
            )

    @pytest.fixture
    def client(self, test_settings):
        """Create test client with isolated database."""
        app = create_app(test_settings)
        return TestClient(app)

    @pytest.fixture
    def test_image(self):
        """Create unique test image for each test."""
        # Generate unique color and size to avoid hash collisions
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        size = (
            random.randint(50, 150),
            random.randint(50, 150)
        )
        
        img = Image.new('RGB', size, color=color)
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

    # ========================================
    # P0: 基本機能の実利用シナリオテスト
    # ========================================
    
    def test_upload_multiple_images(self, client):
        """Test multiple image upload in sequence."""
        images = []
        upload_results = []
        
        # Create multiple unique test images
        colors = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
        ]
        
        for i in range(3):
            # Create unique image with different color and size
            color = colors[i]
            size = (100 + i*10, 100 + i*10)  # Different sizes to ensure uniqueness
            
            img = Image.new('RGB', size, color=color)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95 - i)  # Different quality
            img_bytes.seek(0)
            images.append(img_bytes)
        
        # Upload multiple images
        for i, img_bytes in enumerate(images):
            response = client.post(
                '/images',
                files={'file': (f'test_{i}.jpg', img_bytes, 'image/jpeg')},
                data={'description': f'Test image {i}', 'tags': f'test,multi-upload,image-{i}'}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['upload_success'] is True
            upload_results.append(result)
        
        # Verify all images have unique IDs
        image_ids = [result['image_id'] for result in upload_results]
        assert len(set(image_ids)) == 3, "All uploaded images should have unique IDs"
        
        # Verify metadata for each image
        for i, result in enumerate(upload_results):
            metadata = result['metadata']
            assert metadata['filename'] == f'test_{i}.jpg'
            assert metadata['description'] == f'Test image {i}'
            assert f'image-{i}' in metadata['tags']

    def test_upload_immediate_retrieval(self, client, test_image):
        """Test that uploaded image can be immediately retrieved."""
        # Upload image
        upload_response = client.post(
            '/images',
            files={'file': ('test_retrieval.jpg', test_image, 'image/jpeg')},
            data={'description': 'Immediate retrieval test', 'tags': 'test,retrieval'}
        )
        
        assert upload_response.status_code == 200
        upload_result = upload_response.json()
        image_id = upload_result['image_id']
        
        # Immediately try to list images and find the uploaded one
        list_response = client.get('/images')
        assert list_response.status_code == 200
        
        list_result = list_response.json()
        uploaded_image = None
        for image in list_result['images']:
            if image['id'] == image_id:
                uploaded_image = image
                break
        
        assert uploaded_image is not None, "Uploaded image should be immediately retrievable"
        assert uploaded_image['filename'] == 'test_retrieval.jpg'
        assert uploaded_image['description'] == 'Immediate retrieval test'
        assert 'retrieval' in uploaded_image['tags']

    def test_upload_unicode_metadata(self, client, test_image):
        """Test upload with Unicode characters in metadata."""
        unicode_description = "テスト画像 🔥 Special chars: ñöŕmäl tèxt"
        unicode_tags = "日本語,絵文字😀,特殊文字ñöŕmäl,テスト"
        
        response = client.post(
            '/images',
            files={'file': ('unicode_test.jpg', test_image, 'image/jpeg')},
            data={'description': unicode_description, 'tags': unicode_tags}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['upload_success'] is True
        
        # Verify Unicode metadata is preserved
        metadata = result['metadata']
        assert metadata['description'] == unicode_description
        
        expected_tags = ['日本語', '絵文字😀', '特殊文字ñöŕmäl', 'テスト']
        for tag in expected_tags:
            assert tag in metadata['tags'], f"Tag '{tag}' should be preserved"

    def test_upload_duplicate_image(self, client):
        """Test behavior when uploading the same image multiple times."""
        # Create test image
        img = Image.new('RGB', (50, 50), color='purple')
        img_bytes_1 = io.BytesIO()
        img.save(img_bytes_1, format='JPEG')
        img_bytes_1.seek(0)
        
        img_bytes_2 = io.BytesIO()
        img.save(img_bytes_2, format='JPEG')
        img_bytes_2.seek(0)
        
        # First upload
        response1 = client.post(
            '/images',
            files={'file': ('duplicate_test.jpg', img_bytes_1, 'image/jpeg')},
            data={'description': 'First upload', 'tags': 'duplicate,test,first'}
        )
        
        assert response1.status_code == 200
        result1 = response1.json()
        first_image_id = result1['image_id']
        
        # Second upload (same content, different metadata)
        response2 = client.post(
            '/images',
            files={'file': ('duplicate_test.jpg', img_bytes_2, 'image/jpeg')},
            data={'description': 'Second upload', 'tags': 'duplicate,test,second'}
        )
        
        # Expect 409 Conflict for duplicate image
        assert response2.status_code == 409
        error_detail = response2.json()
        assert 'detail' in error_detail
        assert 'Duplicate image detected' in error_detail['detail']

    # ========================================
    # P1: 境界値・異常系テスト
    # ========================================
    
    def test_upload_boundary_image_sizes(self, client):
        """Test upload with boundary image sizes."""
        # Test minimum size (1x1 pixel)
        min_img = Image.new('RGB', (1, 1), color='black')
        min_img_bytes = io.BytesIO()
        min_img.save(min_img_bytes, format='JPEG')
        min_img_bytes.seek(0)
        
        response = client.post(
            '/images',
            files={'file': ('min_size.jpg', min_img_bytes, 'image/jpeg')},
            data={'description': 'Minimum size test'}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['upload_success'] is True
        assert result['metadata']['width'] == 1
        assert result['metadata']['height'] == 1
        
        # Test large size (within reasonable limits)
        large_img = Image.new('RGB', (2000, 2000), color='white')
        large_img_bytes = io.BytesIO()
        large_img.save(large_img_bytes, format='JPEG', quality=85)
        large_img_bytes.seek(0)
        
        response = client.post(
            '/images',
            files={'file': ('large_size.jpg', large_img_bytes, 'image/jpeg')},
            data={'description': 'Large size test'}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['upload_success'] is True
        assert result['metadata']['width'] == 2000
        assert result['metadata']['height'] == 2000

    def test_upload_boundary_metadata(self, client, test_image):
        """Test upload with boundary metadata values."""
        # Test empty description (should be allowed)
        response = client.post(
            '/images',
            files={'file': ('empty_desc.jpg', test_image, 'image/jpeg')},
            data={'description': '', 'tags': 'boundary,test'}
        )
        
        assert response.status_code == 200
        result = response.json()
        # Empty description should be converted to None or empty string
        assert result['metadata']['description'] in [None, '']
        
        # Test very long description
        long_description = 'A' * 1000  # 1000 characters
        test_image.seek(0)
        response = client.post(
            '/images',
            files={'file': ('long_desc.jpg', test_image, 'image/jpeg')},
            data={'description': long_description, 'tags': 'boundary,test,long'}
        )
        
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            result = response.json()
            assert len(result['metadata']['description']) <= 1000

    def test_upload_unsupported_format(self, client):
        """Test upload with unsupported file formats."""
        # Create a text file disguised as image
        fake_image = io.BytesIO(b"This is not an image file")
        
        response = client.post(
            '/images',
            files={'file': ('fake.jpg', fake_image, 'image/jpeg')},
            data={'description': 'Fake image test'}
        )
        
        # Should return appropriate error
        assert response.status_code in [400, 415, 422, 500]
        
        # Test actual unsupported format (if any)
        # Create a simple BMP that might not be supported
        try:
            bmp_img = Image.new('RGB', (10, 10), color='red')
            bmp_bytes = io.BytesIO()
            bmp_img.save(bmp_bytes, format='BMP')
            bmp_bytes.seek(0)
            
            response = client.post(
                '/images',
                files={'file': ('test.bmp', bmp_bytes, 'image/bmp')},
                data={'description': 'BMP format test'}
            )
            
            # Result depends on implementation - document current behavior
            if response.status_code == 200:
                # BMP is supported
                result = response.json()
                assert result['upload_success'] is True
            else:
                # BMP is not supported - should return appropriate error
                assert response.status_code in [400, 415, 422]
        except Exception:
            # BMP creation failed, skip this part
            pass

    def test_upload_corrupted_image(self, client):
        """Test upload with corrupted image data."""
        # Create partially corrupted JPEG data
        valid_img = Image.new('RGB', (50, 50), color='blue')
        img_bytes = io.BytesIO()
        valid_img.save(img_bytes, format='JPEG')
        
        # Corrupt the data by truncating it
        corrupted_data = img_bytes.getvalue()[:len(img_bytes.getvalue())//2]
        corrupted_bytes = io.BytesIO(corrupted_data)
        
        response = client.post(
            '/images',
            files={'file': ('corrupted.jpg', corrupted_bytes, 'image/jpeg')},
            data={'description': 'Corrupted image test'}
        )
        
        # Should return appropriate error
        assert response.status_code in [400, 422, 500]

    def test_upload_maximum_tags(self, client, test_image):
        """Test upload with many tags."""
        # Test with many tags
        many_tags = ','.join([f'tag_{i}' for i in range(50)])
        
        response = client.post(
            '/images',
            files={'file': ('many_tags.jpg', test_image, 'image/jpeg')},
            data={'description': 'Many tags test', 'tags': many_tags}
        )
        
        # Should either succeed or return appropriate error for too many tags
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            result = response.json()
            # System should handle large number of tags appropriately
            assert len(result['metadata']['tags']) <= 50
