"""
Simple storage test with real image data.
"""

import sys
import os
import tempfile
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.refimage.config import Settings
from src.refimage.storage import StorageManager
from pathlib import Path
from PIL import Image


def create_test_image() -> bytes:
    """Create a simple test image."""
    # Create a small RGB image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_basic_storage():
    """Test basic storage functionality."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(
            image_storage_path=Path(tmp_dir) / "images",
            metadata_storage_path=Path(tmp_dir) / "metadata.db",
        )
        
        storage = StorageManager(settings)
        print("✅ Storage manager created")
        
        # Create test image
        image_data = create_test_image()
        print(f"✅ Test image created: {len(image_data)} bytes")
        
        # Store image
        metadata = storage.store_image(
            image_data=image_data,
            filename="test.jpg",
            description="Test image",
            tags=["test", "red"]
        )
        
        print(f"✅ Image stored: {metadata.filename}")
        print(f"   ID: {metadata.id}")
        print(f"   Size: {metadata.file_size} bytes")
        print(f"   Dimensions: {metadata.width}x{metadata.height}")
        print(f"   Tags: {metadata.tags}")
        
        # Test retrieval
        retrieved = storage.get_metadata(metadata.id)
        assert retrieved is not None
        assert retrieved.filename == "test.jpg"
        print("✅ Metadata retrieval successful")
        
        # Test listing
        images = storage.list_images()
        assert len(images) == 1
        print("✅ Image listing successful")
        
        # Test stats
        stats = storage.get_storage_stats()
        print(f"✅ Storage stats: {stats}")
        
        return True


if __name__ == "__main__":
    print("🧪 Testing basic storage functionality...")
    
    try:
        if test_basic_storage():
            print("\n🎉 Basic storage test passed!")
        else:
            print("\n❌ Storage test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
