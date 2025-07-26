"""
Simple tests for RefImage components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.refimage.config import Settings
from src.refimage.models.schemas import ImageMetadata, DSLQuery, DSLResponse


def test_settings():
    """Test settings initialization."""
    settings = Settings()
    assert settings is not None
    assert settings.clip_model_name is not None
    assert settings.server_host is not None
    print("✅ Settings test passed")


def test_image_metadata():
    """Test ImageMetadata model."""
    from uuid import uuid4
    from pathlib import Path
    
    metadata = ImageMetadata(
        filename="test.jpg",
        file_path=Path("/tmp/test.jpg"),
        file_size=1024,
        mime_type="image/jpeg",
        width=100,
        height=100,
        tags=["test", "image"],
        description="Test image"
    )
    
    assert metadata.filename == "test.jpg"
    assert metadata.file_size == 1024
    assert "test" in metadata.tags
    print("✅ ImageMetadata test passed")


def test_dsl_query():
    """Test DSL query model."""
    query = DSLQuery(
        query="red car AND #sports",
        limit=20,
        threshold=0.5
    )
    
    assert query.query == "red car AND #sports"
    assert query.limit == 20
    assert query.threshold == 0.5
    print("✅ DSL Query test passed")


def test_dsl_response():
    """Test DSL response model."""
    from pathlib import Path
    
    metadata = ImageMetadata(
        filename="car.jpg",
        file_path=Path("/tmp/car.jpg"),
        file_size=2048,
        mime_type="image/jpeg",
        width=200,
        height=150,
        description="Red sports car"
    )
    
    response = DSLResponse(
        query="red car",
        results=[metadata],
        total_count=1,
        query_info={"type": "dsl", "threshold": 0.3}
    )
    
    assert response.query == "red car"
    assert len(response.results) == 1
    assert response.total_count == 1
    print("✅ DSL Response test passed")


if __name__ == "__main__":
    print("🧪 Running RefImage tests...")
    
    try:
        test_settings()
        test_image_metadata()
        test_dsl_query()
        test_dsl_response()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
