"""
Test storage and search engine functionality.
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.refimage.config import Settings
from src.refimage.storage import StorageManager
from src.refimage.search import VectorSearchEngine
from src.refimage.models.schemas import ImageEmbedding
from uuid import uuid4
from pathlib import Path
import numpy as np


def test_storage_manager():
    """Test storage manager functionality."""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test settings
        settings = Settings(
            image_storage_path=Path(tmp_dir) / "images",
            metadata_storage_path=Path(tmp_dir) / "metadata.db",
            index_storage_path=Path(tmp_dir) / "index"
        )
        
        # Initialize storage manager
        storage = StorageManager(settings)
        print("✅ Storage manager initialization successful")
        
        # Test storage stats
        stats = storage.get_storage_stats()
        assert stats['total_images'] == 0
        print(f"✅ Initial storage stats: {stats}")
        
        # Create test image data
        test_image_data = b"fake_image_data_for_testing"
        
        # Test image storage
        metadata = storage.store_image(
            image_data=test_image_data,
            filename="test_image.jpg",
            description="Test image for unit testing",
            tags=["test", "unit", "sample"]
        )
        
        assert metadata.filename == "test_image.jpg"
        assert "test" in metadata.tags
        print(f"✅ Image storage successful: {metadata.id}")
        
        # Test metadata retrieval
        retrieved = storage.get_image_metadata(metadata.id)
        assert retrieved is not None
        assert retrieved.id == metadata.id
        print("✅ Metadata retrieval successful")
        
        # Test image listing
        images = storage.list_images(limit=10)
        assert len(images) == 1
        assert images[0].id == metadata.id
        print("✅ Image listing successful")
        
        # Test tag filtering
        filtered = storage.list_images(tags_filter=["test"])
        assert len(filtered) == 1
        print("✅ Tag filtering successful")
        
        # Test embedding storage
        test_embedding = ImageEmbedding(
            image_id=metadata.id,
            embedding=np.random.random(512).tolist(),
            model_name="test-model"
        )
        
        storage.store_embedding(test_embedding)
        print("✅ Embedding storage successful")
        
        # Test embedding retrieval
        retrieved_embedding = storage.get_embedding(metadata.id)
        assert retrieved_embedding is not None
        assert len(retrieved_embedding.embedding) == 512
        print("✅ Embedding retrieval successful")
        
        return True


def test_search_engine():
    """Test vector search engine functionality."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(
            index_storage_path=Path(tmp_dir) / "index"
        )
        
        # Initialize search engine
        search = VectorSearchEngine(settings)
        print("✅ Search engine initialization successful")
        
        # Test initial stats
        stats = search.get_stats()
        assert stats['total_embeddings'] == 0
        print(f"✅ Initial search stats: {stats}")
        
        # Create test embeddings
        embeddings = []
        for i in range(5):
            embedding = ImageEmbedding(
                image_id=uuid4(),
                embedding=np.random.random(512).tolist(),
                model_name="test-model"
            )
            embeddings.append(embedding)
        
        # Test batch embedding addition
        search.add_embeddings_batch(embeddings)
        print(f"✅ Batch embedding addition successful: {len(embeddings)} embeddings")
        
        # Test search stats after addition
        stats = search.get_stats()
        assert stats['total_embeddings'] == 5
        print(f"✅ Updated search stats: {stats}")
        
        # Test search functionality
        query_embedding = np.random.random(512)
        results = search.search(query_embedding, k=3, threshold=0.0)
        
        assert len(results) <= 3
        assert len(results) > 0
        print(f"✅ Search successful: found {len(results)} results")
        
        # Test search result format
        for image_id, score in results:
            assert isinstance(image_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        print("✅ Search result format validation successful")
        
        return True


def test_integration():
    """Test storage and search integration."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(
            image_storage_path=Path(tmp_dir) / "images",
            metadata_storage_path=Path(tmp_dir) / "metadata.db",
            index_storage_path=Path(tmp_dir) / "index"
        )
        
        # Initialize components
        storage = StorageManager(settings)
        search = VectorSearchEngine(settings)
        print("✅ Components initialization successful")
        
        # Store test images with embeddings
        for i in range(3):
            # Store image
            test_data = f"fake_image_data_{i}".encode()
            metadata = storage.store_image(
                image_data=test_data,
                filename=f"test_{i}.jpg",
                description=f"Test image {i}",
                tags=[f"test_{i}", "integration"]
            )
            
            # Create and store embedding
            embedding = ImageEmbedding(
                image_id=metadata.id,
                embedding=np.random.random(512).tolist(),
                model_name="test-model"
            )
            
            storage.store_embedding(embedding)
            search.add_embedding(embedding)
            
            print(f"✅ Stored image {i} with embedding")
        
        # Test integrated search
        query_embedding = np.random.random(512)
        search_results = search.search(query_embedding, k=5)
        
        # Verify we can get metadata for search results
        for image_id, score in search_results:
            metadata = storage.get_image_metadata(image_id)
            assert metadata is not None
            print(f"✅ Found metadata for search result: {metadata.filename}")
        
        print("✅ Integration test successful")
        return True


if __name__ == "__main__":
    print("🧪 Testing Storage and Search functionality...")
    
    success = True
    
    try:
        success &= test_storage_manager()
        success &= test_search_engine()
        success &= test_integration()
        
        if success:
            print("\n🎉 All storage and search tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)
