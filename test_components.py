"""
Test API without CLIP dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.refimage.config import get_settings


def test_api_imports():
    """Test API module imports."""
    try:
        from src.refimage.models.schemas import ImageMetadata, DSLQuery
        print("✅ Schema imports successful")
        
        from src.refimage.storage import StorageManager
        print("✅ Storage manager import successful")
        
        from src.refimage.search import VectorSearchEngine
        print("✅ Search engine import successful")
        
        from src.refimage.dsl import DSLParser
        print("✅ DSL parser import successful")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True


def test_components_creation():
    """Test component creation."""
    try:
        settings = get_settings()
        print("✅ Settings creation successful")
        
        # Test storage manager
        from src.refimage.storage import StorageManager
        storage = StorageManager(settings)
        stats = storage.get_storage_stats()
        print(f"✅ Storage manager creation successful: {stats}")
        
        # Test search engine
        from src.refimage.search import VectorSearchEngine
        search = VectorSearchEngine(settings)
        search_stats = search.get_stats()
        print(f"✅ Search engine creation successful: {search_stats}")
        
        # Test DSL parser
        from src.refimage.dsl import DSLParser
        parser = DSLParser()
        test_query = parser.parse("red car")
        print(f"✅ DSL parser creation successful: {type(test_query).__name__}")
        
    except Exception as e:
        print(f"❌ Component creation error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🧪 Testing RefImage components...")
    
    success = True
    success &= test_api_imports()
    success &= test_components_creation()
    
    if success:
        print("\n🎉 All component tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
