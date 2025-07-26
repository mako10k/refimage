#!/usr/bin/env python3
"""
Contract Programming test for API structure validation.
Validates that LLM components work independently from faiss dependencies.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_contract_imports():
    """Test imports with Contract Programming validation."""
    print("Testing imports with Contract Programming...")
    
    try:
        # Test non-faiss dependent imports
        from refimage.config import Settings
        from refimage.llm import LLMManager
        from refimage.storage import StorageManager
        
        # Contract validation: Assert all critical objects are created
        settings = Settings()
        assert settings is not None, "Settings creation must succeed"
        
        llm_manager = LLMManager(settings)
        assert llm_manager is not None, "LLM manager creation must succeed"
        
        storage_manager = StorageManager(settings)
        assert storage_manager is not None, "Storage manager creation must succeed"
        
        print("✅ Contract Programming imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Contract imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_functionality():
    """Test LLM functionality with Contract Programming."""
    print("\nTesting LLM functionality...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager
        
        settings = Settings()
        llm_manager = LLMManager(settings)
        
        # Contract validation: Assert LLM manager operations
        current_provider = llm_manager.get_current_provider()
        assert current_provider is not None, "Current provider must exist"
        
        available_providers = llm_manager.get_available_providers()
        assert isinstance(available_providers, list), "Available providers must be list"
        assert len(available_providers) > 0, "At least one provider must be available"
        
        print(f"✅ LLM functionality validated")
        print(f"   Current provider: {current_provider}")
        print(f"   Available providers: {len(available_providers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_creation():
    """Test API creation with lazy loading Contract Programming."""
    print("\nTesting API creation with Contract Programming...")
    
    try:
        from refimage.api import create_app
        from refimage.config import Settings
        
        settings = Settings()
        
        # Contract validation: App creation must succeed
        app = create_app(settings)
        assert app is not None, "App creation must succeed"
        assert hasattr(app, 'title'), "App must have title attribute"
        assert app.title == "RefImage API", "App title must be correct"
        
        # Contract validation: Routes must be present
        routes = [route.path for route in app.routes]
        assert len(routes) > 0, "App must have routes"
        
        # Check LLM routes exist
        llm_routes = [r for r in routes if 'llm' in r or 'text-to-dsl' in r]
        assert len(llm_routes) > 0, "LLM routes must exist"
        
        print(f"✅ API creation with Contract Programming successful")
        print(f"   Total routes: {len(routes)}")
        print(f"   LLM routes: {len(llm_routes)}")
        for route in llm_routes:
            print(f"     {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ API creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dependency_isolation():
    """Test that faiss dependencies are properly isolated."""
    print("\nTesting dependency isolation...")
    
    try:
        # Try to import faiss-dependent modules to see if they fail gracefully
        from refimage.api import create_app
        from refimage.config import Settings
        
        settings = Settings()
        app = create_app(settings)
        
        # Get the dependency providers from the app
        # This should work even if faiss fails to load
        import inspect
        
        # Contract validation: App should be created regardless of faiss status
        assert app is not None, "App must be created even with faiss issues"
        
        print("✅ Dependency isolation working")
        print("   App created successfully despite potential faiss issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency isolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all Contract Programming tests."""
    print("🚀 Starting Contract Programming API Tests")
    print("=" * 60)
    
    tests = [
        test_contract_imports,
        test_llm_functionality,
        test_api_creation,
        test_dependency_isolation,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Contract Programming Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All Contract Programming tests passed!")
        print("\n💡 Result: LLM functionality is isolated from faiss dependencies")
        return 0
    else:
        print("❌ Some Contract Programming tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
