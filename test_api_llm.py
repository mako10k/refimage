#!/usr/bin/env python3
"""
API endpoint test for LLM integration.
"""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_api_import():
    """Test that API module can be imported."""
    print("Testing API module import...")
    
    try:
        # Mock dependencies to avoid faiss issues
        sys.modules['faiss'] = MagicMock()
        
        from refimage.api import create_app
        from refimage.config import Settings
        
        settings = Settings()
        app = create_app(settings)
        
        print("✅ API app created successfully")
        print(f"📝 App title: {app.title}")
        print(f"🔗 Docs URL: {app.docs_url}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import API: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_endpoints_exist():
    """Test that LLM endpoints are present in the API."""
    print("\nTesting LLM endpoints...")
    
    try:
        # Mock dependencies
        sys.modules['faiss'] = MagicMock()
        
        from refimage.api import create_app
        from refimage.config import Settings
        
        settings = Settings()
        app = create_app(settings)
        
        # Check routes
        routes = [route.path for route in app.routes]
        
        expected_llm_routes = [
            "/conversions/text-to-dsl",
            "/llm/providers",
            "/llm/switch"
        ]
        
        found_routes = []
        missing_routes = []
        
        for expected in expected_llm_routes:
            if expected in routes:
                found_routes.append(expected)
                print(f"✅ Found route: {expected}")
            else:
                missing_routes.append(expected)
                print(f"❌ Missing route: {expected}")
        
        print(f"📊 LLM routes found: {len(found_routes)}/{len(expected_llm_routes)}")
        
        if missing_routes:
            print("📋 All available routes:")
            for route in sorted(routes):
                if route.startswith('/'):
                    print(f"   {route}")
        
        return len(missing_routes) == 0
        
    except Exception as e:
        print(f"❌ Failed to test endpoints: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_openapi_schema():
    """Test OpenAPI schema generation."""
    print("\nTesting OpenAPI schema...")
    
    try:
        # Mock dependencies
        sys.modules['faiss'] = MagicMock()
        
        from refimage.api import create_app
        from refimage.config import Settings
        
        settings = Settings()
        app = create_app(settings)
        
        # Get OpenAPI schema
        schema = app.openapi()
        
        print(f"✅ OpenAPI schema generated")
        print(f"📝 API title: {schema.get('info', {}).get('title', 'Unknown')}")
        print(f"🔢 API version: {schema.get('info', {}).get('version', 'Unknown')}")
        
        # Check for LLM-related paths
        paths = schema.get('paths', {})
        llm_paths = [path for path in paths.keys() if 'llm' in path or 'text-to-dsl' in path]
        
        print(f"🎯 LLM-related paths: {len(llm_paths)}")
        for path in llm_paths:
            print(f"   {path}")
        
        return len(llm_paths) > 0
        
    except Exception as e:
        print(f"❌ Failed to test OpenAPI schema: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dependency_injection():
    """Test dependency injection for LLM components."""
    print("\nTesting dependency injection...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager
        
        settings = Settings()
        llm_manager = LLMManager(settings)
        
        print("✅ LLM manager created for dependency injection")
        print(f"🎯 Current provider: {llm_manager.get_current_provider()}")
        print(f"📋 Available providers: {len(llm_manager.get_available_providers())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test dependency injection: {e}")
        return False


async def main():
    """Run all API tests."""
    print("🚀 Starting API LLM Integration Tests")
    print("=" * 50)
    
    tests = [
        test_api_import,
        test_llm_endpoints_exist,
        test_openapi_schema,
        test_dependency_injection,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All API LLM tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
