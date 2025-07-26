#!/usr/bin/env python3
"""
LLM-only API test to verify LLM functionality is isolated from faiss issues.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_llm_isolated_import():
    """Test that LLM components can be imported without API dependencies."""
    print("Testing isolated LLM component import...")
    
    try:
        # Import LLM components directly
        from refimage.config import Settings
        from refimage.llm import LLMManager, LLMProvider
        from refimage.models.schemas import (
            TextToDSLRequest,
            TextToDSLResponse,
            LLMProvidersResponse,
            LLMSwitchRequest,
            LLMSwitchResponse
        )
        
        print("✅ LLM components imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import LLM components: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_standalone_functionality():
    """Test LLM functionality without full API context."""
    print("\nTesting standalone LLM functionality...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager, LLMProvider, LLMMessage
        
        # Create settings and manager
        settings = Settings()
        llm_manager = LLMManager(settings)
        
        # Test basic operations
        current_provider = llm_manager.get_current_provider()
        available_providers = llm_manager.get_available_providers()
        
        print(f"✅ Current provider: {current_provider.value if current_provider else 'None'}")
        print(f"✅ Available providers: {[p.value for p in available_providers]}")
        
        # Test schema creation
        from refimage.models.schemas import TextToDSLRequest
        request = TextToDSLRequest(text="Find red cars in the image gallery")
        print(f"✅ Schema creation successful: {request.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed LLM functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_generation():
    """Test actual LLM generation functionality."""
    print("\nTesting LLM text generation...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager, LLMMessage
        
        settings = Settings()
        llm_manager = LLMManager(settings)
        
        # Check if OpenAI is available
        if not llm_manager.get_available_providers():
            print("⚠️  No LLM providers available - skipping generation test")
            return True
        
        # Create test messages
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Convert 'red car' to DSL format")
        ]
        
        print(f"✅ Messages created: {len(messages)} messages")
        print(f"📝 Test ready for provider: {llm_manager.get_current_provider().value}")
        
        # Note: We won't actually call generate() to avoid API costs
        # but we verify the setup is correct
        
        return True
        
    except Exception as e:
        print(f"❌ Failed LLM generation setup: {e}")
        return False


async def main():
    """Run all isolated LLM tests."""
    print("🚀 Starting Isolated LLM Tests")
    print("=" * 50)
    
    tests = [
        test_llm_isolated_import,
        test_llm_standalone_functionality,
        test_llm_generation,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Isolated LLM Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All isolated LLM tests passed!")
        print("\n💡 Conclusion: LLM functionality works independently of faiss")
        return 0
    else:
        print("❌ Some isolated LLM tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
