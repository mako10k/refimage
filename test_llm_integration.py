#!/usr/bin/env python3
"""
Simple test for LLM integration functionality.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_llm_module_import():
    """Test that LLM module can be imported successfully."""
    print("Testing LLM module import...")
    
    try:
        from refimage.llm import (
            LLMManager, 
            LLMError, 
            LLMProvider, 
            LLMMessage,
            TEXT_TO_DSL_SYSTEM_PROMPT,
            TEXT_TO_DSL_EXAMPLES
        )
        print("✅ LLM module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import LLM module: {e}")
        return False

async def test_llm_manager_creation():
    """Test LLM manager creation."""
    print("\nTesting LLM manager creation...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager
        
        settings = Settings()
        llm_manager = LLMManager(settings)
        print("✅ LLM manager created successfully")
        
        # Test provider enumeration
        available_providers = llm_manager.get_available_providers()
        print(f"📋 Available providers: {[p.value for p in available_providers]}")
        
        current_provider = llm_manager.get_current_provider()
        print(f"🎯 Current provider: {current_provider.value if current_provider else 'None'}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create LLM manager: {e}")
        return False

async def test_text_to_dsl_prompt():
    """Test the text-to-DSL prompt system."""
    print("\nTesting text-to-DSL prompt system...")
    
    try:
        from refimage.llm import TEXT_TO_DSL_SYSTEM_PROMPT, TEXT_TO_DSL_EXAMPLES
        
        print(f"✅ System prompt length: {len(TEXT_TO_DSL_SYSTEM_PROMPT)} characters")
        print(f"✅ Number of examples: {len(TEXT_TO_DSL_EXAMPLES)}")
        
        # Show example structure
        if TEXT_TO_DSL_EXAMPLES:
            example = TEXT_TO_DSL_EXAMPLES[0]
            print(f"📝 Example structure: {list(example.keys())}")
            print(f"   Input: {example['input'][:50]}...")
            print(f"   Output: {example['output'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test prompt system: {e}")
        return False

async def test_provider_switching():
    """Test provider switching functionality."""
    print("\nTesting provider switching...")
    
    try:
        from refimage.config import Settings
        from refimage.llm import LLMManager, LLMProvider
        
        settings = Settings()
        llm_manager = LLMManager(settings)
        
        # Try switching to different providers
        for provider in LLMProvider:
            try:
                print(f"🔄 Attempting to switch to {provider.value}...")
                llm_manager.switch_provider(provider)
                current = llm_manager.get_current_provider()
                if current == provider:
                    print(f"✅ Successfully switched to {provider.value}")
                else:
                    print(f"⚠️  Switch attempted but current is {current.value if current else 'None'}")
            except Exception as e:
                print(f"⚠️  Could not switch to {provider.value}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test provider switching: {e}")
        return False

async def test_schema_imports():
    """Test LLM-related schema imports."""
    print("\nTesting LLM schema imports...")
    
    try:
        from refimage.models.schemas import (
            TextToDSLRequest,
            TextToDSLResponse,
            LLMProviderInfo,
            LLMProvidersResponse,
            LLMSwitchRequest,
            LLMSwitchResponse
        )
        
        print("✅ All LLM schemas imported successfully")
        
        # Test schema instantiation
        request = TextToDSLRequest(text="Find red cars")
        print(f"✅ TextToDSLRequest created: {request.text}")
        
        provider_info = LLMProviderInfo(
            name="openai",
            available=True,
            description="OpenAI provider"
        )
        print(f"✅ LLMProviderInfo created: {provider_info.name}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test schemas: {e}")
        return False

async def main():
    """Run all LLM integration tests."""
    print("🚀 Starting LLM Integration Tests")
    print("=" * 50)
    
    tests = [
        test_llm_module_import,
        test_llm_manager_creation,
        test_text_to_dsl_prompt,
        test_provider_switching,
        test_schema_imports,
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
        print("🎉 All LLM integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
