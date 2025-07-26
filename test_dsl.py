"""
Test DSL parser functionality in detail.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.refimage.dsl import DSLParser, TextQuery, TagFilter, AndQuery, OrQuery


def test_dsl_parser():
    """Test DSL parser with various queries."""
    parser = DSLParser()
    
    # Test simple text query
    query1 = parser.parse("red car")
    print(f"✅ Simple text query: {type(query1).__name__} - '{query1.text}'")
    
    # Test tag query
    query2 = parser.parse("#sports #car")
    print(f"✅ Tag query: {type(query2).__name__} - tags: {query2.tags}")
    
    # Test combined query (text + tags)
    query3 = parser.parse("fast car #luxury")
    print(f"✅ Combined query: {type(query3).__name__} with {len(query3.operands)} operands")
    
    # Test OR query
    query4 = parser.parse("red car OR blue house")
    print(f"✅ OR query: {type(query4).__name__} with {len(query4.operands)} operands")
    
    # Test AND query
    query5 = parser.parse("sports car AND #expensive")
    print(f"✅ AND query: {type(query5).__name__} with {len(query5.operands)} operands")
    
    # Test weighted query
    query6 = parser.parse("luxury car^0.8")
    print(f"✅ Weighted query: {type(query6).__name__} with weight {query6.weight}")
    
    return True


def test_dsl_query_types():
    """Test different DSL query node types."""
    
    # TextQuery test
    text_query = TextQuery("red sports car", weight=0.9)
    assert text_query.text == "red sports car"
    assert text_query.weight == 0.9
    print("✅ TextQuery creation successful")
    
    # TagFilter test
    tag_filter = TagFilter(["sports", "luxury"], mode="all")
    assert "sports" in tag_filter.tags
    assert "luxury" in tag_filter.tags
    assert tag_filter.mode == "all"
    print("✅ TagFilter creation successful")
    
    # AndQuery test
    and_query = AndQuery([text_query, tag_filter])
    assert len(and_query.operands) == 2
    print("✅ AndQuery creation successful")
    
    # OrQuery test
    or_query = OrQuery([text_query, tag_filter])
    assert len(or_query.operands) == 2
    print("✅ OrQuery creation successful")
    
    return True


def test_dsl_edge_cases():
    """Test edge cases and error handling."""
    parser = DSLParser()
    
    try:
        # Empty query should fail
        parser.parse("")
        print("❌ Empty query should have failed")
        return False
    except AssertionError:
        print("✅ Empty query properly rejected")
    
    try:
        # None query should fail
        parser.parse(None)
        print("❌ None query should have failed")
        return False
    except AssertionError:
        print("✅ None query properly rejected")
    
    # Test whitespace handling
    query = parser.parse("  red car  ")
    print(f"✅ Whitespace handling: '{query.text}'")
    
    # Test case insensitive tags
    tag_query = parser.parse("#Sports #CAR")
    assert "sports" in tag_query.tags
    assert "car" in tag_query.tags
    print("✅ Case insensitive tag handling")
    
    return True


if __name__ == "__main__":
    print("🧪 Testing DSL Parser functionality...")
    
    success = True
    success &= test_dsl_parser()
    success &= test_dsl_query_types()
    success &= test_dsl_edge_cases()
    
    if success:
        print("\n🎉 All DSL tests passed!")
    else:
        print("\n❌ Some DSL tests failed!")
        sys.exit(1)
