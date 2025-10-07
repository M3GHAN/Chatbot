#!/usr/bin/env python3
"""
Test script for unified FAISS database search functionality
"""

import os
import sys
from process_brochures import BrochureProcessor

def test_unified_search():
    """Test the unified FAISS database search"""
    print("Testing Unified FAISS Database Search")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = BrochureProcessor()
        
        # Check if unified database files exist
        if not os.path.exists(processor.unified_index_path):
            print(f"❌ Unified index not found: {processor.unified_index_path}")
            return False
            
        if not os.path.exists(processor.unified_metadata_path):
            print(f"❌ Unified metadata not found: {processor.unified_metadata_path}")
            return False
            
        print("✅ Unified database files found")
        
        # Get available files
        available_files = processor.get_available_files()
        print(f"📁 Available files: {len(available_files)}")
        for file in available_files[:5]:  # Show first 5 files
            print(f"   • {file}")
        if len(available_files) > 5:
            print(f"   ... and {len(available_files) - 5} more files")
        
        # Test queries
        test_queries = [
            "What are the admission requirements?",
            "Tell me about business analytics program",
            "What is the fee structure?",
            "Career opportunities in AI",
            "Cybersecurity certification details"
        ]
        
        print(f"\\n🔍 Testing search queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"\\nQuery {i}: {query}")
            try:
                results = processor.search_similar(query, top_k=3)
                print(f"   Results: {len(results)} found")
                
                for j, (chunk, score, source) in enumerate(results, 1):
                    print(f"   {j}. Score: {score:.3f} | Source: {source}")
                    print(f"      Preview: {chunk[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {str(e)}")
                return False
        
        print("\\n✅ All tests passed! Unified search is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_unified_search()
    sys.exit(0 if success else 1)