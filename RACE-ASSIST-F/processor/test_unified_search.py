from unified_processor import UnifiedProcessor
import json

def test_unified_search():
    """Test the complete unified search functionality"""
    
    processor = UnifiedProcessor()
    
    print("RACE-AI Complete Unified Search Test")
    print("=" * 60)
    
    try:
        # Get database statistics
        stats = processor.get_database_stats()
        print("Database Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  PDF files: {stats['pdf_files']} ({stats['pdf_chunks']} chunks)")
        print(f"  Text files: {stats['text_files']} ({stats['text_chunks']} chunks)")
        print()
        
        # Get available files
        available_files = processor.get_available_files()
        print("Available files by type:")
        print(f"PDF files ({len(available_files['PDF'])}):")
        for file in available_files['PDF'][:3]:
            print(f"  - {file}")
        if len(available_files['PDF']) > 3:
            print(f"  ... and {len(available_files['PDF']) - 3} more")
        
        print(f"\nText files ({len(available_files['TEXT'])}):")
        for file in available_files['TEXT'][:3]:
            print(f"  - {file}")
        if len(available_files['TEXT']) > 3:
            print(f"  ... and {len(available_files['TEXT']) - 3} more")
        print()
        
    except Exception as e:
        print(f"Error getting database info: {e}")
        print("Make sure to run the unified processor first!")
        return
    
    # Test comprehensive queries
    test_queries = [
        {
            'query': 'artificial intelligence program',
            'description': 'AI Programs (All Sources)'
        },
        {
            'query': 'cybersecurity courses admission',
            'description': 'Cybersecurity & Admission (All Sources)'
        },
        {
            'query': 'business analytics career opportunities',
            'description': 'Business Analytics Careers (All Sources)'
        }
    ]
    
    for test in test_queries:
        print(f"Query: '{test['query']}' - {test['description']}")
        print("-" * 50)
        
        try:
            # Search all sources
            all_results = processor.search_similar(test['query'], top_k=3)
            print("Top 3 results from all sources:")
            for i, (chunk, score, metadata) in enumerate(all_results, 1):
                print(f"{i}. [{metadata['source_type']}] {metadata['source_file']}")
                print(f"   Score: {score:.4f}")
                print(f"   Text: {chunk[:150]}...")
                print()
            
            # Search PDF sources only
            pdf_results = processor.search_similar(test['query'], top_k=2, source_type='PDF')
            if pdf_results:
                print("Top PDF results:")
                for i, (chunk, score, metadata) in enumerate(pdf_results, 1):
                    print(f"{i}. {metadata['source_file']} (Score: {score:.4f})")
                    print(f"   {chunk[:100]}...")
                print()
            
            # Search text sources only
            text_results = processor.search_similar(test['query'], top_k=2, source_type='TEXT')
            if text_results:
                print("Top text file results:")
                for i, (chunk, score, metadata) in enumerate(text_results, 1):
                    print(f"{i}. {metadata['source_file']} (Score: {score:.4f})")
                    print(f"   {chunk[:100]}...")
                print()
            
        except Exception as e:
            print(f"Error searching for '{test['query']}': {e}")
        
        print("=" * 60)
    
    # Test specific file filtering
    print("Testing file-specific search:")
    try:
        ai_results = processor.search_similar("program structure curriculum", top_k=2, filter_by_file="Artificial Intelligence")
        if ai_results:
            print("AI-specific results:")
            for i, (chunk, score, metadata) in enumerate(ai_results, 1):
                print(f"{i}. [{metadata['source_type']}] {metadata['source_file']}")
                print(f"   Score: {score:.4f}")
                print(f"   {chunk[:120]}...")
        else:
            print("No AI-specific results found")
    except Exception as e:
        print(f"Error with file filtering: {e}")

def run_processor_and_test():
    """Run the unified processor and then test it"""
    
    print("Step 1: Running Unified Processor")
    print("=" * 40)
    
    processor = UnifiedProcessor()
    
    try:
        result = processor.process_unified_content()
        
        if result:
            print("\nStep 2: Testing Unified Search")
            print("=" * 40)
            test_unified_search()
        else:
            print("Failed to process content!")
            
    except Exception as e:
        print(f"Error running processor: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        # Run processor first, then test
        run_processor_and_test()
    else:
        # Just run tests (assumes database already exists)
        test_unified_search()