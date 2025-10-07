from process_webdata import WebDataProcessor
import json

def test_webdata_search():
    """Test the unified webdata search functionality"""
    
    processor = WebDataProcessor()
    
    print("RACE-AI WebData Unified Search Test")
    print("=" * 50)
    
    # Check available files
    try:
        available_files = processor.get_available_files()
        print(f"Available source files ({len(available_files)}):")
        for i, file in enumerate(available_files, 1):
            print(f"{i:2d}. {file}")
        print()
    except Exception as e:
        print(f"Error getting available files: {e}")
        return
    
    # Test queries
    test_queries = [
        "artificial intelligence",
        "cybersecurity courses",
        "business analytics program",
        "cloud architecture",
        "career opportunities",
        "admission requirements"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 30)
        
        try:
            results = processor.search_similar(query, top_k=3)
            
            if results:
                for i, (chunk, score, metadata) in enumerate(results, 1):
                    print(f"Result {i} (Score: {score:.4f}):")
                    print(f"Source: {metadata['source_file']}")
                    print(f"Chunk: {chunk[:200]}...")
                    print()
            else:
                print("No results found.")
                print()
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            print()
    
    # Test filtering by specific file
    print("Testing file-specific search:")
    print("-" * 30)
    
    try:
        ai_results = processor.search_similar("program details", top_k=2, filter_by_file="Artificial Intelligence")
        if ai_results:
            print("AI-related results:")
            for i, (chunk, score, metadata) in enumerate(ai_results, 1):
                print(f"{i}. {metadata['source_file']}: {chunk[:100]}...")
        print()
    except Exception as e:
        print(f"Error with file filtering: {e}")
        print()

if __name__ == "__main__":
    test_webdata_search()