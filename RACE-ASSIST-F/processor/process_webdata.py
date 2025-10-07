import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import Dict, List, Tuple

class WebDataProcessor:
    def __init__(self, webdata_folder: str = "webdata", output_folder: str = "faiss_databases"):
        self.webdata_folder = webdata_folder
        self.output_folder = output_folder
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.unified_db_name = "webdata_unified"
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                return text.strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                    return text.strip()
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                return ""
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_faiss_index(self, texts: List[str]) -> Tuple[faiss.Index, np.ndarray]:
        """Create FAISS index from text chunks"""
        # Generate embeddings
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index, embeddings
    
    def process_unified_webdata(self) -> Dict:
        """Process all text files in webdata folder and create a unified FAISS database"""
        if not os.path.exists(self.webdata_folder):
            print(f"Webdata folder '{self.webdata_folder}' not found!")
            return None
        
        print("Processing webdata files for unified FAISS database...")
        
        all_chunks = []
        chunk_metadata = []  # Store source file info for each chunk
        file_info = {}  # Store information about processed files
        
        # Process each text file
        for filename in os.listdir(self.webdata_folder):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(self.webdata_folder, filename)
                print(f"Processing {filename}...")
                
                # Extract text
                text = self.extract_text_from_file(file_path)
                if not text:
                    print(f"Skipping {filename} - no text extracted")
                    continue
                
                # Chunk text
                chunks = self.chunk_text(text)
                if not chunks:
                    print(f"Skipping {filename} - no chunks created")
                    continue
                
                # Add chunks to unified list with metadata
                chunk_start_idx = len(all_chunks)
                all_chunks.extend(chunks)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata.append({
                        'source_file': filename,
                        'chunk_index': i,
                        'global_chunk_index': chunk_start_idx + i,
                        'chunk_text': chunk
                    })
                
                # Store file information
                file_info[filename] = {
                    'original_text': text,
                    'num_chunks': len(chunks),
                    'chunk_start_idx': chunk_start_idx,
                    'chunk_end_idx': chunk_start_idx + len(chunks) - 1
                }
        
        if not all_chunks:
            print("No text chunks found to process!")
            return None
        
        print(f"Total chunks to process: {len(all_chunks)}")
        
        # Create unified FAISS index
        index, embeddings = self.create_faiss_index(all_chunks)
        
        # Save FAISS index
        index_filename = f"{self.unified_db_name}_index.faiss"
        index_path = os.path.join(self.output_folder, index_filename)
        faiss.write_index(index, index_path)
        
        # Save metadata
        metadata_filename = f"{self.unified_db_name}_metadata.pkl"
        metadata_path = os.path.join(self.output_folder, metadata_filename)
        
        unified_metadata = {
            'chunks': all_chunks,
            'chunk_metadata': chunk_metadata,
            'file_info': file_info,
            'embeddings': embeddings,
            'total_chunks': len(all_chunks),
            'total_files': len(file_info)
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(unified_metadata, f)
        
        # Save mapping JSON for compatibility
        mapping_file = "webdata_mapping.json"
        mapping_data = {
            'unified_database': {
                'name': self.unified_db_name,
                'index_path': index_path,
                'metadata_path': metadata_path,
                'total_chunks': len(all_chunks),
                'total_files': len(file_info),
                'files_processed': list(file_info.keys())
            }
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"\nUnified processing complete!")
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Mapping saved to: {mapping_file}")
        print(f"Total files processed: {len(file_info)}")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Files processed: {list(file_info.keys())}")
        
        return mapping_data['unified_database']
    
    def load_unified_index(self) -> Tuple[faiss.Index, List[str], List[Dict]]:
        """Load the unified FAISS index and metadata"""
        index_path = os.path.join(self.output_folder, f"{self.unified_db_name}_index.faiss")
        metadata_path = os.path.join(self.output_folder, f"{self.unified_db_name}_metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Unified index not found at {index_path}. Please process webdata first.")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Unified metadata not found at {metadata_path}. Please process webdata first.")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata['chunks'], metadata['chunk_metadata']
    
    def search_similar(self, query: str, top_k: int = 5, filter_by_file: str = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar chunks in the unified database"""
        index, chunks, chunk_metadata = self.load_unified_index()
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, min(top_k * 3, len(chunks)))  # Get more results for filtering
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks) and idx < len(chunk_metadata):
                chunk_info = chunk_metadata[idx]
                
                # Apply file filter if specified
                if filter_by_file and filter_by_file not in chunk_info['source_file']:
                    continue
                
                results.append((chunks[idx], float(score), chunk_info))
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_by_source_file(self, source_file: str) -> List[Tuple[str, Dict]]:
        """Get all chunks from a specific source file"""
        _, chunks, chunk_metadata = self.load_unified_index()
        
        results = []
        for i, metadata in enumerate(chunk_metadata):
            if metadata['source_file'] == source_file:
                results.append((chunks[i], metadata))
        
        return results
    
    def get_available_files(self) -> List[str]:
        """Get list of all source files in the unified database"""
        _, _, chunk_metadata = self.load_unified_index()
        
        files = set()
        for metadata in chunk_metadata:
            files.add(metadata['source_file'])
        
        return sorted(list(files))

def main():
    # Initialize processor
    processor = WebDataProcessor()
    
    print("RACE-AI WebData Processor")
    print("=" * 40)
    
    # Check if webdata folder exists
    if not os.path.exists("webdata"):
        print("'webdata' folder not found!")
        print("Please ensure the 'webdata' folder exists with text files.")
        return
    
    # Check if text files exist
    txt_files = [f for f in os.listdir("webdata") if f.lower().endswith('.txt')]
    if not txt_files:
        print("No text files found in 'webdata' folder.")
        print("Please add text files to the 'webdata' folder and run again.")
        return
    
    print(f"Found {len(txt_files)} text files to process:")
    for f in txt_files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(txt_files) > 5:
        print(f"  ... and {len(txt_files) - 5} more")
    
    # Process all webdata files into unified database
    result = processor.process_unified_webdata()
    
    if result:
        print("\nExample usage:")
        print("from process_webdata import WebDataProcessor")
        print("processor = WebDataProcessor()")
        print("results = processor.search_similar('your query')")
        print("files = processor.get_available_files()")

if __name__ == "__main__":
    main()