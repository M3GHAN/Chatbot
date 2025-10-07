import os
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import Dict, List, Tuple

class UnifiedProcessor:
    def __init__(self, brochure_folder: str = "brochures", webdata_folder: str = "webdata", output_folder: str = "faiss_databases"):
        self.brochure_folder = brochure_folder
        self.webdata_folder = webdata_folder
        self.output_folder = output_folder
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.unified_db_name = "complete_unified"
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
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
    
    def process_unified_content(self) -> Dict:
        """Process all PDF files and text files into a single unified FAISS database"""
        print("Processing all content for unified FAISS database...")
        print("=" * 60)
        
        all_chunks = []
        chunk_metadata = []  # Store source file info for each chunk
        file_info = {}  # Store information about processed files
        
        # Process PDF files from brochures folder
        if os.path.exists(self.brochure_folder):
            print(f"\nProcessing PDF files from '{self.brochure_folder}' folder...")
            pdf_files = [f for f in os.listdir(self.brochure_folder) if f.lower().endswith('.pdf')]
            print(f"Found {len(pdf_files)} PDF files")
            
            for filename in pdf_files:
                pdf_path = os.path.join(self.brochure_folder, filename)
                print(f"  Processing {filename}...")
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                if not text:
                    print(f"    Skipping {filename} - no text extracted")
                    continue
                
                # Chunk text
                chunks = self.chunk_text(text)
                if not chunks:
                    print(f"    Skipping {filename} - no chunks created")
                    continue
                
                # Add chunks to unified list with metadata
                chunk_start_idx = len(all_chunks)
                all_chunks.extend(chunks)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata.append({
                        'source_file': filename,
                        'source_type': 'PDF',
                        'source_folder': self.brochure_folder,
                        'chunk_index': i,
                        'global_chunk_index': chunk_start_idx + i,
                        'chunk_text': chunk
                    })
                
                # Store file information
                file_info[filename] = {
                    'source_type': 'PDF',
                    'source_folder': self.brochure_folder,
                    'original_text': text,
                    'num_chunks': len(chunks),
                    'chunk_start_idx': chunk_start_idx,
                    'chunk_end_idx': chunk_start_idx + len(chunks) - 1
                }
                
                print(f"    Created {len(chunks)} chunks")
        else:
            print(f"Brochures folder '{self.brochure_folder}' not found - skipping PDF processing")
        
        # Process text files from webdata folder
        if os.path.exists(self.webdata_folder):
            print(f"\nProcessing text files from '{self.webdata_folder}' folder...")
            txt_files = [f for f in os.listdir(self.webdata_folder) if f.lower().endswith('.txt')]
            print(f"Found {len(txt_files)} text files")
            
            for filename in txt_files:
                file_path = os.path.join(self.webdata_folder, filename)
                print(f"  Processing {filename}...")
                
                # Extract text
                text = self.extract_text_from_file(file_path)
                if not text:
                    print(f"    Skipping {filename} - no text extracted")
                    continue
                
                # Chunk text
                chunks = self.chunk_text(text)
                if not chunks:
                    print(f"    Skipping {filename} - no chunks created")
                    continue
                
                # Add chunks to unified list with metadata
                chunk_start_idx = len(all_chunks)
                all_chunks.extend(chunks)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata.append({
                        'source_file': filename,
                        'source_type': 'TEXT',
                        'source_folder': self.webdata_folder,
                        'chunk_index': i,
                        'global_chunk_index': chunk_start_idx + i,
                        'chunk_text': chunk
                    })
                
                # Store file information
                file_info[filename] = {
                    'source_type': 'TEXT',
                    'source_folder': self.webdata_folder,
                    'original_text': text,
                    'num_chunks': len(chunks),
                    'chunk_start_idx': chunk_start_idx,
                    'chunk_end_idx': chunk_start_idx + len(chunks) - 1
                }
                
                print(f"    Created {len(chunks)} chunks")
        else:
            print(f"Webdata folder '{self.webdata_folder}' not found - skipping text file processing")
        
        if not all_chunks:
            print("No content found to process!")
            return None
        
        print(f"\nTotal content processed:")
        print(f"  Total files: {len(file_info)}")
        print(f"  Total chunks: {len(all_chunks)}")
        
        # Breakdown by source type
        pdf_count = sum(1 for info in file_info.values() if info['source_type'] == 'PDF')
        txt_count = sum(1 for info in file_info.values() if info['source_type'] == 'TEXT')
        pdf_chunks = sum(info['num_chunks'] for info in file_info.values() if info['source_type'] == 'PDF')
        txt_chunks = sum(info['num_chunks'] for info in file_info.values() if info['source_type'] == 'TEXT')
        
        print(f"  PDF files: {pdf_count} files, {pdf_chunks} chunks")
        print(f"  Text files: {txt_count} files, {txt_chunks} chunks")
        
        print(f"\nCreating unified FAISS index...")
        
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
            'total_files': len(file_info),
            'pdf_files': pdf_count,
            'text_files': txt_count,
            'pdf_chunks': pdf_chunks,
            'text_chunks': txt_chunks
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(unified_metadata, f)
        
        # Save mapping JSON
        mapping_file = "unified_mapping.json"
        mapping_data = {
            'unified_database': {
                'name': self.unified_db_name,
                'index_path': index_path,
                'metadata_path': metadata_path,
                'total_chunks': len(all_chunks),
                'total_files': len(file_info),
                'pdf_files': pdf_count,
                'text_files': txt_count,
                'pdf_chunks': pdf_chunks,
                'text_chunks': txt_chunks,
                'files_processed': list(file_info.keys())
            }
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"\nUnified processing complete!")
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Mapping saved to: {mapping_file}")
        
        return mapping_data['unified_database']
    
    def load_unified_index(self) -> Tuple[faiss.Index, List[str], List[Dict]]:
        """Load the unified FAISS index and metadata"""
        index_path = os.path.join(self.output_folder, f"{self.unified_db_name}_index.faiss")
        metadata_path = os.path.join(self.output_folder, f"{self.unified_db_name}_metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Unified index not found at {index_path}. Please process content first.")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Unified metadata not found at {metadata_path}. Please process content first.")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata['chunks'], metadata['chunk_metadata']
    
    def search_similar(self, query: str, top_k: int = 5, source_type: str = None, filter_by_file: str = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar chunks in the unified database
        
        Args:
            query: Search query
            top_k: Number of results to return
            source_type: Filter by 'PDF' or 'TEXT' (optional)
            filter_by_file: Filter by filename substring (optional)
        """
        index, chunks, chunk_metadata = self.load_unified_index()
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search (get more results for filtering)
        search_k = min(top_k * 5, len(chunks))
        scores, indices = index.search(query_embedding, search_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks) and idx < len(chunk_metadata):
                chunk_info = chunk_metadata[idx]
                
                # Apply source type filter
                if source_type and chunk_info['source_type'] != source_type.upper():
                    continue
                
                # Apply file filter
                if filter_by_file and filter_by_file.lower() not in chunk_info['source_file'].lower():
                    continue
                
                results.append((chunks[idx], float(score), chunk_info))
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_by_source_type(self, source_type: str) -> List[Tuple[str, Dict]]:
        """Get all chunks from a specific source type (PDF or TEXT)"""
        _, chunks, chunk_metadata = self.load_unified_index()
        
        results = []
        for i, metadata in enumerate(chunk_metadata):
            if metadata['source_type'] == source_type.upper():
                results.append((chunks[i], metadata))
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the unified database"""
        _, chunks, chunk_metadata = self.load_unified_index()
        
        stats = {
            'total_chunks': len(chunks),
            'total_files': len(set(m['source_file'] for m in chunk_metadata)),
            'pdf_files': len(set(m['source_file'] for m in chunk_metadata if m['source_type'] == 'PDF')),
            'text_files': len(set(m['source_file'] for m in chunk_metadata if m['source_type'] == 'TEXT')),
            'pdf_chunks': len([m for m in chunk_metadata if m['source_type'] == 'PDF']),
            'text_chunks': len([m for m in chunk_metadata if m['source_type'] == 'TEXT']),
            'source_files': sorted(list(set(m['source_file'] for m in chunk_metadata)))
        }
        
        return stats
    
    def get_available_files(self) -> Dict[str, List[str]]:
        """Get list of all source files by type"""
        _, _, chunk_metadata = self.load_unified_index()
        
        pdf_files = set()
        text_files = set()
        
        for metadata in chunk_metadata:
            if metadata['source_type'] == 'PDF':
                pdf_files.add(metadata['source_file'])
            else:
                text_files.add(metadata['source_file'])
        
        return {
            'PDF': sorted(list(pdf_files)),
            'TEXT': sorted(list(text_files)),
            'ALL': sorted(list(pdf_files.union(text_files)))
        }

def main():
    # Initialize processor
    processor = UnifiedProcessor()
    
    print("RACE-AI Unified Content Processor")
    print("=" * 50)
    print("This will process both PDF brochures and webdata text files")
    print("into a single unified FAISS database for enhanced search.")
    print()
    
    # Check folders
    brochure_exists = os.path.exists("brochures")
    webdata_exists = os.path.exists("webdata")
    
    if not brochure_exists and not webdata_exists:
        print("Neither 'brochures' nor 'webdata' folders found!")
        print("Please ensure at least one folder exists with content.")
        return
    
    if brochure_exists:
        pdf_files = [f for f in os.listdir("brochures") if f.lower().endswith('.pdf')]
        print(f"Brochures folder: {len(pdf_files)} PDF files found")
    else:
        print("Brochures folder: Not found")
    
    if webdata_exists:
        txt_files = [f for f in os.listdir("webdata") if f.lower().endswith('.txt')]
        print(f"Webdata folder: {len(txt_files)} text files found")
    else:
        print("Webdata folder: Not found")
    
    print()
    
    # Process all content into unified database
    result = processor.process_unified_content()
    
    if result:
        print("\nExample usage:")
        print("from unified_processor import UnifiedProcessor")
        print("processor = UnifiedProcessor()")
        print("# Search all content:")
        print("results = processor.search_similar('your query')")
        print("# Search only PDFs:")
        print("pdf_results = processor.search_similar('your query', source_type='PDF')")
        print("# Search only text files:")
        print("text_results = processor.search_similar('your query', source_type='TEXT')")
        print("# Get statistics:")
        print("stats = processor.get_database_stats()")

if __name__ == "__main__":
    main()