import os
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import Dict, List, Tuple

class BrochureProcessor:
    def __init__(self, brochure_folder: str = "brochures", output_folder: str = "faiss_databases"):
        self.brochure_folder = brochure_folder
        self.output_folder = output_folder
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.mapping_file = "brochure_mapping.json"
        
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
    
    def process_single_brochure(self, pdf_path: str, category: str) -> Dict:
        """Process a single PDF brochure"""
        print(f"Processing {pdf_path}...")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Chunk text
        chunks = self.chunk_text(text)
        if not chunks:
            return None
        
        # Create FAISS index
        index, embeddings = self.create_faiss_index(chunks)
        
        # Save FAISS index
        index_filename = f"{category}_index.faiss"
        index_path = os.path.join(self.output_folder, index_filename)
        faiss.write_index(index, index_path)
        
        # Save text chunks and metadata
        metadata_filename = f"{category}_metadata.pkl"
        metadata_path = os.path.join(self.output_folder, metadata_filename)
        
        metadata = {
            'chunks': chunks,
            'embeddings': embeddings,
            'original_text': text,
            'pdf_path': pdf_path
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return {
            'category': category,
            'index_path': index_path,
            'metadata_path': metadata_path,
            'num_chunks': len(chunks),
            'pdf_path': pdf_path
        }
    
    def process_all_brochures(self):
        """Process all PDF files in the brochure folder"""
        if not os.path.exists(self.brochure_folder):
            print(f"Brochure folder '{self.brochure_folder}' not found!")
            return
        
        mapping = {}
        
        # Process each PDF file
        for filename in os.listdir(self.brochure_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.brochure_folder, filename)
                category = os.path.splitext(filename)[0].lower()  # Use filename without extension as category
                
                result = self.process_single_brochure(pdf_path, category)
                if result:
                    mapping[category] = result
        
        # Save mapping to JSON
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"\nProcessing complete! Mapping saved to {self.mapping_file}")
        print(f"Available categories: {list(mapping.keys())}")
        
        return mapping
    
    def load_faiss_index(self, category: str) -> Tuple[faiss.Index, List[str]]:
        """Load FAISS index and metadata for a specific category"""
        # Load mapping
        if not os.path.exists(self.mapping_file):
            raise FileNotFoundError("Mapping file not found. Please process brochures first.")
        
        with open(self.mapping_file, 'r') as f:
            mapping = json.load(f)
        
        if category not in mapping:
            raise ValueError(f"Category '{category}' not found. Available: {list(mapping.keys())}")
        
        # Load FAISS index
        index_path = mapping[category]['index_path']
        index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = mapping[category]['metadata_path']
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata['chunks']
    
    def search_similar(self, query: str, category: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks in a specific category"""
        index, chunks = self.load_faiss_index(category)
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks):
                results.append((chunks[idx], float(score)))
        
        return results

def main():
    # Initialize processor
    processor = BrochureProcessor()
    
    # Create sample brochure folder structure
    os.makedirs("brochures", exist_ok=True)
    
    print("RACE-AI Brochure Processor")
    print("=" * 40)
    
    # Check if brochures exist
    if not os.listdir("brochures"):
        print("No PDF files found in 'brochures' folder.")
        print("Please add PDF files to the 'brochures' folder and run again.")
        return
    
    # Process all brochures
    mapping = processor.process_all_brochures()
    
    if mapping:
        print("\nExample usage:")
        print("from process_brochures import BrochureProcessor")
        print("processor = BrochureProcessor()")
        print("results = processor.search_similar('your query', 'category_name')")

if __name__ == "__main__":
    main()