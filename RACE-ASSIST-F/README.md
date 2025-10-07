# 🤖 RACE-AI Customer Support Platform

An AI-powered customer support platform that uses RAG (Retrieval-Augmented Generation) to provide intelligent assistance based on your program brochures and documents.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key to .env file
# GEMINI_API_KEY=your_key_here

# 3. Add PDF brochures to brochures/ folder

# 4. Process brochures
python process_brochures.py

# 5. Launch customer support chat
python start_support.py
```

## Features

- 📄 **PDF Processing**: Automatically extracts text from PDF brochures
- 🔍 **Semantic Search**: Uses FAISS for fast similarity search
- 🤖 **AI-Powered**: Leverages sentence transformers for understanding context
- 📊 **Category-based**: Organizes research by different categories/domains
- 🌐 **Web Interface**: User-friendly Streamlit application

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Brochures**:
   - Create a `brochures` folder in the project directory
   - Add your PDF brochures to this folder
   - Name files descriptively (filename becomes the category name)

3. **Process Brochures**:
   ```bash
   python process_brochures.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
RACE-AI/
├── app.py                 # Main Streamlit application
├── process_brochures.py   # PDF processing and FAISS creation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── brochures/            # PDF files (create this folder)
├── faiss_databases/      # Generated FAISS indices
└── brochure_mapping.json # Category mapping file
```

## How It Works

1. **Text Extraction**: PDFs are processed to extract text content
2. **Text Chunking**: Large texts are split into manageable chunks with overlap
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **FAISS Indexing**: Embeddings are stored in FAISS for fast retrieval
5. **Semantic Search**: User queries are matched against stored embeddings
6. **Result Ranking**: Most relevant chunks are returned with similarity scores

## Usage

1. Select a research category from the sidebar
2. Enter your research question in the search box
3. Adjust the number of results if needed
4. Click "Search" to find relevant information
5. Review the ranked results with relevance scores

## Categories

Categories are automatically created based on PDF filenames. For example:
- `ai_research.pdf` → "AI Research" category
- `funding_opportunities.pdf` → "Funding Opportunities" category
- `publication_guidelines.pdf` → "Publication Guidelines" category

## Technical Details

- **Embedding Model**: all-MiniLM-L6-v2 (lightweight and efficient)
- **Vector Database**: FAISS with cosine similarity
- **Text Processing**: PyPDF2 for PDF extraction
- **Web Framework**: Streamlit for the user interface
- **Chunk Size**: 500 words with 50-word overlap for context preservation

## Future Enhancements

- [ ] Support for multiple file formats (DOC, TXT, etc.)
- [ ] Advanced query processing with GPT integration
- [ ] User authentication and personalized recommendations
- [ ] Export functionality for search results
- [ ] Real-time document updates and reindexing