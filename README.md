# Semantic PDF Document Analyzer

A powerful PDF document analysis tool that leverages advanced NLP techniques and vector search to enable semantic understanding and intelligent querying of PDF documents.

## 🌟 Features

- **Semantic Search**: Utilizes HuggingFace's MPNet embeddings for deep semantic understanding
- **Intelligent Text Processing**: Advanced chunking and preprocessing strategies for optimal context preservation
- **Interactive UI**: Built with Streamlit for an intuitive user experience
- **Configurable Parameters**: Adjustable chunk sizes, overlap, and result count
- **Real-time Search**: Instant semantic search with relevance scoring
- **Multiple Document Support**: Process and analyze multiple PDF documents
- **Advanced Filtering**: Results filtered by relevance scores for higher accuracy

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-pdf-analyzer.git
cd semantic-pdf-analyzer
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### System Dependencies

Some systems might require additional dependencies:

**For macOS:**
```bash
brew install poppler libmagic
```

**For Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils libmagic1
```

## 💻 Usage

1. Start the application:
```bash
streamlit run pdf_analyzer.py
```

2. Upload a PDF document using the file uploader

3. Adjust the processing parameters in the sidebar (optional):
   - Chunk Size: Controls the size of text segments
   - Chunk Overlap: Determines context preservation
   - Number of Results: Sets how many results to display

4. Enter your questions in natural language

## 🛠️ Technical Details

### Architecture

- **Frontend**: Streamlit
- **Embeddings**: HuggingFace sentence-transformers (all-mpnet-base-v2)
- **Vector Store**: Chroma
- **Text Processing**: LangChain's RecursiveCharacterTextSplitter
- **Document Loading**: LangChain's PyPDFLoader

### Key Components

1. **Document Processing**:
   - Intelligent text chunking with configurable size and overlap
   - Advanced preprocessing for clean text extraction
   - Efficient document segmentation

2. **Search System**:
   - Semantic similarity search using cosine distance
   - Relevance score filtering
   - Context-aware result ranking

3. **User Interface**:
   - Interactive parameter configuration
   - Real-time search results
   - Expandable result views with page references

## 📊 Performance

The system utilizes several optimization techniques:
- Efficient text chunking strategies
- Optimized vector similarity search
- Relevance score thresholding
- Cached document processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain for document processing utilities
- Streamlit for the interactive interface
- HuggingFace for the embedding models
- Chroma for vector storage and search

## 📧 Contact

Your Name - rohanbansode567@gmail.com
Project Link: https://github.com/yourusername/semantic-pdf-analyzer
