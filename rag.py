import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os

def preprocess_text(text):
    """Cleaning and preprocess text for better matching"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep meaningful punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def process_pdf(uploaded_file, chunk_size=500, chunk_overlap=100):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load and process the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Enhanced text splitting with smaller chunks and more overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)

        # Preprocess each document's content
        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)

        # Initialize embeddings with a better performing model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Better model
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store with more sophisticated distance metrics
        db = Chroma.from_documents(
            documents, 
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Clean up temporary file
        os.unlink(tmp_path)
        
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def perform_search(db, query, k=4):
    """Enhanced search function with MMR reranking"""
    try:
        # Get more results than needed for reranking
        results = db.similarity_search_with_relevance_scores(
            query,
            k=k*2  # Get more results initially
        )
        
        # Filter results based on relevance score
        threshold = 0.7  # Adjust this threshold based on your needs
        filtered_results = [
            doc for doc, score in results 
            if score > threshold
        ]
        
        # Return top k results after filtering
        return filtered_results[:k]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Enhanced PDF Analysis",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Enhanced PDF Document Analysis")
    st.write("Upload a PDF and ask questions about its content")

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("Advanced Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50,
                             help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 100, 25,
                                help="Overlap between chunks")
        num_results = st.slider("Number of Results", 1, 10, 4,
                              help="Number of results to display")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF if not already processed
        if st.session_state.vector_store is None:
            with st.spinner('Processing PDF...'):
                st.session_state.vector_store = process_pdf(
                    uploaded_file,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            if st.session_state.vector_store:
                st.success('PDF processed successfully!')

        if st.session_state.vector_store:
            # Query input with example
            st.write("Try to be specific in your questions. For example: 'What are the main themes in chapter 3?'")
            query = st.text_input("Enter your question about the document:")
            
            if query:
                with st.spinner('Searching...'):
                    results = perform_search(
                        st.session_state.vector_store,
                        query,
                        k=num_results
                    )
                    
                    if results:
                        st.subheader("Search Results")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}"):
                                st.markdown(f"**Content:**\n{result.page_content}")
                                st.markdown("---")
                                st.markdown(f"**Page:** {result.metadata.get('page', 'N/A')}")
                    else:
                        st.warning("No relevant results found. Try rephrasing your question.")

            # Clear button
            if st.button("Clear and Upload New PDF"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()