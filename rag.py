import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
import tempfile
import os
import torch

def preprocess_text(text):
    """Cleaning and preprocess text for better matching"""
    import re
    text = re.sub(r'\s+', ' ', text.strip())  # Remove extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
    return text

def process_pdf(uploaded_file, chunk_size=500, chunk_overlap=100):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Add page numbers to metadata for better context
        for i, doc in enumerate(docs):
            doc.metadata["page"] = i + 1

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], 
            is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)

        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)

        # Use a more powerful embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        db = Chroma.from_documents(documents, embedding=embeddings)
        os.unlink(tmp_path)
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Custom retriever with better metadata handling
class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever that provides better context management"""
    
    vectorstore: Chroma
    k: int = 4
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query with improved handling"""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        # Log retrieval scores for debugging
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Document {i} score: {score} - First 50 chars: {doc.page_content[:50]}...")
            
        # Always return all docs - let the LLM determine relevance
        return [doc for doc, _ in docs_with_scores]
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async implementation - simply calls the sync version."""
        return self._get_relevant_documents(query)

def create_ollama_llm(model_name="llama3"):
    """Create an Ollama LLM instance with a specified model"""
    try:
        # Connect to local Ollama instance
        # Adjust the base_url if Ollama is not running locally
        llm = Ollama(
            model=model_name,
            temperature=0.1,  # Low temperature for factual responses
            num_ctx=4096,     # Large context window
            base_url="http://localhost:11434"  # Default Ollama URL
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {str(e)}")
        st.info("Make sure Ollama is running locally with the required model installed.")
        return None

def generate_response(query, retriever):
    """Enhanced RAG with better prompting and context integration"""
    try:
        # Create LLM (using Ollama)
        llm = create_ollama_llm()
        if not llm:
            return "Error: Could not initialize language model. Make sure Ollama is running with llama3 model installed."
        
        # Improved prompt template with better instructions
        prompt_template = """
        You are a precise and helpful assistant that answers questions based ONLY on the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        1. Read and understand the context carefully
        2. Answer the question using ONLY information from the CONTEXT provided
        3. If the exact answer isn't in the context, respond with "I don't have enough information to answer this question."
        4. Do not use any external knowledge
        5. Keep your answer focused on the question
        6. Use direct quotes from the context when possible
        7. If the context has partial information, use only what's available without making assumptions
        
        ANSWER:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simple document concatenation
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True  # Return source docs for citation
        )
        
        # Execute chain
        result = qa_chain({"query": query})
        
        # Format response with citation information
        final_response = result["result"]
        source_docs = result.get("source_documents", [])
        
        if source_docs:
            final_response += "\n\nSource pages: "
            pages = set()
            for doc in source_docs:
                if "page" in doc.metadata:
                    pages.add(str(doc.metadata["page"]))
            final_response += ", ".join(sorted(pages))
            
        return final_response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="Enhanced PDF Q&A", page_icon="📚", layout="wide")
    st.title("📚 PDF Chat with Advanced RAG Pipeline")
    st.write("Upload a PDF and ask questions about its content")

    with st.sidebar:
        st.header("Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 100, 25)
        
        st.subheader("Model Settings")
        model_name = st.selectbox(
            "Ollama Model", 
            ["llama3", "mistral", "phi3", "dolphin-phi3", "nous-hermes2"], 
            help="Select which Ollama model to use"
        )
        
        search_k = st.slider("Number of Documents to Retrieve", 2, 10, 4)
        
        st.subheader("System Information")
        hardware_info = "Using GPU" if torch.cuda.is_available() else "Using CPU"
        st.info(f"Hardware Detection: {hardware_info}")
        
        # Instructions for Ollama
        with st.expander("Setup Instructions"):
            st.markdown("""
            1. Install Ollama from [ollama.ai](https://ollama.ai/)
            2. Run `ollama pull llama3` in your terminal
            3. Make sure Ollama is running before using this app
            """)

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.session_state.vector_store is None:
            with st.spinner('Processing PDF and building vector index...'):
                st.session_state.vector_store = process_pdf(uploaded_file, chunk_size, chunk_overlap)
            if st.session_state.vector_store:
                st.success('PDF processed successfully!')

        if st.session_state.vector_store:
            # Use our enhanced retriever
            retriever = EnhancedRetriever(
                vectorstore=st.session_state.vector_store,
                k=search_k
            )
            
            query = st.text_input("Enter your question:")
            if query:
                with st.spinner('Retrieving information and generating response...'):
                    response = generate_response(query, retriever)
                    
                    st.subheader("Generated Response")
                    st.write(response)
                    
                    # Add option to view retrieved documents
                    with st.expander("View Raw Retrieved Documents"):
                        # Use the vectorstore directly to get documents with scores for display
                        docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=search_k)
                        for i, (doc, score) in enumerate(docs_with_scores):
                            st.markdown(f"**Document {i+1} (Page {doc.metadata.get('page', 'unknown')}) - Relevance: {score:.4f}**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.divider()

            if st.button("Clear and Upload New PDF"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
