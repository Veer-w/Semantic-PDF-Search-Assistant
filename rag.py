import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
import tempfile
import os
import transformers
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
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)

        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)

        # Use a more powerful embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        db = Chroma.from_documents(documents, embedding=embeddings)
        os.unlink(tmp_path)
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def create_llm_pipeline():
    """Create a more powerful LLM pipeline using a larger model"""
    try:
        # Use a larger model if GPU available, fallback to smaller if not
        if torch.cuda.is_available():
            model_name = "gpt2-medium"  # Larger model (can use gpt2-large if enough GPU memory)
        else:
            model_name = "gpt2"
            
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        
        # Configure generation parameters for better responses
        pipeline = transformers.pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=150,  # Generate longer responses
            temperature=0.7,     # Balance creativity and coherence
            top_p=0.95,          # Nucleus sampling for better text quality
            repetition_penalty=1.2,  # Reduce repetition
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=pipeline)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Properly implement BaseRetriever for LangChain compatibility
class ScoreThresholdRetriever(BaseRetriever):
    """Retriever that filters results based on similarity score threshold."""
    
    vectorstore: Chroma
    k: int = 4
    threshold: float = 0.7
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query with score filtering."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        filtered_docs = [doc for doc, score in docs_with_scores if score >= self.threshold]
        # Return at least one document even if below threshold
        return filtered_docs if filtered_docs else [doc for doc, _ in docs_with_scores[:1]]
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async implementation - simply calls the sync version."""
        return self._get_relevant_documents(query)

def generate_response(query, retriever):
    """Enhanced RAG with better prompting and context integration"""
    try:
        # Create LLM pipeline
        llm = create_llm_pipeline()
        if not llm:
            return "Error: Could not initialize language model."
        
        # Custom prompt template that formats retrieved documents properly
        prompt_template = """
        You are an assistant that answers questions based on the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION: {question}
        
        Based only on the context provided, please give a comprehensive answer to the question.
        If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
        
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
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        
        st.subheader("Advanced Settings")
        search_type = st.radio("Search Type", ["Similarity", "MMR (Diverse Results)"])
        search_k = st.slider("Number of Results to Retrieve", 3, 15, 5)
        
        hardware_info = "Using GPU" if torch.cuda.is_available() else "Using CPU"
        st.info(f"Hardware Detection: {hardware_info}")

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
            # Use our properly implemented BaseRetriever subclass
            retriever = ScoreThresholdRetriever(
                vectorstore=st.session_state.vector_store,
                k=search_k,
                threshold=similarity_threshold
            )
            
            # Store the vanilla retriever for document display only
            display_retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr" if search_type == "MMR (Diverse Results)" else "similarity",
                search_kwargs={"k": search_k}
            )
            
            query = st.text_input("Enter your question:")
            if query:
                with st.spinner('Retrieving information and generating response...'):
                    response = generate_response(query, retriever)
                    
                    st.subheader("Generated Response")
                    st.write(response)
                    
                    # Add option to view retrieved documents
                    with st.expander("View Raw Retrieved Documents"):
              
                        docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=search_k)
                        for i, (doc, score) in enumerate(docs_with_scores):
                            st.markdown(f"**Document {i+1} (Page {doc.metadata.get('page', 'unknown')}) - Relevance: {score:.4f}**")
                            threshold_indicator = "✅" if score >= similarity_threshold else "❌"
                            st.markdown(f"**Above threshold: {threshold_indicator}**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.divider()

            if st.button("Clear and Upload New PDF"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
