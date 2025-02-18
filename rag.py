import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os
import transformers
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], is_separator_regex=False
        )
        documents = text_splitter.split_documents(docs)

        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})

        db = Chroma.from_documents(documents, embedding=embeddings)
        retriever = db.as_retriever()
        os.unlink(tmp_path)
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def perform_search(db, query, k=4):
    """Enhanced search function with RAG integration"""
    try:
        retriever = db.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)
        return retrieved_docs[:k]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_response(query, retriever):
    """Use GPT-2 to generate responses using retrieved documents"""
    try:
        # Load GPT-2 model
        model_name = "gpt2"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20)
        llm = HuggingFacePipeline(pipeline=pipeline)

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        # Generate response
        result = qa_chain.run(query)
        return result
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="Enhanced PDF Q&A", page_icon="📚", layout="wide")
    st.title("📚 PDF Chat with RAG and GPT-2")
    st.write("Upload a PDF and ask questions about its content")

    with st.sidebar:
        st.header("Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 400, 100, 25)
        num_results = st.slider("Number of Results", 1, 10, 4)

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.session_state.vector_store is None:
            with st.spinner('Processing PDF...'):
                st.session_state.vector_store = process_pdf(uploaded_file, chunk_size, chunk_overlap)
            if st.session_state.vector_store:
                st.success('PDF processed successfully!')

        if st.session_state.vector_store:
            query = st.text_input("Enter your question:")
            if query:
                with st.spinner('Retrieving and generating response...'):
                    retriever = st.session_state.vector_store.as_retriever()
                    response = generate_response(query, retriever)
                    st.subheader("Generated Response")
                    st.write(response)

            if st.button("Clear and Upload New PDF"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
