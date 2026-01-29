import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

# --- NEW IMPORTS FOR READING PDFS (RAG) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import your fatigue logic
from fatigue import UPSFatigueModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.title("✈️ UPS Pilot Assistant: Contract & Tech")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("🛠 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.header("📚 Knowledge Base")
    st.info("Upload Contract, Ref Guide, AOM, FOM here.")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and api_key and st.button("Process Documents"):
        with st.spinner("Analyzing Contract & Manuals... (This may take a moment)"):
            # 1. Save uploaded files to temp so LangChain can read them
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                # Add source metadata so we know if it came from AOM or Contract
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(docs)
                os.remove(tmp_file_path) # Cleanup

            # 2. Split text into chunks (AOMs are big, so we need efficient chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # 3. Create Vector Store (The "Brain")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.success(f"Processed {len(chunks)} pages of data!")

# --- TABS ---
tab1, tab2 = st.tabs(["💬 Contract & Tech Chat", "📊 Fatigue Analysis"])

# --- TAB 1: CHATBOT (NOW FUNCTIONAL) ---
with tab1:
    st.subheader("Ask about Contract, Schedule, or Aircraft Systems")
    
    # Initialize the QA Chain if vector store exists
    if st.session_state.vector_store and api_key:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
        
        # Custom Prompt to handle mix of Contract vs AOM
        template = """
        You are a UPS Pilot Assistant. You have access to the Pilot Contract, Reference Guides, and Aircraft Manuals (AOM/FOM).
        
        RULES:
        1. If the question is about **Pay, Schedule, or Rules**, use the 'Contract' or 'Reference Guide'.
        2. If the question is about **Aircraft Systems, Limitations, or Procedures**, use the 'AOM' or 'FOM'.
        3. Always cite the document you found the answer in (e.g., [Source: 757 AOM.pdf]).
        4. If you don't know, say you don't know. Do not guess on safety items.

        Context: {context}
        Question: {question}
        Answer:
        """
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        # Chat Interface
        user_query = st.chat_input("Ex: 'What is
