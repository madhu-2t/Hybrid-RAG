# app.py
import streamlit as st
import os
import shutil
from pathlib import Path

# --- Import your backend modules ---
# We use the robust logic we already built
from ingest.ingest import ingest_folder
from indexer_simple import build_indexes_incremental
import retrieval_full
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
st.set_page_config(page_title="DocuChat RAG", page_icon="📚")
DATA_FOLDER = "data/raw_pdfs"
CHUNKS_FILE = "data/chunks.jsonl"

# Ensure directories exist
Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

# --- 1. Session State & History Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# --- 2. Define RAG Chain (Cached to avoid rebuilding) ---
@st.cache_resource
def get_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the context provided."),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return with_history

rag_chain = get_rag_chain()

# --- 3. Sidebar: File Upload & Processing ---
with st.sidebar:
    st.title("📂 Document Manager")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("Process Documents", type="primary")
    
    if process_btn and uploaded_files:
        with st.status("Processing Documents...", expanded=True) as status:
            # A. Save files to data/raw_pdfs
            st.write("Saving files...")
            new_files_count = 0
            for uploaded_file in uploaded_files:
                file_path = Path(DATA_FOLDER) / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_count += 1
            
            # B. Run Ingestion
            st.write("Ingesting PDFs (Incremental)...")
            ingest_folder(DATA_FOLDER, CHUNKS_FILE)
            
            # C. Run Indexing
            st.write("Updating Vector Index...")
            build_indexes_incremental()
            
            status.update(label="Processing Complete!", state="complete", expanded=False)
        st.success(f"Processed {new_files_count} files successfully!")

# --- 4. Main Chat Interface ---
st.title("💬 Chat with your PDFs")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Ask a question about your documents..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # A. Retrieve Context
            with st.spinner("Searching documents..."):
                retrieval_result = retrieval_full.query_pipeline(user_input)
                context_text = "\n\n".join([
                    f"[Source: {d['id']}]\n{d['excerpt']}" 
                    for d in retrieval_result.get('reranked_top', [])
                ])
            
            # B. Stream/Generate Answer
            # We pass a fixed session_id so the chain remembers THIS user's conversation
            response = rag_chain.invoke(
                {"question": user_input, "context": context_text},
                config={"configurable": {"session_id": "current_user_session"}}
            )
            
            # C. Display Response
            message_placeholder.markdown(response)
            
            # D. Save to History
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Optional: Show sources in an expander
            with st.expander("View Retrieved Sources"):
                st.markdown(context_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")