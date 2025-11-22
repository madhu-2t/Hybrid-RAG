# app.py
import streamlit as st
import config
from ingest.ingest import run_ingest
from indexing.indexer import run_indexing
from retrieval.retrieval import retrieve_documents

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="DocuChat Enterprise", page_icon="🧠")

@st.cache_resource
def get_rag_chain():
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly based on the context below.\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

rag_chain = get_rag_chain()

with st.sidebar:
    st.title("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Files", type="primary") and uploaded_files:
        with st.status("Updating Knowledge Base...", expanded=True):
            st.write("Saving files...")
            config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                with open(config.RAW_PDFS_DIR / uf.name, "wb") as f:
                    f.write(uf.getbuffer())
            
            st.write("Ingesting content...")
            run_ingest()
            
            st.write("Building indexes...")
            run_indexing()
            
        st.success("System Updated!")

st.title("🧠 Enterprise RAG Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    try:
        with st.spinner("Searching..."):
            docs = retrieve_documents(query)
            context_str = "\n\n".join([f"[{d['id']}]: {d['excerpt']}" for d in docs])
        
        with st.chat_message("assistant"):
            response = rag_chain.invoke(
                {"question": query, "context": context_str},
                config={"configurable": {"session_id": "session_1"}}
            )
            st.markdown(response)
            with st.expander("View Sources"):
                st.markdown(context_str)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {e}")