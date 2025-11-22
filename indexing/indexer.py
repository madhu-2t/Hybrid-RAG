# indexing/indexer.py
import sys
import os
import json
import pickle
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Allow importing config from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def load_chunks():
    if not config.CHUNKS_FILE.exists():
        return []
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_indexing():
    chunks = load_chunks()
    if not chunks:
        print("⚠️ No chunks found. Run ingest/ingest.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # --- 1. FAISS (Incremental) ---
    existing_ids = set()
    vectorstore = None
    
    if config.FAISS_INDEX_PATH.exists():
        print("Loading existing FAISS index...")
        try:
            vectorstore = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            for doc in vectorstore.docstore._dict.values():
                if "id" in doc.metadata:
                    existing_ids.add(doc.metadata["id"])
        except Exception as e:
            print(f"Index corrupted ({e}), rebuilding...")

    new_docs = []
    for c in chunks:
        if c["id"] not in existing_ids:
            new_docs.append(Document(page_content=c["text"], metadata=c["metadata"]))

    if new_docs:
        print(f"Adding {len(new_docs)} new docs to FAISS...")
        if vectorstore:
            vectorstore.add_documents(new_docs)
        else:
            vectorstore = FAISS.from_documents(new_docs, embeddings)
        vectorstore.save_local(config.FAISS_INDEX_PATH)
    else:
        print("FAISS is up to date.")

    # --- 2. BM25 (Full Rebuild) ---
    print("Updating BM25 index...")
    all_docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = config.TOP_K_RETRIEVE
    
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    
    print("✅ Indexing complete.")

if __name__ == "__main__":
    run_indexing()