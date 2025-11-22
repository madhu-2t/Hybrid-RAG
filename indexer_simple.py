# indexer_simple.py
import json
import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

CHUNKS_PATH = "data/chunks.jsonl"
FAISS_INDEX_PATH = "faiss_index"
BM25_PATH = "bm25_retriever.pkl"
EMB_MODEL = "all-MiniLM-L6-v2"

def load_all_chunks(path=CHUNKS_PATH):
    if not Path(path).exists():
        return []
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_indexes_incremental():
    all_chunks_data = load_all_chunks()
    if not all_chunks_data:
        print("No chunks found to index.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # --- FAISS INCREMENTAL ---
    if Path(FAISS_INDEX_PATH).exists():
        print("Loading existing FAISS index...")
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            
            # --- FIX: Check METADATA for IDs, not internal UUIDs ---
            existing_ids = set()
            # Access the internal docstore dictionary to check stored metadata
            for doc in vectorstore.docstore._dict.values():
                if "id" in doc.metadata:
                    existing_ids.add(doc.metadata["id"])
            
            print(f"Found {len(existing_ids)} existing documents in index.")
        except Exception as e:
            print(f"Index corrupted or incompatible, rebuilding: {e}")
            vectorstore = None
            existing_ids = set()
    else:
        print("Creating new FAISS index...")
        vectorstore = None
        existing_ids = set()

    # Filter for chunks that are NOT in the index
    new_docs = []
    for data in all_chunks_data:
        if data["id"] not in existing_ids:
            doc = Document(
                page_content=data["text"],
                metadata={"id": data["id"], **data.get("metadata", {})}
            )
            new_docs.append(doc)

    if new_docs:
        print(f"Adding {len(new_docs)} new documents to FAISS...")
        if vectorstore:
            vectorstore.add_documents(new_docs)
        else:
            vectorstore = FAISS.from_documents(new_docs, embeddings)
        
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("FAISS index updated.")
    else:
        print("FAISS is up to date (0 new documents).")

    # --- BM25 (Full Rebuild) ---
    # BM25 is fast enough to rebuild every time for consistency
    print("Updating BM25 index...")
    all_docs_obj = [
        Document(page_content=d["text"], metadata={"id": d["id"]}) 
        for d in all_chunks_data
    ]
    retriever_bm25 = BM25Retriever.from_documents(all_docs_obj)
    retriever_bm25.k = 30
    with open(BM25_PATH, "wb") as f:
        pickle.dump(retriever_bm25, f)
    print("BM25 index updated.")

if __name__ == "__main__":
    build_indexes_incremental()