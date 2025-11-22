# retrieval/retrieval.py
import sys
import os
import pickle
from collections import defaultdict

# Allow importing config from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("Critical: Missing libraries.")
    sys.exit(1)

def reciprocal_rank_fusion(results_list, k=60):
    scores = defaultdict(float)
    doc_map = {}
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            doc_map[doc_id] = doc
            scores[doc_id] += 1 / (k + rank + 1)
            
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": doc_id, "excerpt": doc_map[doc_id].page_content, "score": score, "metadata": doc_map[doc_id].metadata} 
        for doc_id, score in sorted_docs
    ]

def retrieve_documents(query: str):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    if not config.FAISS_INDEX_PATH.exists() or not config.BM25_INDEX_PATH.exists():
        raise FileNotFoundError("Indexes missing. Run indexing/indexer.py.")

    faiss_db = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever_dense = faiss_db.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVE})
    
    with open(config.BM25_INDEX_PATH, "rb") as f:
        retriever_sparse = pickle.load(f)
        retriever_sparse.k = config.TOP_K_RETRIEVE

    docs_dense = retriever_dense.invoke(query)
    docs_sparse = retriever_sparse.invoke(query)

    return reciprocal_rank_fusion([docs_dense, docs_sparse])[:config.TOP_K_RETURN]