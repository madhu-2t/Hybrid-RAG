# # retrieval_full.py
# import json
# import pickle
# import argparse
# from pathlib import Path
# # from langchain_community.vectorstores import FAISS
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.retrievers import EnsembleRetriever
# # FAISS is in community
# from langchain_community.vectorstores import FAISS
# # BM25 is in community
# from langchain_community.retrievers import BM25Retriever
# # Ensemble is in the main langchain package
# from langchain.retrievers import EnsembleRetriever
# # Embeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# # Constants
# FAISS_INDEX_PATH = "faiss_index"
# BM25_PATH = "bm25_retriever.pkl"
# EMB_MODEL = "all-MiniLM-L6-v2"

# def get_retrieval_chain(top_k_retrieve=30):
#     """
#     Loads indices and returns an EnsembleRetriever (Hybrid Search).
#     """
#     # Load Dense Retriever (FAISS)
#     embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
#     try:
#         vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#         retriever_dense = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
#     except Exception as e:
#         raise FileNotFoundError(f"Could not load FAISS index: {e}")

#     # Load Sparse Retriever (BM25)
#     try:
#         with open(BM25_PATH, "rb") as f:
#             retriever_bm25 = pickle.load(f)
#             retriever_bm25.k = top_k_retrieve
#     except Exception as e:
#         raise FileNotFoundError(f"Could not load BM25 index: {e}")

#     # Hybrid (Ensemble)
#     # Weight: 0.5 for Dense, 0.5 for Sparse
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[retriever_bm25, retriever_dense],
#         weights=[0.5, 0.5]
#     )
#     return ensemble_retriever

# def query_pipeline(query: str, top_k_retrieve: int = 30, top_k_return: int = 5, **kwargs):
#     """
#     Runs the ensemble retriever and formats output for the generator.
#     """
#     retriever = get_retrieval_chain(top_k_retrieve)
    
#     # Retrieve docs
#     docs = retriever.invoke(query)
    
#     # Simple truncation to top_k_return
#     selected_docs = docs[:top_k_return]

#     # Format for consistency with generator input
#     reranked_top = []
#     for i, doc in enumerate(selected_docs):
#         reranked_top.append({
#             "id": doc.metadata.get("id", f"doc_{i}"),
#             "excerpt": doc.page_content,
#             "score": 1.0 / (i + 1) # Dummy score based on rank, as Ensemble doesn't return raw scores easily
#         })

#     return {
#         "query": query,
#         "reranked_top": reranked_top
#     }

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", required=True)
#     args = parser.parse_args()
#     print(json.dumps(query_pipeline(args.query), indent=2))

# # retrieval_full.py
# import json
# import pickle
# import argparse
# import sys
# from pathlib import Path

# # --- ROBUST IMPORT BLOCK ---
# try:
#     # 1. Check if langchain is installed at all
#     import langchain
    
#     # 2. Try importing standard components
#     from langchain_community.vectorstores import FAISS
#     from langchain_community.retrievers import BM25Retriever
#     from langchain_huggingface import HuggingFaceEmbeddings
    
#     # 3. Try importing EnsembleRetriever (location varies by version)
#     try:
#         from langchain.retrievers import EnsembleRetriever
#     except ImportError:
#         # Fallback for newer v0.3+ structures or older ones
#         from langchain.retrievers.ensemble import EnsembleRetriever

# except ImportError as e:
#     print("----------------------------------------------------------------")
#     print(f"CRITICAL IMPORT ERROR: {e}")
#     print("----------------------------------------------------------------")
#     print("Your environment is missing required packages.")
#     print("Please run the following command exactly:")
#     print("pip install --force-reinstall langchain langchain-community langchain-core langchain-huggingface faiss-cpu rank_bm25")
#     print("----------------------------------------------------------------")
#     sys.exit(1)

# # Constants
# FAISS_INDEX_PATH = "faiss_index"
# BM25_PATH = "bm25_retriever.pkl"
# EMB_MODEL = "all-MiniLM-L6-v2"

# def get_retrieval_chain(top_k_retrieve=30):
#     """
#     Loads indices and returns an EnsembleRetriever (Hybrid Search).
#     """
#     # 1. Load Dense Retriever (FAISS)
#     # We need the embedding function to load the local index
#     embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
#     if not Path(FAISS_INDEX_PATH).exists():
#         raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Run indexer_simple.py first.")
        
#     try:
#         vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
#         retriever_dense = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
#     except Exception as e:
#         raise RuntimeError(f"Failed to load FAISS index: {e}")

#     # 2. Load Sparse Retriever (BM25)
#     if not Path(BM25_PATH).exists():
#         raise FileNotFoundError(f"BM25 index not found at {BM25_PATH}. Run indexer_simple.py first.")

#     try:
#         with open(BM25_PATH, "rb") as f:
#             retriever_bm25 = pickle.load(f)
#             # Update k (number of docs to return)
#             retriever_bm25.k = top_k_retrieve
#     except Exception as e:
#         raise RuntimeError(f"Failed to load BM25 index: {e}")

#     # 3. Hybrid (Ensemble)
#     # Weight: 0.5 for Dense (Semantic), 0.5 for Sparse (Keyword)
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[retriever_bm25, retriever_dense],
#         weights=[0.5, 0.5]
#     )
#     return ensemble_retriever

# def query_pipeline(query: str, top_k_retrieve: int = 30, top_k_return: int = 5, **kwargs):
#     """
#     Runs the ensemble retriever and formats output for the generator.
#     """
#     retriever = get_retrieval_chain(top_k_retrieve)
    
#     # Retrieve docs
#     # invoke() returns a list of Document objects
#     docs = retriever.invoke(query)
    
#     # Simple truncation to top_k_return
#     selected_docs = docs[:top_k_return]

#     # Format for consistency with generator input
#     reranked_top = []
#     for i, doc in enumerate(selected_docs):
#         reranked_top.append({
#             "id": doc.metadata.get("id", f"doc_{i}"),
#             "excerpt": doc.page_content,
#             # Ensemble doesn't provide a unified score easily, so we assign a rank-based dummy score
#             "score": 1.0 / (i + 1) 
#         })

#     return {
#         "query": query,
#         "reranked_top": reranked_top
#     }

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", required=True)
#     args = parser.parse_args()
#     try:
#         print(json.dumps(query_pipeline(args.query), indent=2))
#     except Exception as e:
#         print(f"Error: {e}")


# retrieval_full.py
import json
import pickle
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# --- SAFE IMPORTS (Community only) ---
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"Import Error: {e}")
    print("Run: pip install langchain-community langchain-huggingface faiss-cpu rank_bm25")
    sys.exit(1)

# Constants
FAISS_INDEX_PATH = "faiss_index"
BM25_PATH = "bm25_retriever.pkl"
EMB_MODEL = "all-MiniLM-L6-v2"

def reciprocal_rank_fusion(results_list, k=60):
    """
    Manually fuse results from multiple retrievers using RRF.
    results_list: list of lists of Documents
    """
    fused_scores = defaultdict(float)
    doc_map = {}

    for docs in results_list:
        for rank, doc in enumerate(docs):
            # Use doc content as unique key (or metadata['id'] if reliable)
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            doc_map[doc_id] = doc
            # RRF Formula: 1 / (k + rank)
            fused_scores[doc_id] += 1 / (k + rank + 1)

    # Sort by score descending
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return list of dicts compatible with generator
    results = []
    for doc_id, score in sorted_ids:
        doc = doc_map[doc_id]
        results.append({
            "id": doc.metadata.get("id", "unknown"),
            "excerpt": doc.page_content,
            "score": score
        })
    return results

def query_pipeline(query: str, top_k_retrieve: int = 30, top_k_return: int = 5, **kwargs):
    """
    Runs BM25 and FAISS separately, then manually merges them.
    """
    # 1. Load Dense Retriever (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError(f"FAISS index missing at {FAISS_INDEX_PATH}")
    
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever_dense = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS: {e}")

    # 2. Load Sparse Retriever (BM25)
    if not Path(BM25_PATH).exists():
        raise FileNotFoundError(f"BM25 index missing at {BM25_PATH}")

    try:
        with open(BM25_PATH, "rb") as f:
            retriever_bm25 = pickle.load(f)
            retriever_bm25.k = top_k_retrieve
    except Exception as e:
        raise RuntimeError(f"Failed to load BM25: {e}")

    # 3. Run both retrievers independently
    docs_dense = retriever_dense.invoke(query)
    docs_bm25 = retriever_bm25.invoke(query)

    # 4. Manual Hybrid Merge (RRF)
    reranked_top = reciprocal_rank_fusion([docs_dense, docs_bm25])
    
    # 5. Slice to top_k_return
    return {
        "query": query,
        "reranked_top": reranked_top[:top_k_return]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(query_pipeline(args.query), indent=2))
    except Exception as e:
        print(f"Error: {e}")