# Enterprise-Grade RAG System with Hybrid Search & Reranking 🧠

This project implements an advanced **Retrieval-Augmented Generation (RAG)** system designed to solve common LLM issues like hallucinations and semantic drift. It moves beyond "Naive RAG" by integrating **Hybrid Search (BM25 + FAISS)**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking**.

## 🚀 Key Features

* **Hybrid Search:** Combines keyword precision (BM25) with semantic understanding (FAISS) to capture both specific acronyms and conceptual queries.
* **Re-Ranking:** Utilizes a Cross-Encoder to re-score the top retrieved documents, ensuring the LLM receives only the most relevant context.
* **Incremental Ingestion:** MD5 hashing logic prevents re-processing existing files, optimizing data pipeline efficiency.
* **Memory-Aware Chat:** Streamlit UI with session memory for natural, multi-turn conversations.
* **Evaluation Pipeline:** Includes scripts using **RAGAS** to quantitatively measure Faithfulness and Answer Relevance.

## 🛠️ Tech Stack

* **LLM:** Google Gemini 2.5 Flash
* **Orchestration:** LangChain
* **Vector Store:** FAISS (Local)
* **Retrieval:** BM25 (Sparse) + HuggingFace Embeddings (Dense)
* **UI:** Streamlit

## 📂 Project Structure

```bash
├── app.py                  # Main Streamlit Application
├── retrieval_full.py       # Hybrid Search & RRF Logic
├── indexer_simple.py       # Vector Database Management
├── ingest/
│   └── ingest.py           # Incremental Document Loader
├── generator/
│   └── generator.py        # LLM Generation Module
└── data/                   # Data storage (Ignored by Git)