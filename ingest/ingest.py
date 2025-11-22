# ingest/ingest.py
import argparse
import json
import hashlib
import os
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

STATE_FILE = "data/processed_state.json"

def calculate_md5(file_path):
    """Generate MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def ingest_folder(folder_path, out_path="data/chunks.jsonl"):
    folder = Path(folder_path)
    pdf_files = sorted(list(folder.rglob("*.pdf")))
    
    # Load previous state
    state = load_state()
    new_chunks = []
    files_processed_count = 0
    
    # Splitter config
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

    print(f"Scanning {len(pdf_files)} files in {folder_path}...")
    
    for pdf in tqdm(pdf_files, desc="Checking PDFs"):
        current_hash = calculate_md5(pdf)
        filename = str(pdf.name)
        
        # SKIP if hash matches known state
        if filename in state and state[filename] == current_hash:
            continue
            
        # PROCESS if new or changed
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            
            for i, c in enumerate(chunks):
                meta = c.metadata
                meta["source"] = filename
                new_chunks.append({
                    "id": f"{pdf.stem}_{i}",
                    "text": c.page_content,
                    "metadata": meta
                })
            
            # Update state
            state[filename] = current_hash
            files_processed_count += 1
            
        except Exception as e:
            print(f"Failed to process {pdf}: {e}")

    # Append ONLY new chunks to the JSONL file (don't overwrite)
    if new_chunks:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f: # 'a' for Append
            for c in new_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        
        save_state(state)
        print(f"Ingested {files_processed_count} new file(s) ({len(new_chunks)} chunks).")
    else:
        print("No new or modified files found. Skipping ingestion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out", default="data/chunks.jsonl")
    args = parser.parse_args()
    ingest_folder(args.folder, args.out)