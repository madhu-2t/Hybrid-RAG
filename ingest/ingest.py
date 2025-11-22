# ingest/ingest.py
import sys
import os
import json
import hashlib
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow importing config from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run_ingest():
    config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    
    state = {}
    if config.STATE_FILE.exists():
        with open(config.STATE_FILE, "r") as f:
            state = json.load(f)

    pdf_files = sorted(list(config.RAW_PDFS_DIR.rglob("*.pdf")))
    new_chunks = []
    processed_count = 0

    print(f"Scanning {len(pdf_files)} files in {config.RAW_PDFS_DIR}...")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

    for pdf in tqdm(pdf_files, desc="Ingesting"):
        file_hash = calculate_md5(pdf)
        filename = pdf.name
        
        if filename in state and state[filename] == file_hash:
            continue

        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            
            for i, c in enumerate(chunks):
                new_chunks.append({
                    "id": f"{pdf.stem}_{i}",
                    "text": c.page_content,
                    "metadata": {"source": filename, **c.metadata}
                })
            
            state[filename] = file_hash
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if new_chunks:
        with open(config.CHUNKS_FILE, "a", encoding="utf-8") as f:
            for c in new_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        
        with open(config.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
            
        print(f"✅ Ingested {processed_count} new files ({len(new_chunks)} chunks).")
    else:
        print("⚡ No new files to ingest.")

if __name__ == "__main__":
    run_ingest()