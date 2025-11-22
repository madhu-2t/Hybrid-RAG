# e2e_runner.py
import argparse
import json
import sys
import time
import importlib.util
from pathlib import Path

# Define Paths
INGEST_PATH = Path("ingest/ingest.py").resolve()
INDEXER_PATH = Path("indexer_simple.py").resolve()
RETRIEVAL_PATH = Path("retrieval_full.py").resolve()
GENERATOR_PATH = Path("generator/generator.py").resolve()

def load_module(path, name):
    if not path.exists():
        print(f"Error: Module {name} not found at {path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Runner")
    parser.add_argument("--query", required=True, help="Question to ask")
    
    # OPTIONAL: Only provide this if you want to update the database
    parser.add_argument("--data_folder", help="Path to PDF folder. IF PROVIDED, runs ingestion/indexing checks first.")
    
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()

    # --- CONDITIONAL STEP: DATA UPDATE ---
    if args.data_folder:
        print("\n" + "="*40)
        print(f"📂 Update Mode: Checking {args.data_folder}...")
        print("="*40)
        
        # 1. Run Ingest
        ingest_mod = load_module(INGEST_PATH, "ingest")
        ingest_mod.ingest_folder(args.data_folder, "data/chunks.jsonl")
        
        # 2. Run Indexing
        indexer_mod = load_module(INDEXER_PATH, "indexer")
        indexer_mod.build_indexes_incremental()
    else:
        print("\n" + "="*40)
        print("🚀 Fast Mode: Using existing database (Skipping file checks)")
        print("="*40)

    # --- ALWAYS RUN: RETRIEVAL & GENERATION ---
    try:
        retrieval_mod = load_module(RETRIEVAL_PATH, "retrieval_full")
        generator_mod = load_module(GENERATOR_PATH, "generator")
    except Exception as e:
        print(f"CRITICAL: Failed to load core modules: {e}")
        sys.exit(1)

    start_time = time.time()
    
    # 3. Retrieval
    print(f"🔍 Retrieving context for: '{args.query}'")
    try:
        retrieval_out = retrieval_mod.query_pipeline(args.query)
        reranked = retrieval_out.get("reranked_top", [])
    except Exception as e:
        print(f"Retrieval Error: {e}")
        print("Tip: Did you run with --data_folder at least once to build the index?")
        sys.exit(1)

    # 4. Generation
    print("🧠 Generating answer...")
    gen_result = generator_mod.run_generation_from_reranked(args.query, reranked)
    
    elapsed = round(time.time() - start_time, 2)

    # --- OUTPUT ---
    final_output = {
        "query": args.query,
        "timings_sec": elapsed,
        "answer": gen_result["answer"],
        "retrieval_docs": reranked
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved output to {args.out}")
        print(f"\nAnswer: {gen_result['answer']}")
    else:
        # Pretty print to console
        print("\n" + "-"*50)
        print(f"🤖 Answer ({elapsed}s):")
        print(gen_result["answer"])
        print("-" * 50)

if __name__ == "__main__":
    main()