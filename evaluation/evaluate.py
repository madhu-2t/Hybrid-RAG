# evaluation/evaluate.py
import sys
import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Import system modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from retrieval.retrieval import retrieve_documents
from generator.generator import generate_answer

def run_eval():
    TEST_DATA_FILE = config.DATA_DIR / "test_dataset.json"
    
    if not TEST_DATA_FILE.exists():
        print("❌ Test data not found. Run evaluation/gen_data.py first.")
        return

    with open(TEST_DATA_FILE, "r") as f:
        test_data = json.load(f)

    results_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print(f"Running evaluation on {len(test_data)} samples...")
    
    for item in test_data:
        q = item["question"]
        
        # 1. Run YOUR Retrieval
        docs = retrieve_documents(q)
        contexts = [d['excerpt'] for d in docs]
        
        # 2. Run YOUR Generation
        ans = generate_answer(q, docs)
        
        results_dict["question"].append(q)
        results_dict["answer"].append(ans)
        results_dict["contexts"].append(contexts)
        results_dict["ground_truth"].append(item["ground_truth"])

    # 3. RAGAS Evaluation
    dataset = Dataset.from_dict(results_dict)
    
    eval_llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL)
    eval_emb = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print("Calculating RAGAS metrics...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_emb
    )

    df = scores.to_pandas()
    df.to_csv(config.DATA_DIR / "evaluation_report.csv", index=False)
    print("\n✅ Evaluation Complete! Report saved to data/evaluation_report.csv")
    print(scores)

if __name__ == "__main__":
    run_eval()