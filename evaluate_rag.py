# evaluate_rag.py
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Import your actual RAG pipeline
import retrieval_full
from generator import generator

# Import RAGAS metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

TEST_DATA_PATH = "data/test_dataset.json"

def run_evaluation():
    # 1. Load Test Data
    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)

    questions = []
    ground_truths = []
    rag_answers = []
    retrieved_contexts = []

    print(f"Running RAG pipeline on {len(test_data)} test questions...")

    # 2. Run YOUR Pipeline on every question
    for item in tqdm(test_data):
        q = item["question"]
        
        # A. Retrieval
        retrieval_out = retrieval_full.query_pipeline(q)
        # Extract just the text list for RAGAS
        contexts = [doc['excerpt'] for doc in retrieval_out['reranked_top']]
        
        # B. Generation
        gen_out = generator.run_generation_from_reranked(q, retrieval_out['reranked_top'])
        
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        rag_answers.append(gen_out["answer"])
        retrieved_contexts.append(contexts)

    # 3. Prepare Dataset for RAGAS
    data = {
        "question": questions,
        "answer": rag_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # 4. Configure RAGAS to use Gemini (instead of OpenAI)
    # RAGAS uses an LLM to judge the quality
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Calculating Metrics (this calls the API to judge answers)...")
    
    # 5. Run Evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,      # Is the answer derived from context? (Hallucination check)
            answer_relevancy,  # Does the answer address the question?
            context_precision, # Did the retrieval find relevant chunks?
            context_recall,    # Did the retrieval find ALL necessary chunks?
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 6. Output Results
    print("\n========= 📊 EVALUATION REPORT =========")
    print(results)
    
    # Save to CSV for recruiters
    df = results.to_pandas()
    df.to_csv("rag_evaluation_report.csv", index=False)
    print("Saved detailed report to rag_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()