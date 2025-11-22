# generator/generator.py
import os
import json
import argparse
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# UPDATED: Changed to gemini-2.5-flash (1.5 is deprecated/removed)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 

def run_generation_from_reranked(query: str, reranked_top: List[Dict]):
    """
    Uses LangChain ChatGoogleGenerativeAI to generate an answer.
    """
    if not GEMINI_API_KEY:
        return {"answer": "Error: GEMINI_API_KEY not set.", "sources_used": []}

    # 1. Prepare Context
    context_parts = []
    for i, doc in enumerate(reranked_top, 1):
        context_parts.append(f"[{i}] ID: {doc.get('id')}\nContent: {doc.get('excerpt', '')}")
    context_str = "\n\n".join(context_parts)

    # 2. Setup LangChain Model
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        max_retries=1, # Reduced retries to fail fast if model is wrong
        api_key=GEMINI_API_KEY
    )

    # 3. Create Prompt
    template = """You are an expert assistant. Use ONLY the contexts below to answer the question.
If the information is not present in the contexts, reply exactly: 'Not found in the provided sources.'

Contexts:
{context}

Question:
{question}

Instructions:
1. Answer concisely (3-8 sentences).
2. Do not hallucinate.
3. After the answer, list the source IDs used.
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 4. Invoke
    try:
        response_text = chain.invoke({"context": context_str, "question": query})
        
        # Basic source extraction (naive)
        sources = [doc.get('id') for doc in reranked_top if doc.get('id') in response_text]
        
        return {
            "query": query,
            "answer": response_text,
            "sources_used": sources,
            "raw_gemini_response": "Handled by LangChain"
        }
    except Exception as e:
        return {
            "query": query,
            "answer": f"Error during generation: {str(e)}",
            "sources_used": []
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_file", help="JSON file with reranked_top")
    parser.add_argument("--query", required=True)
    parser.add_argument("--out", help="Output file")
    args = parser.parse_args()

    reranked_top = []
    if args.retrieval_file and os.path.exists(args.retrieval_file):
        with open(args.retrieval_file, "r") as f:
            data = json.load(f)
            reranked_top = data.get("reranked_top", [])

    result = run_generation_from_reranked(args.query, reranked_top)
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()