# gen_eval_data.py
import json
import random
import os
import asyncio
from typing import List, Dict
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Config
CHUNKS_FILE = "data/chunks.jsonl"
OUTPUT_FILE = "data/test_dataset.json"
NUM_SAMPLES = 10  # How many test questions to generate

# Setup Model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# Define Output Schema (Question/Answer Pair)
class QA(BaseModel):
    question: str = Field(description="A specific question based on the text.")
    ground_truth: str = Field(description="The precise answer to the question found in the text.")

parser = JsonOutputParser(pydantic_object=QA)

prompt = ChatPromptTemplate.from_template(
    """You are a teacher preparing an exam. 
    Given the following text chunk, generate ONE specific question and its correct answer (ground_truth).
    
    Text Chunk:
    {text}
    
    {format_instructions}
    """
)

chain = prompt | llm | parser

def load_random_chunks(n=NUM_SAMPLES):
    chunks = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Pick random lines to ensure variety
    selected_lines = random.sample(lines, min(n, len(lines)))
    for line in selected_lines:
        chunks.append(json.loads(line))
    return chunks

async def generate_row(chunk):
    try:
        response = await chain.ainvoke({
            "text": chunk['text'], 
            "format_instructions": parser.get_format_instructions()
        })
        return {
            "question": response['question'],
            "ground_truth": response['ground_truth'],
            "source_chunk_id": chunk['id']
        }
    except Exception as e:
        print(f"Error generating QA: {e}")
        return None

async def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_random_chunks(NUM_SAMPLES)
    
    print(f"Generating {len(chunks)} synthetic QA pairs using Gemini...")
    tasks = [generate_row(chunk) for chunk in chunks]
    results = []
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await task
        if res:
            results.append(res)
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved {len(results)} test cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())