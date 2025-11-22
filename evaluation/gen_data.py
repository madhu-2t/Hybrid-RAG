# evaluation/gen_data.py
import sys
import os
import json
import random
import asyncio
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Settings
TEST_DATA_FILE = config.DATA_DIR / "test_dataset.json"
NUM_SAMPLES = 10

# Define Schema
class QA(BaseModel):
    question: str = Field(description="A specific question based on the text.")
    ground_truth: str = Field(description="The precise answer found in the text.")

def generate_test_data():
    print(f"Loading chunks from {config.CHUNKS_FILE}...")
    if not config.CHUNKS_FILE.exists():
        print("❌ Chunks file not found. Run ingest first.")
        return

    with open(config.CHUNKS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        chunks = [json.loads(line) for line in random.sample(lines, min(NUM_SAMPLES, len(lines)))]

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.5)
    parser = JsonOutputParser(pydantic_object=QA)
    prompt = ChatPromptTemplate.from_template(
        "Generate ONE question and its answer based ONLY on this text:\n{text}\n{format_instructions}"
    )
    chain = prompt | llm | parser

    results = []
    print(f"Generating {len(chunks)} QA pairs...")
    
    async def process(chunk):
        try:
            res = await chain.ainvoke({"text": chunk['text'], "format_instructions": parser.get_format_instructions()})
            return {"question": res['question'], "ground_truth": res['ground_truth'], "context": chunk['text']}
        except: return None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [process(c) for c in chunks]
    # Simple gather for script
    results = loop.run_until_complete(asyncio.gather(*tasks))
    results = [r for r in results if r]

    with open(TEST_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved {len(results)} test cases to {TEST_DATA_FILE}")

if __name__ == "__main__":
    generate_test_data()