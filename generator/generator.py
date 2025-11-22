# generator/generator.py
import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Allow importing config from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_llm():
    return ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)

def generate_answer(query: str, context_docs: list):
    """
    Generates an answer using the LLM and retrieved docs.
    Used by both the UI and the Evaluation script.
    """
    llm = get_llm()
    
    # Format context string
    context_str = "\n\n".join([f"[Source: {d['id']}]\n{d['excerpt']}" for d in context_docs])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Answer the question strictly based on the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context_str, "question": query})
        return response
    except Exception as e:
        return f"Error generating answer: {e}"