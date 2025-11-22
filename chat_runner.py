from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import retrieval_full  # Your robust retrieval module

# 1. Setup Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Setup Prompt with History
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based on the context provided."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# 3. Create Chain
chain = prompt | llm | StrOutputParser()

# 4. Add History Management
store = {}  # In-memory storage for sessions

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. Chat Loop
print("Bot: Hello! Ask me anything about your documents. (Type 'exit' to quit)")
session_id = "user_1"

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
        
    # Retrieve context first
    retrieval = retrieval_full.query_pipeline(user_input)
    context_text = "\n".join([d['excerpt'] for d in retrieval['reranked_top']])
    
    # Generate answer with history
    response = with_history.invoke(
        {"question": user_input, "context": context_text},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Bot: {response}")