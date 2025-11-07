import os
import getpass
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

from tools.rag_retriever import rag_search

# ---------------------- Logging ----------------------
logging.basicConfig(
    filename="messages.log",
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

# ---------------------- API Key ----------------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# ---------------------- Tools ------------------------
web_search = DuckDuckGoSearchRun(max_results=5, return_direct=False)

tools = [rag_search]

# ---------------------- Model ------------------------
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# ---------------------- State ------------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str   

# ---------------------- Agent ------------------------
# Create the system prompt as a string
def get_system_prompt(language: str) -> str:
    return f"Você é o ChatTributo, um assistente para compreender as mudanças do Projeto de Lei 1.087/2025 sobre o Imposto de Renda que utiliza uma base de conhecimento personalizada. Responda todas as perguntas em {language}."

# Create agent with react pattern
#def create_agent_graph():
    # Use create_react_agent from langgraph.prebuilt
    # This handles the agent loop properly
memory_app = create_agent(
    model,
    tools,
    state_schema=State,
    system_prompt=get_system_prompt("{language}"),
    checkpointer=MemorySaver(),
)
#    return agent_executor

# Compile with memory
#workflow = StateGraph(state_schema=State)
#workflow.add_node("model", create_agent_graph)
#workflow.add_edge(START, "model")

#memory_app = agent_executor.compile(checkpointer=MemorySaver())

# ---------------------- FastAPI ----------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str
    language: str = "Portuguese"
    session_id: str = "default"  # so multiple users can have different memory sessions

@app.post("/chat")
async def chat(m: Message, request: Request):
    client_host = request.client.host
    logging.info(f"Incoming request from {client_host} (session={m.session_id}) with message: {m.message}")

    input_messages = [HumanMessage(m.message)]
    config = {"configurable": {"thread_id": m.session_id}}

    output = memory_app.invoke({"messages": input_messages, "language": m.language}, config)
    reply = output["messages"][-1].content

    logging.info(f"Replying to {client_host} with: {reply}")
    return {"reply": reply}

# ---------------------- Run --------------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run("demo_wrap:app", host=host, port=port, reload=True, log_level="debug")
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}), falling back to 8080...")
        uvicorn.run("demo_wrap:app", host=host, port=8080, reload=True, log_level="debug")