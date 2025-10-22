import os
import getpass
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

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

# ---------------------- Model ------------------------
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# ---------------------- Prompt -----------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------- State ------------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# ---------------------- Node -------------------------
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# ---------------------- Workflow ---------------------
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory_app = workflow.compile(checkpointer=MemorySaver())

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
    session_id: str

@app.post("/chat")
async def chat(m: Message, request: Request):
    client_host = request.client.host
    logging.info(f"Incoming request from {client_host} (session={client_host}) with message: {m.message}")

    input_messages = [HumanMessage(m.message)]
    config = {"configurable": {"thread_id": client_host}}

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