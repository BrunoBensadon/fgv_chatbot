# C:\Users\bruno\Documents\Bensadon\FGV\Projetos III\fgv_chatbot

import os
import getpass
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

app = FastAPI()

#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],   # or specify Typebot's frontend port ["http://localhost:3000"] 
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)

class Message(BaseModel):
    message: str

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

prompt = PromptTemplate.from_template(
    "User said: {msg}. Reply politely."
)

# Dummy chain
chain = prompt | llm

@app.post("/chat")
async def chat(m: Message, request: Request):
    # Log incoming request
    client_host = request.client.host
    logging.info(f"Incoming request from {client_host} with message: {m.message}")

    response = chain.invoke({"msg": m.message})
    reply_text = response.content if hasattr(response, "content") else str(response)

    # Log outgoing reply
    logging.info(f"Replying with: {reply_text}")

    return {"reply": reply_text}

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run("demo:app", host=host, port=port, reload=True, log_level="debug")
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}), falling back to 8080...")
        uvicorn.run("demo:app", host=host, port=8080, reload=True, log_level="debug")