import os
import getpass
import logging
import asyncio

import uvicorn
from fastapi import FastAPI, Request, Query
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
    encoding='utf-8',
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
ngrok_cmd = ["ngrok", "http", "--domain=enough-blatantly-whale.ngrok-free.app", "8000"]

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
    return f"""Você é o ChatTributo, um assistente para compreender as mudanças do Projeto de Lei 1.087/2025 sobre o Imposto de Renda.

IMPORTANTE: Quando você usar a ferramenta rag_search, ela retornará informações em formato JSON com citações. Você DEVE:
1. Usar essas informações para responder a pergunta
2. SEMPRE incluir citações ao final da resposta no formato:

**Fontes consultadas:**
[1] Nome_do_documento.pdf, p. X, Seção Y
[2] Nome_do_documento.pdf, p. Z

3. Referenciar as fontes no texto usando [1], [2], etc. quando mencionar informações específicas

Responda todas as perguntas em {language}."""
# Create agent with react pattern
#def create_agent_graph():
    # Use create_react_agent from langgraph.prebuilt
    # This handles the agent loop properly
agent = create_agent(
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

@app.middleware("http")
async def disable_buffering(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Accel-Buffering"] = "no"   # disable buffering for nginx proxies
    return response

class Message(BaseModel):
    message: str
    language: str = "Portuguese"
    session_id: str = "default"
    stream: bool = False

@app.post("/chat")
async def chat(m: Message, request: Request):
    client_host = request.client.host
    logging.info(f"Incoming request from {client_host} (session={m.session_id}) with message: {m.message}")

    system_prompt = get_system_prompt(m.language)
    input_messages = [HumanMessage(m.message)]
    config = {"configurable": {"thread_id": m.session_id}}

    if m.stream:
        # Streaming response
        async def generate():
            try:
                full_response = ""
                async for event in agent.astream(
                    {"messages": input_messages},
                    config=config,
                    stream_mode="values"
                ):
                    if "messages" in event and len(event["messages"]) > 0:
                        last_msg = event["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            content = last_msg.content
                            if content and content != full_response:
                                # Send only the new chunk
                                chunk = content[len(full_response):]
                                full_response = content
                                yield f"data: {chunk}\n\n"
                
                logging.info(f"Streamed response to {client_host}: {full_response}")
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_msg = f"Error during streaming: {str(e)}"
                logging.error(error_msg)
                yield f"data: {error_msg}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            output = agent.invoke(
                {"messages": input_messages},
                config=config
            )
            reply = output["messages"][-1].content
            logging.info(f"Replying to {client_host} with: {reply}")
            return {"reply": reply}
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(error_msg)
            return {"reply": error_msg}

@app.get("/chat")
async def chat_stream(
    message: str = Query(...),
    language: str = "Portuguese",
    session_id: str = "default",
    request: Request = None,
):
    client_host = request.client.host if request.client else "unknown"
    logging.info(f"SSE stream opened from {client_host} - message: {message}")

    async def generate():
        full_response = ""
        try:
            async for event in agent.astream(
                {"messages": [HumanMessage(message)]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="values",
            ):
                if not event:
                    continue
                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    content = last_msg.content or ""
                    if content != full_response:
                        chunk = content[len(full_response):]
                        full_response = content
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0)  # let event loop breathe

            yield "data: [DONE]\n\n"
            logging.info(f"SSE stream closed for {client_host}")
        except Exception as e:
            logging.error(f"SSE error: {e}")
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generate(), headers=headers, media_type="text/event-stream")

@app.options("/chat")
async def options_chat():
    return {"status": "ok"}

# ---------------------- Run --------------------------
def run_uvicorn():
    """Start the FastAPI server exactly as you already defined."""
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug",
            reload_excludes=["messages.log"]
        )
        return True
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}), falling back to 8080...")
        uvicorn.run(
            "app:app",
            host=host,
            port=8080,
            reload=True,
            log_level="debug",
            reload_excludes=["messages.log"]
        )
        return True