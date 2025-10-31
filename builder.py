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

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

# ── Build & persist a Chroma index ────────────────────────────────
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

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


SOURCE_DIR   = Path("docs")          # put your files here
INDEX_DIR    = Path("faiss_db_1")   # will be created if missing
EMBED_MODEL  = "all-MiniLM-L6-v2"

def load_single_doc(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    else:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    return docs  # list of Document

# Split & embed

def split(docs, chunk_size=1000, chunk_overlap=200):
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted = splitter.split_documents(docs)
    return splitted

def vectorize(docs):
     # Embeddings (sentence-transformers local)
    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    # FAISS index
    db = FAISS.from_documents(split(docs), embed)
    return db

docs = load_single_doc("C:/Users/bruno/Documents/Bensadon/FGV/Projetos III/fgv_chatbot/docs/sample.pdf")

db = vectorize(docs)

# ---------------------- Tools ------------------------

retriever = db.as_retriever(search_kwargs={"k": 2})

@tool
def rag_search_tool(query: str) -> str:
    """Search the knowledge‑base for relevant chunks"""
    results = retriever.invoke(query)
    return "".join(d.page_content for d in results)

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

class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = None

class RagJudge(BaseModel):
    sufficient: bool

class AgentState(State):          # add to previous `State`
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route:    str          # "rag", "answer", "web", "end"
    rag:      str | None   # KB result
    web:      str | None   # web‑search snippets

# ---------------------- Nodes ------------------------
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# ── Structured helpers ─────────────────
class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = None

class RagJudge(BaseModel):
    sufficient: bool

router_llm = model.with_structured_output(RouteDecision)
judge_llm  = model.with_structured_output(RagJudge)
answer_llm = model

# ── Router ─────────────────────────────
def router_node(state: AgentState) -> AgentState:
    q = state["messages"][-1].content
    decision = router_llm.invoke([
        ("system", "Decide route: rag / answer / end"),
        ("user", q)
    ])
    new_state = {**state, "route": decision.route}
    if decision.route == "end":
        new_state["messages"] += [AIMessage(content=decision.reply or "Hello!")]
    return new_state

# ── RAG lookup ─────────────────────────
def rag_node(state: AgentState) -> AgentState:
    q = state["messages"][-1].content
    chunks = rag_search_tool.invoke(q)
    verdict = judge_llm.invoke([("user", f"Question: {q} Docs: {chunks[:300]}…")])
    return {**state, "rag": chunks, "route": "answer" if verdict.sufficient else "end"}

# ── Node 4: final answer ─────────────────────────────────────────────
def answer_node(state: AgentState) -> AgentState:
    user_q = next((m.content for m in reversed(state["messages"])
                   if isinstance(m, HumanMessage)), "")

    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        ctx_parts.append("Web Search Results:\n" + state["web"])

    context = "\n\n".join(ctx_parts) if ctx_parts else "No external context available."

    prompt = f"""Please answer the user's question using the provided context.

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""

    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }

# ── Routing helpers ─────────────────────────────────────────────────
def from_router(st: AgentState) -> Literal["rag", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"

# ---------------------- Workflow ---------------------
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory_app = workflow.compile(checkpointer=MemorySaver())

agent_graph = StateGraph(AgentState)
agent_graph.add_node("router",      router_node)
agent_graph.add_node("rag_lookup",  rag_node)
agent_graph.add_node("answer",      answer_node)

agent_graph.set_entry_point("router")
agent_graph.add_conditional_edges("router", from_router,
        {"rag": "rag_lookup", "answer": "answer", "end": END})
agent_graph.add_conditional_edges("rag_lookup", after_rag,
        {"answer": "answer"})
agent_graph.add_edge("answer", END)

agent = agent_graph.compile(checkpointer=MemorySaver())



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

    output = agent.invoke({"messages": input_messages, "language": m.language}, config)
    reply = output["messages"][-1].content

    logging.info(f"Replying to {client_host} with: {reply}")
    return {"reply": reply}

# ---------------------- Run --------------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run("builder:app", host=host, port=port, reload=True, log_level="debug")
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}), falling back to 8080...")
        uvicorn.run("builder:app", host=host, port=8080, reload=True, log_level="debug")