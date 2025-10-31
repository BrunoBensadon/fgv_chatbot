# demo_rag_langgraph.py
import os
import json
import logging
import getpass
from typing import List, Dict, Any, Sequence, Optional
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated

# LangChain & LangGraph imports
from langchain.chat_models import init_chat_model
from langchain.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage

# ---------------------- Logging ----------------------
logging.basicConfig(
    filename="chattributo_messages.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------- Configuration ----------------------
BASE_DIR = Path(__file__).parent
INGEST_DIR = BASE_DIR / "ingest"
RAG_CORE = BASE_DIR / "rag_core"
LEGAL_DIR = BASE_DIR / "legal"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"  # persistence directory for Chroma

INDEX_MAP_PATH = INGEST_DIR / "index.json"
INGEST_CONFIG_PATH = INGEST_DIR / "ingestion_config.json"

# Environment / API keys
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API key (GOOGLE_API_KEY): ")

# Retrieval params (can be overridden by ingestion_config.json)
DEFAULT_K = 4
SIMILARITY_THRESHOLD = 0.70  # rough threshold for confidence (0-1)

# ---------------------- LangChain Model + Prompt ----------------------
# Using the same init_chat_model pattern as your demo_wrap.
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

system_instruction = (
    "You are ChatTributo, an assistant specialized on Brazil's Income Tax Reform Bill PL-1087/2025. "
    "Always cite article/section sources from the retrieved context when present, and keep calculations traceable."
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------- Utility: Load index.json (intent routing map) ----------------------
def load_index_map(path: Path = INDEX_MAP_PATH) -> Dict[str, Any]:
    if not path.exists():
        logging.warning(f"index.json not found at {path}. Intent routing will be disabled.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

INDEX_MAP = load_index_map()

# ---------------------- Ingestion: Load files, chunk, embed, and persist vectorstore ----------------------
def ingest_documents(
    source_dirs: List[Path] = [RAG_CORE, LEGAL_DIR],
    persist_directory: Path = VECTORSTORE_DIR,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
):
    """
    Runs a simple ingestion pipeline:
      - Loads .md and .pdf documents
      - Splits into chunks (RecursiveCharacterTextSplitter)
      - Creates embeddings and persists a Chroma vector DB
    """
    docs: List[Document] = []
    # Load text files from rag_core and legal folders
    for sd in source_dirs:
        if not sd.exists():
            logging.info(f"Source dir {sd} does not exist; skipping.")
            continue
        for p in sd.rglob("*"):
            if p.suffix.lower() in [".md", ".txt"]:
                loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    # attach metadata
                    d.metadata.update({"source_file": str(p.relative_to(BASE_DIR))})
                docs.extend(loaded)
            elif p.suffix.lower() in [".pdf"]:
                loader = UnstructuredPDFLoader(str(p))
                loaded = loader.load()
                for d in loaded:
                    d.metadata.update({"source_file": str(p.relative_to(BASE_DIR))})
                docs.extend(loaded)

    # fallback: if no docs found, try explicit files
    if not docs:
        fallback_paths = [
            RAG_CORE / "pl1087_rag_master.md",
            RAG_CORE / "pl1087_faq.md",
            LEGAL_DIR / "pl1087_text_extracted.md",
        ]
        for p in fallback_paths:
            if p.exists():
                loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    d.metadata.update({"source_file": str(p.relative_to(BASE_DIR))})
                docs.extend(loaded)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Persist vectorstore
    persist_directory.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    vectordb.persist()
    logging.info(f"Ingested {len(split_docs)} chunks into Chroma at {persist_directory}")
    return vectordb

# Build or load vector store at startup
if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
    logging.info("No vectorstore detected; running ingestion.")
    VECTORDB = ingest_documents()
else:
    # load existing
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    VECTORDB = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)

# ---------------------- Helper: Retriever function ----------------------
def retrieve_for_intent(query: str, intent: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """
    Use INDEX_MAP to decide which sources/filters to query; fallback to global retriever.
    Returns list of dicts: { 'page_content', 'metadata', 'score' }
    """
    # Decide filters and namespaces based on INDEX_MAP
    filters = None
    namespace = None
    if INDEX_MAP and intent in INDEX_MAP:
        mapping = INDEX_MAP[intent]
        # mapping might include preferred `source_files` or `tags`
        filters = mapping.get("filters")
        namespace = mapping.get("namespace")

    try:
        retriever = VECTORDB.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.get_relevant_documents(query)
        # LangChain Document has page_content + metadata
        out = []
        # Score estimation (chroma does not always return scores via this method) — attempt similarity by re-embedding:
        # For demo, we'll omit precise score and mark as 1.0
        for d in results:
            out.append({"page_content": d.page_content, "metadata": d.metadata, "score": 1.0})
        return out
    except Exception as e:
        logging.exception("Retriever error")
        return []

# ---------------------- Pseudocode / Numeric executor (example: IRPFM sample) ----------------------
def compute_irpfm_demo(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A small deterministic calculation to simulate IRPFM logic.
    For real deployment, replace with validated fiscal formulas from pl1087_rag_master.md.
    """
    # Example inputs: salary, dependents, deductions
    salary = float(params.get("salary", 0))
    dependents = int(params.get("dependents", 0))
    deduction_per_dependent = float(params.get("deduction_per_dependent", 189.59))  # example
    taxable = max(0.0, salary - dependents * deduction_per_dependent)
    # simple progressive table mock
    if taxable <= 1903.98:
        rate = 0.0
        tax = 0.0
    elif taxable <= 2826.65:
        rate = 0.075
        tax = taxable * rate - 142.80
    elif taxable <= 3751.05:
        rate = 0.15
        tax = taxable * rate - 354.80
    elif taxable <= 4664.68:
        rate = 0.225
        tax = taxable * rate - 636.13
    else:
        rate = 0.275
        tax = taxable * rate - 869.36
    return {
        "salary": salary,
        "dependents": dependents,
        "taxable": round(taxable, 2),
        "rate": rate,
        "tax_due": round(max(0.0, tax), 2)
    }

# ---------------------- LangGraph State Schema & Nodes ----------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    session_id: Optional[str]

# Node: Intent classification
def intent_classifier_node(state: State) -> Dict[str, Any]:
    """
    Use the LLM to classify the user's intent into one of the known intents.
    As a fallback, use keyword heuristics.
    """
    user_msg = state["messages"][-1].content if state["messages"] else ""
    intents_list = list(INDEX_MAP.keys()) if INDEX_MAP else [
        "profile", "exemptions", "irpfm_general", "dividends", "deductions", "international", "inequality"
    ]
    classification_prompt = (
        f"You are an intent classifier specialized for PL-1087 assistant. "
        f"Available intents: {', '.join(intents_list)}.\n\n"
        f"User question: '''{user_msg}'''\n\n"
        f"Return only a single JSON object with fields: intent (one of the available intents), confidence (0-1), "
        f"and if any numeric parameters exist, include them under params (e.g., {{'salary': 5000}})."
    )
    response = model.invoke(
        ChatPromptTemplate.from_messages([("system", "You are a classifier."), ("user", classification_prompt)]).invoke(state)
    )
    text = response.content.strip()
    logging.info(f"Intent classifier raw output: {text}")
    # Try to parse JSON out of the response; fallback to heuristic
    intent = None
    confidence = 0.0
    params = {}
    try:
        parsed = json.loads(text)
        intent = parsed.get("intent")
        confidence = float(parsed.get("confidence", 0.0))
        params = parsed.get("params", {})
    except Exception:
        # Heuristics
        low_text = user_msg.lower()
        if "dividend" in low_text or "dividendos" in low_text:
            intent = "dividends"
            confidence = 0.6
        elif "isento" in low_text or "isenção" in low_text:
            intent = "exemptions"
            confidence = 0.6
        elif "calcula" in low_text or "simula" in low_text or "salário" in low_text:
            intent = "irpfm_general"
            confidence = 0.5
        else:
            intent = "profile"
            confidence = 0.4
    # store into state for next node
    return {"intent": intent, "confidence": confidence, "params": params}

# Node: Retriever
def retriever_node(state: State, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the classified intent to fetch top-K chunks. Returns `retrievals` list.
    """
    # get latest user message and intent from meta
    user_msg = state["messages"][-1].content if state["messages"] else ""
    intent = meta.get("intent") or "general"
    k = int(meta.get("k", DEFAULT_K))
    docs = retrieve_for_intent(user_msg, intent, k=k)
    # if no docs, empty list returned
    return {"retrievals": docs, "intent": intent}

# Node: Pseudocode executor (optional)
def pseudocode_executor_node(state: State, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    If intent corresponds to a computational node (example: irpfm_general),
    try to run the pseudocode/executor and include the result.
    """
    intent = meta.get("intent")
    params = meta.get("params", {}) or {}
    result = None
    if intent in ("irpfm_general", "irpfm_calc", "irpfm_demo"):
        result = compute_irpfm_demo(params)
    return {"exec_result": result}

# Node: Generator (LLM final composition)
def generator_node(state: State, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compose final answer using:
      - user question
      - retrieved chunks (with metadata)
      - optional exec_result
    The reply should include inline citations of the form [source_file:section_id] when possible.
    """
    user_msg = state["messages"][-1].content if state["messages"] else ""
    retrievals = meta.get("retrievals", [])
    exec_result = meta.get("exec_result")
    intent = meta.get("intent", "general")

    # Build context block for the LLM
    context_snippets = []
    for i, r in enumerate(retrievals, start=1):
        md = r.get("metadata", {})
        src = md.get("source_file", md.get("source", "unknown"))
        # attempt to include article_ref if present in metadata
        article_ref = md.get("article_ref") or md.get("section_id")
        citation_label = f"{src}#{article_ref}" if article_ref else f"{src}"
        snippet = f"[{i}] ({citation_label})\n{r.get('page_content')[:1200]}"  # truncate long content
        context_snippets.append(snippet)

    # system + user prompt to generate answer
    system_text = (
        "You are ChatTributo — an assistant specialized on PL-1087/2025. "
        "Use the provided context snippets to answer precisely. When you use a snippet, cite it using the bracketed index [n]. "
        "If a calculation was run, show steps and include the exec_result JSON."
    )

    user_prompt = f"Question: {user_msg}\n\n"
    if exec_result:
        user_prompt += f"Precomputed calculation result (exec_result): {json.dumps(exec_result)}\n\n"
    if context_snippets:
        user_prompt += "Context snippets:\n" + "\n\n".join(context_snippets) + "\n\n"
    user_prompt += "Answer concisely but fully, and provide cited sources indices. If you are not confident, say so and return the top 3 snippets."

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_text), ("user", user_prompt)]
    ).invoke(state)

    response = model.invoke(prompt)
    # Construct reply object with text + citations list
    reply_text = response.content
    citations = [r.get("metadata", {}).get("source_file", "unknown") for r in retrievals[:3]]
    return {"messages": [response], "reply_text": reply_text, "citations": citations}

# Node: Fallback
def fallback_node(state: State, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    When retrieval confidence low or no intent, return top 3 excerpts + options.
    """
    user_msg = state["messages"][-1].content if state["messages"] else ""
    retrievals = meta.get("retrievals", [])
    top3 = retrievals[:3]
    options = [
        "Run a sample calculation",
        "Search external regulation",
        "Escalate to human expert"
    ]
    # Create a textual reply summarizing top3
    reply_lines = ["I couldn't confidently find a direct answer. Here are the top excerpts I found:"]
    for i, r in enumerate(top3, start=1):
        src = r.get("metadata", {}).get("source_file", "unknown")
        excerpt = r.get("page_content", "")[:400].replace("\n", " ")
        reply_lines.append(f"[{i}] {src} — {excerpt}...")
    reply_lines.append("\nOptions: " + " | ".join(options))
    reply_text = "\n\n".join(reply_lines)
    # Wrap as a fake LLM message object for memory compatibility
    fake_response = HumanMessage(reply_text)
    return {"messages": [fake_response], "reply_text": reply_text, "options": options}

# ---------------------- Build LangGraph workflow ----------------------
workflow = StateGraph(state_schema=State)

# Register nodes
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("pseudocode_executor", pseudocode_executor_node)
workflow.add_node("generator", generator_node)
workflow.add_node("fallback", fallback_node)

# Graph edges: START -> intent_classifier -> retriever -> (pseudocode_executor) -> generator -> END
workflow.add_edge(START, "intent_classifier")
workflow.add_edge("intent_classifier", "retriever")
workflow.add_edge("retriever", "pseudocode_executor")
workflow.add_edge("pseudocode_executor", "generator")
# fallback edge: generator could decide to call fallback; for simplicity, we'll call fallback when retrievals empty
# (this logic implemented in compiled invocation config)

memory_app = workflow.compile(checkpointer=MemorySaver())

# ---------------------- FastAPI ----------------------
app = FastAPI(title="ChatTributo - PL-1087 RAG (LangGraph Demo)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = "Portuguese"
    session_id: Optional[str] = None
    k: Optional[int] = DEFAULT_K

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    client_host = request.client.host
    session_id = req.session_id or client_host
    logging.info(f"Incoming request from {client_host} (session={session_id}) message: {req.message}")

    # Build initial message list
    input_messages = [HumanMessage(content=req.message)]
    state_payload = {"messages": input_messages, "language": req.language, "session_id": session_id}

    # 1) Intent classification
    intent_out = intent_classifier_node(state_payload)
    intent = intent_out.get("intent")
    confidence = intent_out.get("confidence", 0.0)
    params = intent_out.get("params", {})

    # 2) Retrieval
    retrievals = retrieve_for_intent(req.message, intent, k=req.k or DEFAULT_K)
    # Quick confidence check: if no retrievals or low confidence, forward to fallback
    if not retrievals or confidence < 0.35:
        fb = fallback_node(state_payload, {"retrievals": retrievals})
        reply = fb["reply_text"]
        logging.info(f"Fallback used for session {session_id}; returning top excerpts.")
        return {"reply": reply, "meta": {"intent": intent, "confidence": confidence, "used_fallback": True}}

    # 3) Pseudocode execution (only for some intents)
    exec_result = None
    if intent in ("irpfm_general", "irpfm_calc", "irpfm_demo"):
        exec_result = compute_irpfm_demo(params)

    # 4) Generation
    gen_meta = {"retrievals": retrievals, "intent": intent, "exec_result": exec_result}
    gen_out = generator_node(state_payload, gen_meta)
    reply_text = gen_out.get("reply_text", "")
    citations = gen_out.get("citations", [])

    # Log
    logging.info(f"Session {session_id}; intent={intent}; confidence={confidence}; reply_len={len(reply_text)}")

    # Persist to LangGraph memory (invoke compiled workflow so memory is saved)
    try:
        # We pass a minimal config: thread_id -> session_id
        memory_app.invoke({"messages": input_messages, "language": req.language, "session_id": session_id}, {"configurable": {"thread_id": session_id}})
    except Exception as e:
        logging.warning(f"Memory save failed: {e}")

    return {"reply": reply_text, "citations": citations, "meta": {"intent": intent, "confidence": confidence}}

# ---------------------- Run ----------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run("demo_rag_langgraph:app", host=host, port=port, reload=True, log_level="debug")
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}); falling back to 8080")
        uvicorn.run("demo_rag_langgraph:app", host=host, port=8080, reload=True, log_level="debug")
