import os
from typing import List, Optional
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.tools import tool

class RAGRetriever:
    """
    Wrapper around a persistent FAISS vectorstore.
    - .invoke(query, k) returns List[Document] (LangChain Document objects)
    - .retrieve(query, k) returns list of dicts {"content", "metadata"} (convenience)
    """

    def __init__(self, vectorstore_path: str = "vectorstore", model_name: str = "all-MiniLM-L6-v2"):
        self.vectorstore_path = vectorstore_path
        self.model_name = model_name
        self._db: Optional[FAISS] = None
        self._embeddings = None
        self._load()

    def _load(self):
        path = Path(self.vectorstore_path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorstore path not found: {path.resolve()}")
        # Load embeddings and FAISS
        self._embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)
        # allow_dangerous_deserialization=True is required for some saved FAISS indices
        self._db = FAISS.load_local(str(self.vectorstore_path), self._embeddings, allow_dangerous_deserialization=True)

    def invoke(self, query: str, k: int = 3) -> List[Document]:
        """
        Return top-k Documents for the query.
        This method name mirrors your example: retriever.invoke(query, k=3)
        """
        if self._db is None:
            self._load()
        # Use similarity_search so return type is List[Document]
        docs = self._db.similarity_search(query, k=k)
        return docs

    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        """
        Convenience helper returning simple dicts with content and metadata.
        """
        docs = self.invoke(query, k=k)
        return [{"content": d.page_content, "metadata": d.metadata or {}} for d in docs]

    def pretty_retrieve(self, query: str, k: int = 3):
        results = self.retrieve(query, k)
        print(f"\n🔍 Top {len(results)} results for: {query}\n")
        for i, r in enumerate(results, 1):
            snippet = r["content"].strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            print(f"[{i}] {snippet}\n")


_GLOBAL_RETRIEVER: Optional[RAGRetriever] = None

def get_global_retriever(vectorstore_path: str = "vectorstore", model_name: str = "all-MiniLM-L6-v2") -> RAGRetriever:
    """
    Returns a single global retriever instance. Will initialize once.
    You can call this from tools or other modules.
    """
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = RAGRetriever(vectorstore_path=vectorstore_path, model_name=model_name)
    return _GLOBAL_RETRIEVER


@tool
def rag_search(query: str) -> str:
    """Top-3 chunks from KB (empty string if none). Returns RAG_ERROR::... on exception."""
    try:
        # You may customize vectorstore path and model via env vars if desired
        vs_path = os.environ.get("VECTORSTORE_PATH", "vectorstore")
        embed_model = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        retriever = get_global_retriever(vectorstore_path=vs_path, model_name=embed_model)

        docs = retriever.invoke(query, k=3)
        if not docs:
            return ""
        # Join the raw page_content so tools/agents get plain text
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        # Keep the exact error format you requested
        return f"RAG_ERROR::{e}"