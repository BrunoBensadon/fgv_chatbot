import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool


class RAGRetriever:
    """
    Wrapper around a persistent FAISS vectorstore with citation support.
    - .invoke(query, k) returns List[Document] (LangChain Document objects)
    - .retrieve(query, k) returns list of dicts {"content", "metadata"} (convenience)
    - .retrieve_with_citations(query, k) returns structured data for citations
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
        self._db = FAISS.load_local(
            str(self.vectorstore_path), 
            self._embeddings, 
            allow_dangerous_deserialization=True
        )

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

    def retrieve_with_citations(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Retrieve documents with structured citation metadata.
        
        Returns:
            {
                "query": str,
                "chunks": [
                    {
                        "content": str,
                        "source": str,
                        "page": int or str,
                        "chunk_id": str,
                        "chunk_index": int,
                        "citation": str  # Pre-formatted citation string
                    },
                    ...
                ]
            }
        """
        docs = self.invoke(query, k=k)
        
        chunks = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            
            # Extract citation metadata with fallbacks
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page", "N/A")
            chunk_id = metadata.get("chunk_id", f"chunk_{i}")
            chunk_index = metadata.get("chunk_index", i-1)
            
            # Format citation string
            citation = self._format_citation(source, page, i)
            
            chunks.append({
                "content": doc.page_content,
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "citation": citation
            })
        
        return {
            "query": query,
            "chunks": chunks
        }

    def _format_citation(self, source: str, page: Any, index: int) -> str:
        """
        Format a citation string from metadata.
        
        Args:
            source: Source document name
            page: Page number (can be int, str, or None)
            index: Citation index number
        
        Returns:
            Formatted citation like "[1] Document.pdf, p. 5"
        """
        citation = f"[{index}] {source}"
        
        # Add page if available and not "Unknown" or None
        if page and page != "N/A" and page != "Unknown":
            try:
                # Try to format as integer if possible
                page_num = int(page)
                citation += f", p. {page_num}"
            except (ValueError, TypeError):
                # If not a number, include as-is
                citation += f", p. {page}"
        
        return citation

    def pretty_retrieve(self, query: str, k: int = 3):
        """Print formatted retrieval results with citations."""
        result = self.retrieve_with_citations(query, k)
        chunks = result["chunks"]
        
        print(f"\n🔍 Top {len(chunks)} results for: {query}\n")
        print("=" * 80)
        
        for chunk in chunks:
            # Print citation header
            print(f"\n{chunk['citation']}")
            print("-" * 80)
            
            # Print content (truncated)
            snippet = chunk["content"].strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            print(snippet)
            
            # Print metadata
            print(f"\n  📎 Chunk ID: {chunk['chunk_id']}")
            print(f"  📄 Position: Chunk {chunk['chunk_index']}")
        
        print("\n" + "=" * 80)
        print("\n📚 Citations Summary:")
        for chunk in chunks:
            print(f"  {chunk['citation']}")
        print()


# Global retriever instance
_GLOBAL_RETRIEVER: Optional[RAGRetriever] = None


def get_global_retriever(
    vectorstore_path: str = "vectorstore", 
    model_name: str = "all-MiniLM-L6-v2"
) -> RAGRetriever:
    """
    Returns a single global retriever instance. Will initialize once.
    You can call this from tools or other modules.
    """
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = RAGRetriever(
            vectorstore_path=vectorstore_path, 
            model_name=model_name
        )
    return _GLOBAL_RETRIEVER


@tool
def rag_search(query: str) -> str:
    """
    Search knowledge base and return top-3 chunks WITH citation metadata.
    
    Returns JSON string with structure:
    {
        "query": "...",
        "chunks": [
            {
                "content": "...",
                "source": "document.pdf",
                "page": 5,
                "chunk_id": "doc_chunk_0",
                "citation": "[1] document.pdf, p. 5"
            },
            ...
        ]
    }
    
    Returns empty chunks list if no results.
    Returns {"error": "..."} if exception occurs.
    """
    try:
        # Get paths from environment or use defaults
        vs_path = os.environ.get("VECTORSTORE_PATH", "rag/vectorstore")
        embed_model = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        
        # Get retriever instance
        retriever = get_global_retriever(vectorstore_path=vs_path, model_name=embed_model)
        
        # Retrieve with citations
        result = retriever.retrieve_with_citations(query, k=3)
        
        # Return as JSON string for the agent
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except FileNotFoundError as e:
        error_result = {
            "error": f"Vectorstore not found: {str(e)}",
            "query": query,
            "chunks": []
        }
        return json.dumps(error_result, ensure_ascii=False)
    except Exception as e:
        error_result = {
            "error": f"RAG_ERROR: {str(e)}",
            "query": query,
            "chunks": []
        }
        return json.dumps(error_result, ensure_ascii=False)


# Convenience function for direct usage (not as a tool)
def search_with_citations(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Direct function to search and get citations (not wrapped as a tool).
    Useful for testing or non-agent usage.
    """
    vs_path = os.environ.get("VECTORSTORE_PATH", "rag/vectorstore")
    embed_model = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
    
    retriever = get_global_retriever(vectorstore_path=vs_path, model_name=embed_model)
    return retriever.retrieve_with_citations(query, k=k)


if __name__ == "__main__":
    """Test the retriever with a sample query."""
    print("Testing RAG Retriever with Citations\n")
    
    # Set path if needed
    if not os.environ.get("VECTORSTORE_PATH"):
        os.environ["VECTORSTORE_PATH"] = "rag/vectorstore"
    
    # Test query
    test_query = "What are the main changes in income tax?"
    
    try:
        retriever = get_global_retriever()
        retriever.pretty_retrieve(test_query, k=3)
        
        print("\n" + "="*80)
        print("Testing tool output (JSON format):")
        print("="*80)
        result = rag_search(test_query)
        print(result)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nMake sure you've:")
        print("1. Run rag_builder.py to create the vectorstore")
        print("2. Set VECTORSTORE_PATH environment variable if not using default")