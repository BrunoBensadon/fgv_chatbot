"""
build_vectorstore.py

Run once to create a persistent FAISS vectorstore from a single document.
The vectorstore will be saved in ./vectorstore by default.

ENHANCED: Now includes comprehensive metadata for citations:
- source_file: Original document filename
- page: Page number (for PDFs)
- chunk_id: Unique identifier for each chunk
- chunk_index: Sequential position within document
- total_chunks: Total number of chunks from this document

Usage:
    python build_vectorstore.py path/to/document.pdf
"""

import os
import sys
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def load_document(path: Path) -> List[Document]:
    """Load a single PDF or TXT document with metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # PyPDFLoader already includes page numbers in metadata
        print(f"[+] Loaded PDF with {len(docs)} pages")
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        # Add page 1 as default for text files
        for doc in docs:
            doc.metadata["page"] = 1
        print(f"[+] Loaded text file")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Add source filename to all documents
    source_name = path.name
    for doc in docs:
        doc.metadata["source"] = source_name
        # Ensure page exists (fallback to None if not set)
        if "page" not in doc.metadata:
            doc.metadata["page"] = None
    
    return docs


def enhance_chunk_metadata(chunks: List[Document], source_filename: str) -> List[Document]:
    """
    Add comprehensive metadata to each chunk for citation purposes.
    
    Adds:
    - chunk_id: Unique identifier (source_chunkN)
    - chunk_index: Position in the document (0-indexed)
    - total_chunks: Total number of chunks
    - source: Already set, but ensure it's present
    """
    total = len(chunks)
    
    for i, chunk in enumerate(chunks):
        # Ensure source is set
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = source_filename
        
        # Add chunk identification
        chunk.metadata["chunk_id"] = f"{Path(source_filename).stem}_chunk_{i}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total
        
        # Page should already be set by the splitter (inherited from parent doc)
        # If not, set a default
        if "page" not in chunk.metadata or chunk.metadata["page"] is None:
            chunk.metadata["page"] = "Unknown"
    
    return chunks


def build_vectorstore(
    docs: List[Document], 
    source_filename: str,
    embed_model: str = "all-MiniLM-L6-v2", 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> FAISS:
    """
    Split documents and build FAISS vectorstore with enhanced metadata.
    
    Args:
        docs: List of Document objects from loader
        source_filename: Name of source file for metadata
        embed_model: Name of sentence transformer model
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        FAISS vectorstore with metadata-enriched chunks
    """
    # Split documents while preserving metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        add_start_index=True  # Adds character position within source
    )
    chunks = splitter.split_documents(docs)
    print(f"[+] Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    # Enhance metadata for citations
    chunks = enhance_chunk_metadata(chunks, source_filename)
    
    # Verify metadata (sample check)
    if chunks:
        sample = chunks[0]
        print(f"[+] Sample chunk metadata: {sample.metadata}")
    
    # Generate embeddings and create vectorstore
    embeddings = SentenceTransformerEmbeddings(model_name=embed_model)
    print(f"[+] Generating embeddings using model: {embed_model} ...")
    db = FAISS.from_documents(chunks, embeddings)
    
    return db


def main(doc_path: str, out_dir: str = "rag/vectorstore"):
    """Build and persist FAISS vectorstore with citation metadata."""
    path = Path(doc_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading document: {path.name}")
    docs = load_document(path)

    print("[*] Building FAISS vectorstore with citation metadata...")
    db = build_vectorstore(docs, source_filename=path.name)

    print(f"[*] Saving to {out_path}/")
    db.save_local(str(out_path))

    print(f"\n[✓] Vectorstore successfully saved to {out_path.absolute()}")
    print("\n" + "="*60)
    print("CITATION METADATA INCLUDED:")
    print("  - source: Document filename")
    print("  - page: Page number (for PDFs)")
    print("  - chunk_id: Unique chunk identifier")
    print("  - chunk_index: Position in document")
    print("  - total_chunks: Total chunks from document")
    print("="*60)
    print("\nYou can now load it later using:")
    print('  from langchain_community.vectorstores import FAISS')
    print('  from langchain_community.embeddings import SentenceTransformerEmbeddings')
    print('  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")')
    print(f'  db = FAISS.load_local("{out_dir}", embeddings, allow_dangerous_deserialization=True)')
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check for environment variable
        doc_path = os.environ.get("INGEST_PATH")
        if not doc_path:
            print("Usage: python build_vectorstore.py path/to/document.pdf")
            print("   OR: Set INGEST_PATH environment variable")
            sys.exit(1)
    else:
        doc_path = sys.argv[1]
    
    # Optional: custom output directory
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "rag/vectorstore"
    
    main(doc_path, out_dir)