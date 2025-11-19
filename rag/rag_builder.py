"""
build_vectorstore.py

Build a persistent FAISS vectorstore from ALL documents in a folder.
Supports incremental updates and multiple document ingestion.

ENHANCED: Now includes comprehensive metadata for citations:
- source_file: Original document filename
- page: Page number (for PDFs)
- chunk_id: Unique identifier for each chunk
- chunk_index: Sequential position within document
- total_chunks: Total number of chunks from this document
- doc_type: File extension (pdf, txt, md)

Usage:
    # Ingest all documents from a folder
    python build_vectorstore.py rag/ingest/
    
    # Ingest single document (adds to existing or creates new)
    python build_vectorstore.py rag/ingest/document.pdf
    
    # Fresh start (delete existing vectorstore first)
    python build_vectorstore.py rag/ingest/ --fresh
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def load_document(path: Path) -> List[Document]:
    """Load a single PDF or TXT document with metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    
    doc_type = path.suffix.lower().replace(".", "")
    
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        print(f"    ✓ Loaded PDF with {len(docs)} pages")
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        # Add page 1 as default for text files
        for doc in docs:
            doc.metadata["page"] = 1
        print(f"    ✓ Loaded text file")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Add source filename and doc type to all documents
    source_name = path.name
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["doc_type"] = doc_type
        # Ensure page exists (fallback to None if not set)
        if "page" not in doc.metadata:
            doc.metadata["page"] = None
    
    return docs


def load_documents_from_folder(folder_path: Path) -> List[tuple[Path, List[Document]]]:
    """
    Load all supported documents from a folder.
    
    Returns:
        List of tuples: (file_path, documents)
    """
    supported_extensions = {".pdf", ".txt", ".md"}
    all_docs = []
    
    # Find all files with supported extensions
    files = [
        f for f in folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not files:
        print(f"⚠️  No supported documents found in {folder_path}")
        print(f"    Looking for: {', '.join(supported_extensions)}")
        return []
    
    print(f"\n📁 Found {len(files)} document(s) to process:")
    for f in files:
        print(f"    • {f.name}")
    print()
    
    # Load each file
    for file_path in files:
        try:
            print(f"  📄 Loading {file_path.name}...")
            docs = load_document(file_path)
            all_docs.append((file_path, docs))
        except Exception as e:
            print(f"    ⚠️  Error loading {file_path.name}: {e}")
            continue
    
    return all_docs


def enhance_chunk_metadata(
    chunks: List[Document], 
    source_filename: str,
    doc_index: int = 0
) -> List[Document]:
    """
    Add comprehensive metadata to each chunk for citation purposes.
    
    Args:
        chunks: List of document chunks
        source_filename: Name of source file
        doc_index: Index of document in batch (for unique IDs across docs)
    
    Adds:
        - chunk_id: Unique identifier (source_chunkN)
        - chunk_index: Position in the document (0-indexed)
        - total_chunks: Total number of chunks from this document
        - source: Already set, but ensure it's present
    """
    total = len(chunks)
    source_stem = Path(source_filename).stem
    
    for i, chunk in enumerate(chunks):
        # Ensure source is set
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = source_filename
        
        # Add chunk identification with doc_index to ensure uniqueness
        chunk.metadata["chunk_id"] = f"{source_stem}_d{doc_index}_c{i}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total
        
        # Page should already be set by the splitter (inherited from parent doc)
        # If not, set a default
        if "page" not in chunk.metadata or chunk.metadata["page"] is None:
            chunk.metadata["page"] = "Unknown"
    
    return chunks


def create_vectorstore_from_documents(
    docs_list: List[tuple[Path, List[Document]]],
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> FAISS:
    """
    Create a FAISS vectorstore from multiple documents.
    
    Args:
        docs_list: List of (file_path, documents) tuples
        embed_model: Embedding model name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        FAISS vectorstore
    """
    all_chunks = []
    
    # Split each document and enhance metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    
    print("\n🔨 Processing documents...")
    for doc_index, (file_path, docs) in enumerate(docs_list):
        print(f"\n  📝 Chunking {file_path.name}...")
        chunks = splitter.split_documents(docs)
        chunks = enhance_chunk_metadata(chunks, file_path.name, doc_index)
        all_chunks.extend(chunks)
        print(f"    ✓ Created {len(chunks)} chunks")
    
    print(f"\n✓ Total chunks across all documents: {len(all_chunks)}")
    
    # Sample check
    if all_chunks:
        sample = all_chunks[0]
        print(f"\n📋 Sample chunk metadata:")
        for key, value in sample.metadata.items():
            print(f"    • {key}: {value}")
    
    # Create embeddings and vectorstore
    print(f"\n🧮 Generating embeddings using {embed_model}...")
    embeddings = SentenceTransformerEmbeddings(model_name=embed_model)
    db = FAISS.from_documents(all_chunks, embeddings)
    
    return db


def merge_vectorstores(
    existing_db: FAISS,
    new_docs_list: List[tuple[Path, List[Document]]],
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> FAISS:
    """
    Merge new documents into an existing vectorstore.
    
    Args:
        existing_db: Existing FAISS database
        new_docs_list: List of (file_path, documents) tuples to add
        embed_model: Embedding model name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        Updated FAISS vectorstore
    """
    # Create vectorstore from new documents
    new_db = create_vectorstore_from_documents(
        new_docs_list,
        embed_model,
        chunk_size,
        chunk_overlap
    )
    
    # Merge the databases
    print("\n🔗 Merging with existing vectorstore...")
    existing_db.merge_from(new_db)
    
    return existing_db


def main(
    input_path: str,
    out_dir: str = "rag/vectorstore",
    fresh: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model: str = "all-MiniLM-L6-v2"
):
    """
    Build or update FAISS vectorstore from documents.
    
    Args:
        input_path: Path to document file or folder
        out_dir: Output directory for vectorstore
        fresh: If True, delete existing vectorstore first
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embed_model: Embedding model to use
    """
    path = Path(input_path)
    out_path = Path(out_dir)
    
    print("\n" + "="*70)
    print("📚 RAG VECTORSTORE BUILDER WITH CITATIONS")
    print("="*70)
    
    # Determine if we're processing a file or folder
    if path.is_file():
        print(f"\n🎯 Mode: Single document")
        print(f"📄 Document: {path.name}")
        docs_list = [(path, load_document(path))]
    elif path.is_dir():
        print(f"\n🎯 Mode: Folder ingestion")
        print(f"📁 Folder: {path}")
        docs_list = load_documents_from_folder(path)
        if not docs_list:
            print("\n❌ No documents to process. Exiting.")
            return
    else:
        print(f"\n❌ Error: Path not found: {path}")
        return
    
    # Handle fresh start
    if fresh and out_path.exists():
        print(f"\n🗑️  Fresh start: Deleting existing vectorstore at {out_path}")
        shutil.rmtree(out_path)
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Check if vectorstore exists
    index_file = out_path / "index.faiss"
    vectorstore_exists = index_file.exists()
    
    if vectorstore_exists and not fresh:
        print(f"\n♻️  Existing vectorstore found. Will merge new documents.")
        try:
            # Load existing vectorstore
            embeddings = SentenceTransformerEmbeddings(model_name=embed_model)
            existing_db = FAISS.load_local(
                str(out_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Merge new documents
            db = merge_vectorstores(
                existing_db,
                docs_list,
                embed_model,
                chunk_size,
                chunk_overlap
            )
        except Exception as e:
            print(f"\n⚠️  Error loading existing vectorstore: {e}")
            print("Creating new vectorstore instead...")
            db = create_vectorstore_from_documents(
                docs_list,
                embed_model,
                chunk_size,
                chunk_overlap
            )
    else:
        print(f"\n🆕 Creating new vectorstore...")
        db = create_vectorstore_from_documents(
            docs_list,
            embed_model,
            chunk_size,
            chunk_overlap
        )
    
    # Save vectorstore
    print(f"\n💾 Saving vectorstore to {out_path}/")
    db.save_local(str(out_path))
    
    # Summary
    print("\n" + "="*70)
    print("✅ VECTORSTORE SUCCESSFULLY SAVED")
    print("="*70)
    print(f"\n📍 Location: {out_path.absolute()}")
    print(f"📊 Documents processed: {len(docs_list)}")
    
    # List all processed documents
    print(f"\n📚 Included documents:")
    for file_path, _ in docs_list:
        print(f"    • {file_path.name}")
    
    print(f"\n🔖 Citation metadata included:")
    print(f"    • source: Document filename")
    print(f"    • page: Page number (for PDFs)")
    print(f"    • chunk_id: Unique chunk identifier")
    print(f"    • chunk_index: Position in document")
    print(f"    • total_chunks: Total chunks per document")
    print(f"    • doc_type: File type (pdf, txt, md)")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build FAISS vectorstore from documents with citation support"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to document file or folder (default: from INGEST_PATH env var)"
    )
    parser.add_argument(
        "-o", "--output",
        default="rag/vectorstore",
        help="Output directory for vectorstore (default: rag/vectorstore)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing vectorstore and start fresh"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )
    
    args = parser.parse_args()
    
    # Get input path from args or environment
    input_path = args.input_path or os.environ.get("INGEST_PATH")
    
    if not input_path:
        parser.print_help()
        print("\n❌ Error: No input path provided")
        print("   Use: python build_vectorstore.py <path>")
        print("   Or set INGEST_PATH environment variable")
        sys.exit(1)
    
    main(
        input_path=input_path,
        out_dir=args.output,
        fresh=args.fresh,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_model=args.embed_model
    )