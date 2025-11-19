"""
build_vectorstore.py

Run once to create a persistent FAISS vectorstore from a single document.
The vectorstore will be saved in ./vectorstore by default.

Usage:
    python build_vectorstore.py path/to/document.pdf
"""

import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


def load_document(path: Path):
    """Load a single PDF or TXT document."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return loader.load()


def build_vectorstore(docs, embed_model="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
    """Split documents and build FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[+] Split into {len(chunks)} chunks.")

    embeddings = SentenceTransformerEmbeddings(model_name=embed_model)
    print(f"[+] Generating embeddings using model: {embed_model} ...")
    db = FAISS.from_documents(chunks, embeddings)
    return db


def main(doc_path: str, out_dir: str = "rag/vectorstore"):
    """Build and persist FAISS vectorstore."""
    path = Path(doc_path)
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    print(f"[*] Loading document: {path.name}")
    docs = load_document(path)

    print("[*] Building FAISS vectorstore ...")
    db = build_vectorstore(docs)

    print(f"[*] Saving to {out_path}/")
    db.save_local(str(out_path))

    print(f"[✓] Vectorstore successfully saved to {out_path.absolute()}")
    print("You can now load it later using:")
    print('  from langchain.vectorstores import FAISS')
    print(f'  from langchain.embeddings import SentenceTransformerEmbeddings')
    print(f'  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")')
    print(f'  db = FAISS.load_local("{out_dir}", embeddings, allow_dangerous_deserialization=True)')
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_vectorstore.py path/to/document.pdf")

    doc_path = os.environ.get("INGEST_PATH")
    main(doc_path)
