
import os
import json
import re
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_PATH = Path("spotify.pdf")
DATA_DIR = Path("data")
GRAPH_PATH = DATA_DIR / "knowledge_graph.graphml"
CHUNKS_PATH = DATA_DIR / "chunks.json"

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH.resolve()}")

    DATA_DIR.mkdir(exist_ok=True)

    print("Reading PDF...")
    reader = PdfReader(str(PDF_PATH))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": PDF_PATH.name, "page": page_number},
                )
            )

    if not pages:
        raise ValueError("No extractable text found. The PDF may be scanned.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")

    print("Creating FAISS vector database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(DATA_DIR / "faiss_index"))

    print("Building knowledge graph...")
    graph = nx.Graph()

    for chunk_id, chunk in enumerate(chunks):
        chunk_node = f"chunk_{chunk_id}"
        graph.add_node(
            chunk_node,
            node_type="chunk",
            text=chunk.page_content,
            page=str(chunk.metadata.get("page", "")),
        )

        # Basic entity extraction for an initial graph.
        entities = set(
            re.findall(
                r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}\b",
                chunk.page_content,
            )
        )

        for entity in entities:
            if len(entity) < 2:
                continue

            graph.add_node(entity, node_type="entity")
            graph.add_edge(chunk_node, entity, relation="mentioned_in")

    nx.write_graphml(graph, GRAPH_PATH)

    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [
                {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"FAISS index saved to: {DATA_DIR / 'faiss_index'}")
    print(f"Knowledge graph saved to: {GRAPH_PATH}")
    print(f"Chunks saved to: {CHUNKS_PATH}")
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    main()
