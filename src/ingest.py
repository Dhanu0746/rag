"""
Document ingestion pipeline.

Loads supported documents (PDF, DOCX, TXT, Markdown),
chunks them using LlamaIndex, creates embeddings,
stores them in ChromaDB, and builds a BM25 index.
"""

import argparse
import pickle
from pathlib import Path

import chromadb
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
STORE_DIR = Path(__file__).parent.parent / "storage"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------
# Load Documents
# ---------------------------------------------------

def load_documents(username) -> list[Document]:
    """
    Load supported documents from the data directory.
    """

    user_data_dir = DATA_DIR / username

    reader = SimpleDirectoryReader(
    input_dir=str(user_data_dir),
        recursive=True,
        required_exts=[
            ".pdf",
            ".md",
            ".txt",
            ".docx",
        ],
    )

    documents = reader.load_data()

    print(f"Loaded {len(documents)} documents")

    return documents


# ---------------------------------------------------
# Chunk Documents
# ---------------------------------------------------

def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
):

    md_parser = MarkdownNodeParser()

    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    final_nodes = []

    for doc in documents:

        try:
            nodes = md_parser.get_nodes_from_documents([doc])

        except Exception:

            nodes = sentence_splitter.get_nodes_from_documents([doc])

        for node in nodes:

            if len(node.text) > chunk_size * 4:

                sub_nodes = sentence_splitter.get_nodes_from_documents(
                    [
                        Document(
                            text=node.text,
                            metadata=node.metadata,
                        )
                    ]
                )

                final_nodes.extend(sub_nodes)

            else:

                final_nodes.append(node)

    print(f"Produced {len(final_nodes)} chunks")

    return final_nodes


# ---------------------------------------------------
# Dense Index
# ---------------------------------------------------

def build_dense_index(nodes, config_name):

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME
    )

    chroma_client = chromadb.PersistentClient(
        path=str(STORE_DIR / "chroma")
    )

    collection = chroma_client.get_or_create_collection(
        name=config_name
    )

    texts = [
        node.get_content()
        for node in nodes
    ]

    embeddings = embed_model.get_text_embedding_batch(
        texts,
        show_progress=True,
    )

    ids = []
    metadatas = []

    for i, node in enumerate(nodes):

        meta = dict(node.metadata)

        source = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("source")
            or "Unknown"
        )

        page = (
            meta.get("page_label")
            or meta.get("page")
            or meta.get("page_number")
            or ""
        )

        ids.append(f"{config_name}_{i}")

        metadatas.append(
            {
                "source": str(source),
                "page": str(page),
                "chunk_id": i,
                "text": node.get_content(),
            }
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(
        f"[dense] Indexed {len(nodes)} chunks into Chroma collection '{config_name}'"
    )


# ---------------------------------------------------
# Sparse Index
# ---------------------------------------------------

def build_sparse_index(nodes, config_name):

    tokenized = [
        node.get_content().lower().split()
        for node in nodes
    ]

    bm25 = BM25Okapi(tokenized)

    STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    texts = [
        node.get_content()
        for node in nodes
    ]

    metadata = []

    for i, node in enumerate(nodes):

        meta = dict(node.metadata)

        source = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("source")
            or "Unknown"
        )

        page = (
            meta.get("page_label")
            or meta.get("page")
            or meta.get("page_number")
            or ""
        )

        metadata.append(
            {
                "source": str(source),
                "page": str(page),
                "chunk_id": i,
            }
        )

    with open(
        STORE_DIR / f"bm25_{config_name}.pkl",
        "wb",
    ) as f:

        pickle.dump(
            {
                "bm25": bm25,
                "texts": texts,
                "metadatas": metadata,
            },
            f,
        )

    print(
        f"[sparse] Indexed {len(nodes)} chunks into BM25 index"
    )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--config-name",
        default="default",
        help="Also used as the username / data subfolder (data/<config-name>/), "
             "matching how the Streamlit app names indexes.",
    )

    args = parser.parse_args()

    # config_name doubles as the username so that a CLI ingest run lands in the
    # same storage/bm25_<name>.pkl + Chroma collection the app looks for.
    documents = load_documents(args.config_name)

    if not documents:
        print(
            f"\n[ERROR] No documents found in data/{args.config_name}/.\n"
            f"  Add .pdf, .md, .txt, or .docx files there first, "
            f"or upload via the Streamlit app's Document Manager.\n"
        )
        return

    nodes = chunk_documents(
        documents,
        args.chunk_size,
        args.chunk_overlap,
    )

    build_dense_index(
        nodes,
        args.config_name,
    )

    build_sparse_index(
        nodes,
        args.config_name,
    )


if __name__ == "__main__":
    main()