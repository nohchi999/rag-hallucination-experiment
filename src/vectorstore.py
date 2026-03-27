"""
src/vectorstore.py

ChromaDB vector store setup and evidence retrieval for 3 conditions:
  - full:    standard top-k retrieval
  - partial: top-k minus chunks that contain the ground-truth answer
  - none:    retrieval excluding all chunks from the target question's document

Run standalone: python src/vectorstore.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_experiment"


def _chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def _contains_answer(chunk: str, answer: str) -> bool:
    return answer.strip().lower() in chunk.strip().lower()


def get_client() -> chromadb.PersistentClient:
    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_DB_PATH)


def get_collection(client: chromadb.PersistentClient):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def setup_vectorstore(dataset: list[dict]) -> chromadb.Collection:
    """
    Index all 200 documents into ChromaDB.
    Each chunk gets metadata: question_id, chunk_index, contains_answer.
    """
    client = get_client()
    collection = get_collection(client)

    existing = collection.count()
    if existing > 0:
        logger.info(f"Collection already has {existing} chunks — skipping re-indexing.")
        return collection

    logger.info(f"Indexing {len(dataset)} documents into ChromaDB...")

    batch_ids = []
    batch_docs = []
    batch_metas = []

    for item in tqdm(dataset, desc="Indexing"):
        qid = item["question_id"]
        answer = item["ground_truth"]
        doc = item["document"]
        chunks = _chunk_text(doc)

        for ci, chunk in enumerate(chunks):
            chunk_id = f"q{qid}_c{ci}"
            batch_ids.append(chunk_id)
            batch_docs.append(chunk)
            batch_metas.append({
                "question_id": qid,
                "chunk_index": ci,
                "contains_answer": int(_contains_answer(chunk, answer)),
            })

            # Flush in batches of 500 to avoid memory issues
            if len(batch_ids) >= 500:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    logger.info(f"Indexed {collection.count()} chunks total.")
    return collection


def get_evidence(
    question: str,
    question_id: int,
    condition: str,
    collection: chromadb.Collection,
) -> list[str]:
    """
    Retrieve evidence chunks for a question under a given condition.

    Returns a list of text chunks.
    """
    if condition == "full":
        results = collection.query(
            query_texts=[question],
            n_results=config.TOP_K,
        )
        return results["documents"][0]

    elif condition == "partial":
        # Retrieve more candidates, then filter out answer-containing chunks
        n_fetch = config.TOP_K * 5
        results = collection.query(
            query_texts=[question],
            n_results=min(n_fetch, collection.count()),
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        non_answer_chunks = [
            doc for doc, meta in zip(docs, metas)
            if not meta.get("contains_answer", 0)
        ]

        if len(non_answer_chunks) >= config.TOP_K:
            return non_answer_chunks[:config.TOP_K]

        # Fallback: not enough non-answer chunks — truncate the best chunk to first sentence
        answer_chunks = [
            doc for doc, meta in zip(docs, metas)
            if meta.get("contains_answer", 0)
        ]
        truncated = []
        for chunk in answer_chunks:
            first_sentence = chunk.split(".")[0] + "." if "." in chunk else chunk[:100]
            truncated.append(first_sentence)

        combined = non_answer_chunks + truncated
        return combined[:config.TOP_K]

    elif condition == "none":
        # Retrieve from collection excluding all chunks of this question
        results = collection.query(
            query_texts=[question],
            n_results=min(config.TOP_K * 3, collection.count()),
            where={"question_id": {"$ne": question_id}},
        )
        return results["documents"][0][:config.TOP_K]

    else:
        raise ValueError(f"Unknown condition: {condition}")


if __name__ == "__main__":
    from src.dataset import load_cached_dataset
    import config as cfg

    data = load_cached_dataset(cfg.FILTERED_NQ_FILE)
    collection = setup_vectorstore(data)

    q = data[0]
    for cond in cfg.CONDITIONS:
        chunks = get_evidence(q["question"], q["question_id"], cond, collection)
        print(f"\n[{cond}] Retrieved {len(chunks)} chunk(s) for: {q['question'][:60]}")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i}: {c[:80]}...")
