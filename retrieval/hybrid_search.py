import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

CHUNKS_PATH = "data/processed/chunks.json"
QDRANT_PATH = "data/processed/qdrant_db"
COLLECTION_NAME = "arxiv_chunks"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path=QDRANT_PATH)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25_index(chunks):
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def dense_search(query, top_k=20):
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    # returns list of chunk_ids in rank order
    return [point.payload["chunk_id"] for point in response.points]


def sparse_search(query, bm25, chunks, top_k=20):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked_indices]


def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    """
    k=60 is the standard RRF constant used in the original paper/practice.
    Score = sum of 1/(k + rank) across all lists a chunk appears in.
    """
    scores = {}

    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(query, bm25, chunks, chunk_lookup, top_k=5):
    dense_ids = dense_search(query, top_k=20)
    sparse_ids = sparse_search(query, bm25, chunks, top_k=20)

    fused = reciprocal_rank_fusion(dense_ids, sparse_ids)

    results = []
    for chunk_id, score in fused[:top_k]:
        results.append({
            "chunk_id": chunk_id,
            "rrf_score": score,
            "text": chunk_lookup[chunk_id]["text"],
            "paper_id": chunk_lookup[chunk_id]["paper_id"],
        })
    return results


if __name__ == "__main__":
    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = build_bm25_index(chunks)

    query = "what is retrieval augmented generation"
    results = hybrid_search(query, bm25, chunks, chunk_lookup)

    for r in results:
        print(f"\nRRF Score: {r['rrf_score']:.4f}")
        print(f"Paper: {r['paper_id']}")
        print(f"Text: {r['text'][:200]}...")

from retrieval.hyde import generate_hypothetical_answer

def dense_search_with_hyde(query, top_k=20):
    """
    Same as dense_search, but embeds a hypothetical answer instead of the raw query.
    """
    hypothetical = generate_hypothetical_answer(query)
    query_vector = model.encode(hypothetical).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return [point.payload["chunk_id"] for point in response.points]