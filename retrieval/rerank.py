import json
from sentence_transformers import CrossEncoder
from retrieval.hybrid_search import (
    load_chunks,
    build_bm25_index,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, candidates, top_k=5):
    """
    candidates: list of dicts with at least 'text' key
    Returns candidates reordered by cross-encoder relevance score.
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def hybrid_search_with_rerank(query, bm25, chunks, chunk_lookup, fusion_top_k=20, final_top_k=5):
    dense_ids = dense_search(query, top_k=fusion_top_k)
    sparse_ids = sparse_search(query, bm25, chunks, top_k=fusion_top_k)
    fused = reciprocal_rank_fusion(dense_ids, sparse_ids)

    # take a wider pool from fusion, then let the reranker pick the true top_k
    candidates = []
    for chunk_id, rrf_score in fused[:fusion_top_k]:
        candidates.append({
            "chunk_id": chunk_id,
            "rrf_score": rrf_score,
            "text": chunk_lookup[chunk_id]["text"],
            "paper_id": chunk_lookup[chunk_id]["paper_id"],
        })

    return rerank(query, candidates, top_k=final_top_k)


if __name__ == "__main__":
    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = build_bm25_index(chunks)

    query = "what is retrieval augmented generation"
    results = hybrid_search_with_rerank(query, bm25, chunks, chunk_lookup)

    for r in results:
        print(f"\nRerank Score: {r['rerank_score']:.4f}  (RRF was: {r['rrf_score']:.4f})")
        print(f"Paper: {r['paper_id']}")
        print(f"Text: {r['text'][:200]}...")