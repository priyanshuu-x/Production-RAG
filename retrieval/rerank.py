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



from retrieval.metadata_filter import load_metadata, filter_by_date, filter_by_paper_id

if __name__ == "__main__":
    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = build_bm25_index(chunks)
    metadata = load_metadata()

    query = "what is retrieval augmented generation"
    results = hybrid_search_with_rerank(query, bm25, chunks, chunk_lookup, fusion_top_k=20, final_top_k=10)

    # only keep papers published after 2025-08-01, as an example filter
    filtered_results = filter_by_paper_id(results, paper_ids=["2608.20316v1"])

    for r in filtered_results[:5]:
        pub_date = metadata[r["paper_id"]]["published"]
        title = metadata[r["paper_id"]]["title"]
        print(f"\nRerank Score: {r['rerank_score']:.4f}  | Published: {pub_date}")
        print(f"Title: {title}")
        print(f"Text: {r['text'][:200]}...")