from fastapi import FastAPI
from pydantic import BaseModel
from caching.cache import get_cached_response, set_cached_response

from retrieval.hybrid_search import load_chunks, build_bm25_index
from retrieval.rerank import hybrid_search_with_rerank
from retrieval.metadata_filter import load_metadata, filter_by_paper_id
from generation.generate_answer import generate_answer

app = FastAPI(
    title="PaperMind",
    description="Ask questions, get cited answers.",
    version="1.0.0",
)

chunks = load_chunks()
chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
bm25 = build_bm25_index(chunks)
metadata = load_metadata()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    citation_number: int
    paper_id: str
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]


@app.get("/health")
def health_check():
    return {"status": "ok", "chunks_loaded": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    cached = get_cached_response(request.question)
    if cached:
        return QueryResponse(**cached)

    results = hybrid_search_with_rerank(
        request.question,
        bm25,
        chunks,
        chunk_lookup,
        final_top_k=request.top_k,
    )

    answer, citation_map = generate_answer(request.question, results)

    sources = [
        SourceInfo(citation_number=num, paper_id=info["paper_id"], chunk_id=info["chunk_id"])
        for num, info in citation_map.items()
    ]

    response_data = {
        "question": request.question,
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
    }

    set_cached_response(request.question, response_data)

    return QueryResponse(**response_data)