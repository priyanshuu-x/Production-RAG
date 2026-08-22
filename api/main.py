from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from caching.cache import get_cached_response, set_cached_response
from api.rate_limit import check_rate_limit

from retrieval.hybrid_search import load_chunks, build_bm25_index
from retrieval.rerank import hybrid_search_with_rerank
from retrieval.metadata_filter import load_metadata, filter_by_paper_id
from generation.generate_answer import generate_answer

from guardrails.input_guardrails import contains_pii, contains_injection_attempt
from guardrails.off_topic_check import is_on_topic
from guardrails.output_guardrails import has_citation


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
def query(request: QueryRequest, _: None = Depends(check_rate_limit)):
    has_pii, pii_type = contains_pii(request.question)
    if has_pii:
        raise HTTPException(status_code=400, detail=f"Question appears to contain personal information ({pii_type}). Please rephrase without it.")

    is_injection, matched_phrase = contains_injection_attempt(request.question)
    if is_injection:
        raise HTTPException(status_code=400, detail="This question could not be processed.")

    if not is_on_topic(request.question):
        raise HTTPException(status_code=400, detail="This question doesn't appear to be related to the papers in this system. Try asking about RAG, retrieval, or related AI/ML topics.")

    cached = get_cached_response(request.question)
    if cached:
        return QueryResponse(**cached)

    results = hybrid_search_with_rerank(
        request.question, bm25, chunks, chunk_lookup, final_top_k=request.top_k,
    )

    answer, citation_map = generate_answer(request.question, results)

    if not has_citation(answer):
        answer += "\n\n*(Note: this answer may not be fully grounded in the source papers.)*"

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