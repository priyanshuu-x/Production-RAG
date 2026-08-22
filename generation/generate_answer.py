import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GENERATION_MODEL = "openai/gpt-oss-20b"


def build_context(chunks):
    """
    Numbers each chunk so the LLM can cite them as [1], [2], etc.
    Returns the formatted context string and a lookup to map citation number -> source info.
    """
    context_parts = []
    citation_map = {}

    for idx, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{idx}] {chunk['text']}")
        citation_map[idx] = {
            "paper_id": chunk["paper_id"],
            "chunk_id": chunk["chunk_id"],
        }

    context = "\n\n".join(context_parts)
    return context, citation_map


def generate_answer(query, chunks):
    context, citation_map = build_context(chunks)

    prompt = f"""Answer the question using ONLY the numbered sources below. Cite sources inline using [1], [2], etc. after each claim. If the sources don't contain enough information, say so clearly instead of guessing.

Sources:
{context}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        answer = content.strip() if content else "I couldn't generate an answer. Please try again."
    except Exception as e:
        answer = "Something went wrong while generating the answer. Please try again shortly."
        print(f"ERROR in generate_answer: {e}")  # logged server-side for debugging

    return answer, citation_map


if __name__ == "__main__":
    from retrieval.rerank import hybrid_search_with_rerank
    from retrieval.hybrid_search import load_chunks, build_bm25_index

    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = build_bm25_index(chunks)

    query = "what is retrieval augmented generation"
    top_chunks = hybrid_search_with_rerank(query, bm25, chunks, chunk_lookup, final_top_k=5)

    answer, citation_map = generate_answer(query, top_chunks)

    print("Question:", query)
    print("\nAnswer:\n", answer)

    print("\n\nSources:")
    for num, info in citation_map.items():
        print(f"[{num}] Paper: {info['paper_id']}  (chunk: {info['chunk_id']})")