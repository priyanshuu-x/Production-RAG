import json
from retrieval.hybrid_search import load_chunks, build_bm25_index
from retrieval.rerank import hybrid_search_with_rerank
from generation.generate_answer import generate_answer
from evals.llm_judge import (
    judge_faithfulness,
    judge_answer_relevance,
    judge_context_precision,
    judge_context_recall,
)

EVAL_SET_PATH = "evals/eval_set.json"


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_generation_eval():
    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = build_bm25_index(chunks)

    eval_set = load_eval_set()

    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []
    recall_scores = []

    for item in eval_set:
        question = item["question"]
        expected_topics = item["expected_topics"]

        results = hybrid_search_with_rerank(question, bm25, chunks, chunk_lookup, final_top_k=5)
        context_text = "\n\n".join(r["text"] for r in results)

        answer, _ = generate_answer(question, results)

        faith_score, faith_reason = judge_faithfulness(answer, context_text)
        rel_score, rel_reason = judge_answer_relevance(question, answer)
        prec_score, prec_reason = judge_context_precision(question, context_text)
        rec_score, rec_reason = judge_context_recall(context_text, expected_topics)

        faithfulness_scores.append(faith_score)
        relevance_scores.append(rel_score)
        precision_scores.append(prec_score)
        recall_scores.append(rec_score)

        print(f"\nQuestion: {question}")
        print(f"  Faithfulness: {faith_score:.2f} — {faith_reason}")
        print(f"  Answer Relevance: {rel_score:.2f} — {rel_reason}")
        print(f"  Context Precision: {prec_score:.2f} — {prec_reason}")
        print(f"  Context Recall: {rec_score:.2f} — {rec_reason}")

    print(f"\n=== Overall Averages ===")
    print(f"Faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.2f}")
    print(f"Answer Relevance: {sum(relevance_scores)/len(relevance_scores):.2f}")
    print(f"Context Precision: {sum(precision_scores)/len(precision_scores):.2f}")
    print(f"Context Recall: {sum(recall_scores)/len(recall_scores):.2f}")


if __name__ == "__main__":
    run_generation_eval()