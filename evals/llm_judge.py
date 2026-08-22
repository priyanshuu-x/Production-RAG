import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "openai/gpt-oss-20b"


def _ask_judge(prompt):
    """Sends a judge prompt, extracts a 0-1 score from the response."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
        reasoning_effort="low",
    )
    content = response.choices[0].message.content or ""

    match = re.search(r"score:\s*([0-1](?:\.\d+)?)", content.lower())
    score = float(match.group(1)) if match else 0.0
    return score, content.strip()


def judge_faithfulness(answer, context_text):
    prompt = f"""Given the CONTEXT and ANSWER below, determine if every claim in the ANSWER is supported by the CONTEXT (no unsupported/hallucinated claims).

Rate faithfulness from 0 (completely unsupported) to 1 (fully supported).
Respond in this exact format:
Score: <number between 0 and 1>
Reason: <one sentence>

Context:
{context_text}

Answer:
{answer}"""
    return _ask_judge(prompt)


def judge_answer_relevance(question, answer):
    prompt = f"""Given the QUESTION and ANSWER below, determine how directly the ANSWER addresses the QUESTION.

Rate relevance from 0 (does not address the question at all) to 1 (directly and fully addresses it).
Respond in this exact format:
Score: <number between 0 and 1>
Reason: <one sentence>

Question:
{question}

Answer:
{answer}"""
    return _ask_judge(prompt)


def judge_context_precision(question, context_text):
    prompt = f"""Given the QUESTION and the retrieved CONTEXT below, determine what fraction of the CONTEXT is actually relevant/useful for answering the QUESTION (not noise).

Rate precision from 0 (entirely irrelevant) to 1 (entirely relevant).
Respond in this exact format:
Score: <number between 0 and 1>
Reason: <one sentence>

Question:
{question}

Context:
{context_text}"""
    return _ask_judge(prompt)


def judge_context_recall(context_text, expected_topics):
    prompt = f"""Given the retrieved CONTEXT and the EXPECTED TOPICS that should be covered, determine what fraction of the EXPECTED TOPICS are actually present in the CONTEXT.

Rate recall from 0 (none of the expected topics are covered) to 1 (all expected topics are covered).
Respond in this exact format:
Score: <number between 0 and 1>
Reason: <one sentence>

Expected topics:
{expected_topics}

Context:
{context_text}"""
    return _ask_judge(prompt)