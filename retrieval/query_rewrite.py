import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
REWRITE_MODEL = "openai/gpt-oss-20b"


def rewrite_query(query):
    """
    Rewrites a casual/vague question into clearer, more formal search phrasing.
    Much cheaper than HyDE -- outputs a short rewritten query, not a full paragraph.
    """
    prompt = f"""Rewrite this question into a clear, formal search query using precise terminology, as it might appear in an academic paper. Output ONLY the rewritten query, nothing else.

Question: {query}

Rewritten query:"""

    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
        reasoning_effort="low",
    )

    content = response.choices[0].message.content
    return content.strip() if content else query  # fallback to original if empty


if __name__ == "__main__":
    query = "how do models avoid making things up when answering questions"
    rewritten = rewrite_query(query)
    print("Original:", query)
    print("Rewritten:", rewritten)