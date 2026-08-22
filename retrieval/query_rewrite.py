import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
REWRITE_MODEL = "openai/gpt-oss-20b"


def rewrite_query(query):
    prompt = f"""Rewrite this question into a clear, formal search query using precise terminology, as it might appear in an academic paper. Output ONLY the rewritten query, nothing else.

Question: {query}

Rewritten query:"""

    try:
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        return content.strip() if content else query
    except Exception as e:
        print(f"ERROR in rewrite_query: {e}")
        return query 