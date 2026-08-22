import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found — check your .env file exists and has the key set")

client = Groq(api_key=api_key)

HYDE_MODEL = "openai/gpt-oss-20b"


def generate_hypothetical_answer(query):
    """
    Asks the LLM to write a plausible academic-style answer to the query.
    This hypothetical answer is what we'll embed for search -- not the raw query.
    """
    prompt = f"""Write a short, factual, academic-style paragraph that answers this question, as if it appeared in a research paper. Do not mention that this is hypothetical.

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )

    print("DEBUG finish_reason:", response.choices[0].finish_reason)
    print("DEBUG raw content:", repr(response.choices[0].message.content))

    content = response.choices[0].message.content
    return content.strip() if content else ""


if __name__ == "__main__":
    query = "how do models avoid making things up when answering questions"
    hypothetical = generate_hypothetical_answer(query)
    print("\nOriginal query:", query)
    print("\nHypothetical answer:\n", hypothetical)