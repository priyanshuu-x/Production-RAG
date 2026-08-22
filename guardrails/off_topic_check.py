import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
CHECK_MODEL = "openai/gpt-oss-20b"


def is_on_topic(question):
    prompt = f"""Is the following question plausibly related to AI, machine learning, retrieval-augmented generation (RAG), or academic research in computer science? Answer with only "yes" or "no".

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=CHECK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
        reasoning_effort="low",
    )

    content = response.choices[0].message.content
    answer = content.strip().lower() if content else "yes"

    return "yes" in answer