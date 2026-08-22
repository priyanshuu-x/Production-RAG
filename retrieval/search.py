from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_PATH = "data/processed/qdrant_db"
COLLECTION_NAME = "arxiv_chunks"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path=QDRANT_PATH)


def search(query, top_k=5):
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return response.points


if __name__ == "__main__":
    query = "what is retrieval augmented generation"
    results = search(query)

    for r in results:
        print(f"\nScore: {r.score:.3f}")
        print(f"Paper: {r.payload['paper_id']}")
        print(f"Text: {r.payload['text'][:200]}...")