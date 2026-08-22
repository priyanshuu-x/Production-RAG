import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CHUNKS_PATH = "data/processed/chunks.json"
QDRANT_PATH = "data/processed/qdrant_db"
COLLECTION_NAME = "arxiv_chunks"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_collection(client, vector_size):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def embed_and_upload(chunks, client):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "text": chunk["text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main():
    chunks = load_chunks()
    client = QdrantClient(path=QDRANT_PATH)

    vector_size = model.get_sentence_embedding_dimension()
    build_collection(client, vector_size)

    embed_and_upload(chunks, client)

    print(f"Embedded and stored {len(chunks)} chunks in Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()