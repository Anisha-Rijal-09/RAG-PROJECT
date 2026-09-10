from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "rag_documents"

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


def store_in_qdrant(
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Store chunks and their embeddings in Qdrant."""

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    vector_size = len(embeddings[0])

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"text": chunk},
            )
        )
        

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def search_qdrant(
    query_embedding: list[float],
    limit: int = 5,
) -> list[str]:
    """Search Qdrant for the most similar chunks."""

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
    ).points

    texts = []

    for result in results:

      if result.payload and "text" in result.payload:
         text = result.payload["text"]
         texts.append(text)

    return texts







    