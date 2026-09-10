from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks:list[str])->list[list[float]]:

    """
    Embeds a list of text chunks using the SentenceTransformer model.

    Args:
        chunks (list[str]): A list of text chunks to be embedded.

    Returns:
        list[list[float]]: A list of embeddings corresponding to the input chunks.
    """
    if not chunks:
        return []

    embeddings = model.encode(chunks,convert_to_numpy=True,normalize_embeddings=True)
    return embeddings.tolist()