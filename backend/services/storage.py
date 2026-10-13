"""
Multi-Tenant Qdrant Storage with per-repository collection isolation
and stale vector purge on re-indexing.
"""
import os
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_STORAGE_PATH = os.path.join(os.getcwd(), "storage", "qdrant")
os.makedirs(QDRANT_STORAGE_PATH, exist_ok=True)

_client = None

def get_qdrant_client():
    global _client
    if _client is None:
        try:
            _client = QdrantClient(path=QDRANT_STORAGE_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Qdrant storage at '{QDRANT_STORAGE_PATH}'. "
                f"The database may be locked by another process or storage is inaccessible: {str(e)}"
            ) from e
    return _client

def get_repo_collection_name(repo_id_or_url: str) -> str:
    """Generates an isolated deterministic collection name per repository."""
    digest = hashlib.sha256(repo_id_or_url.encode("utf-8")).hexdigest()[:12]
    return f"repo_{digest}"

def recreate_isolated_collection(collection_name: str):
    """
    Deletes the entire existing collection and recreates a fresh one.
    Guarantees no stale vectors remain regardless of repository size.
    """
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        # If running in local client mode, ensure open file handles are closed before rmtree on Windows
        if hasattr(client, "_client") and hasattr(client._client, "collections"):
            col = client._client.collections.get(collection_name)
            if col and hasattr(col, "close"):
                try:
                    col.close()
                except Exception:
                    pass
        client.delete_collection(collection_name=collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def store_isolated_chunks(collection_name: str, chunks: list, embeddings):
    client = get_qdrant_client()
    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload=chunk
        ))
    client.upsert(collection_name=collection_name, points=points)

def search_isolated_query(collection_name: str, embedding, limit: int = 20):
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        return []
    result = client.query_points(
        collection_name=collection_name,
        query=embedding.tolist(),
        limit=limit,
        with_payload=True
    )
    return result.points
