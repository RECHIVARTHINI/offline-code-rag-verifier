"""
100% Offline SentenceTransformer with strict local-only enforcement.
"""
import os
from typing import List, Any
import numpy as np

# Strictly enforce offline execution in the environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Embedding model 'BAAI/bge-small-en-v1.5' is not available in the local cache: {str(e)}. "
                "Runtime internet downloads are strictly disabled for offline security. "
                "To pre-download the model into your local cache, run: "
                "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')\" "
                "in an environment with internet access before starting offline mode."
            ) from e
    return _model

def generate_embeddings(chunks: List[Any]) -> np.ndarray:
    model = get_embedding_model()
    texts = [
        f"{chunk.get('file_path', '')} :: {chunk.get('symbol', '')}\n{chunk.get('content', '')}"
        if isinstance(chunk, dict) and ("file_path" in chunk or "symbol" in chunk)
        else (chunk["content"] if isinstance(chunk, dict) else str(chunk))
        for chunk in chunks
    ]
    return np.array(model.encode(texts, normalize_embeddings=True))

def query_embedding(query: str) -> np.ndarray:
    model = get_embedding_model()
    return np.array(model.encode(query, normalize_embeddings=True))
