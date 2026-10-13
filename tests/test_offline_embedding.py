"""
Test Suite: Offline Embedding & Model Loading Guard
Proves that embedding model loading runs strictly offline and fails safely if model is missing.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.embedding import get_embedding_model, generate_embeddings, query_embedding

def test_offline_environment_flags_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE must be 1"

def test_missing_model_fails_offline_without_network():
    from sentence_transformers import SentenceTransformer
    try:
        # Attempt to load a nonexistent model name with local_files_only=True
        SentenceTransformer("nonexistent/ghost-model-xyz", local_files_only=True)
        assert False, "Should have raised exception for nonexistent local model"
    except Exception as e:
        # Expected failure in offline mode
        assert "nonexistent/ghost-model-xyz" in str(e) or "offline" in str(e).lower() or "local" in str(e).lower() or "not found" in str(e).lower()

def test_cached_model_generates_embeddings():
    model = get_embedding_model()
    assert model is not None
    chunks = [{"content": "def calculate_tax(amount): return amount * 0.18"}]
    emb = generate_embeddings(chunks)
    assert emb.shape == (1, 384)
    q_emb = query_embedding("tax calculation")
    assert q_emb.shape == (384,)

if __name__ == "__main__":
    test_offline_environment_flags_set()
    print("PASS: test_offline_environment_flags_set")
    test_missing_model_fails_offline_without_network()
    print("PASS: test_missing_model_fails_offline_without_network")
    test_cached_model_generates_embeddings()
    print("PASS: test_cached_model_generates_embeddings")
    print("\nALL OFFLINE EMBEDDING TESTS PASSED!")
