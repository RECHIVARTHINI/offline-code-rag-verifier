"""
Test Suite: Qdrant Isolation & Stale Vector Purge on Re-index
Proves that re-indexing a repository wipes old vectors and that collections are isolated.
"""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.storage import (
    get_qdrant_client,
    recreate_isolated_collection,
    store_isolated_chunks,
    search_isolated_query,
    get_repo_collection_name
)

from unittest.mock import patch

def test_recreate_purges_stale_vectors():
    client = get_qdrant_client()
    repo_url = "https://github.com/org/test-stale-purge"
    collection_name = get_repo_collection_name(repo_url)

    # Initial indexing: 3 chunks
    chunks_v1 = [
        {"file_path": "old_file_1.py", "symbol": "old_func1", "content": "def old_func1(): pass"},
        {"file_path": "old_file_2.py", "symbol": "old_func2", "content": "def old_func2(): pass"},
        {"file_path": "old_file_3.py", "symbol": "old_func3", "content": "def old_func3(): pass"}
    ]
    emb_v1 = np.ones((3, 384), dtype=np.float32)
    
    recreate_isolated_collection(collection_name)
    store_isolated_chunks(collection_name, chunks_v1, emb_v1)
    
    # Verify initial count
    cnt1 = client.count(collection_name=collection_name).count
    assert cnt1 == 3, f"Expected 3 points, got {cnt1}"

    # Re-indexing: Only 1 updated chunk
    chunks_v2 = [
        {"file_path": "new_file.py", "symbol": "new_func", "content": "def new_func(): pass"}
    ]
    emb_v2 = np.ones((1, 384), dtype=np.float32)
    
    recreate_isolated_collection(collection_name)
    store_isolated_chunks(collection_name, chunks_v2, emb_v2)

    # Verify stale chunks were purged completely via delete_collection
    cnt2 = client.count(collection_name=collection_name).count
    assert cnt2 == 1, f"Expected 1 point after re-index purge, got {cnt2}"
    
    res = client.scroll(collection_name=collection_name, limit=10, with_payload=True)[0]
    assert res[0].payload["symbol"] == "new_func"

def test_cross_repository_isolation():
    repo_a = "https://github.com/org/repo-a"
    repo_b = "https://github.com/org/repo-b"

    col_a = get_repo_collection_name(repo_a)
    col_b = get_repo_collection_name(repo_b)
    assert col_a != col_b

    chunks_a = [{"file_path": "secret_a.py", "symbol": "vault_a", "content": "SECRET KEY A"}]
    chunks_b = [{"file_path": "secret_b.py", "symbol": "vault_b", "content": "SECRET KEY B"}]

    recreate_isolated_collection(col_a)
    store_isolated_chunks(col_a, chunks_a, np.ones((1, 384), dtype=np.float32))

    recreate_isolated_collection(col_b)
    store_isolated_chunks(col_b, chunks_b, np.ones((1, 384), dtype=np.float32))

    # Query col_a
    pts_a = search_isolated_query(col_a, np.ones(384, dtype=np.float32))
    symbols_in_a = [p.payload["symbol"] for p in pts_a]
    assert "vault_a" in symbols_in_a
    assert "vault_b" not in symbols_in_a

def test_qdrant_initialization_fails_clearly_on_lock_error():
    import backend.services.storage as storage_mod
    original_client = storage_mod._client
    try:
        storage_mod._client = None
        with patch("backend.services.storage.QdrantClient", side_effect=PermissionError("Folder locked by process 999")):
            try:
                storage_mod.get_qdrant_client()
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Failed to initialize Qdrant storage" in str(e)
                assert "Folder locked by process 999" in str(e)
    finally:
        storage_mod._client = original_client

if __name__ == "__main__":
    test_recreate_purges_stale_vectors()
    print("PASS: test_recreate_purges_stale_vectors")
    test_cross_repository_isolation()
    print("PASS: test_cross_repository_isolation")
    test_qdrant_initialization_fails_clearly_on_lock_error()
    print("PASS: test_qdrant_initialization_fails_clearly_on_lock_error")
    print("\nALL STORAGE ISOLATION & STALE PURGE TESTS PASSED!")
