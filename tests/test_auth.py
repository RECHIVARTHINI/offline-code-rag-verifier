"""
Test Suite: Bearer Token Authentication (HTTPBearer)
Validates HTTP 401 on missing, invalid, or unconfigured tokens on protected endpoints.
"""
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app

client = TestClient(app)

def test_missing_auth_token_returns_401():
    os.environ["RAG_AUTH_TOKEN"] = "test-secret-token-12345"
    
    # 1. POST /api/v1/upload without header
    res_upload = client.post("/api/v1/upload", json={"repo_url": "https://github.com/test/repo"})
    assert res_upload.status_code == 401, f"Expected 401, got {res_upload.status_code}"
    assert "Missing Bearer authentication token" in res_upload.json()["detail"]

    # 2. POST /api/v1/userquery without header
    res_query = client.post("/api/v1/userquery", json={"query": "test query"})
    assert res_query.status_code == 401, f"Expected 401, got {res_query.status_code}"
    assert "Missing Bearer authentication token" in res_query.json()["detail"]

def test_invalid_auth_token_returns_401():
    os.environ["RAG_AUTH_TOKEN"] = "test-secret-token-12345"
    headers = {"Authorization": "Bearer wrong-token-xyz"}
    
    res_upload = client.post("/api/v1/upload", json={"repo_url": "https://github.com/test/repo"}, headers=headers)
    assert res_upload.status_code == 401, f"Expected 401, got {res_upload.status_code}"
    assert "Invalid Bearer authentication token" in res_upload.json()["detail"]

    res_query = client.post("/api/v1/userquery", json={"query": "test query"}, headers=headers)
    assert res_query.status_code == 401, f"Expected 401, got {res_query.status_code}"
    assert "Invalid Bearer authentication token" in res_query.json()["detail"]

def test_unconfigured_auth_token_env_returns_401():
    old_token = os.environ.pop("RAG_AUTH_TOKEN", None)
    try:
        headers = {"Authorization": "Bearer any-token"}
        res = client.post("/api/v1/upload", json={"repo_url": "https://github.com/test/repo"}, headers=headers)
        assert res.status_code == 401
        assert "RAG_AUTH_TOKEN environment variable is not configured" in res.json()["detail"]
    finally:
        if old_token:
            os.environ["RAG_AUTH_TOKEN"] = old_token

def test_valid_auth_token_passes():
    os.environ["RAG_AUTH_TOKEN"] = "test-secret-token-12345"
    headers = {"Authorization": "Bearer test-secret-token-12345"}
    
    # Validation passes auth check (fails at repo validation or 404 repo, not 401)
    res_query = client.post("/api/v1/userquery", json={"query": "test", "repo_id": "nonexistent_repo"}, headers=headers)
    assert res_query.status_code in [400, 404] # Passed auth!

if __name__ == "__main__":
    test_missing_auth_token_returns_401()
    print("PASS: test_missing_auth_token_returns_401")
    test_invalid_auth_token_returns_401()
    print("PASS: test_invalid_auth_token_returns_401")
    test_unconfigured_auth_token_env_returns_401()
    print("PASS: test_unconfigured_auth_token_env_returns_401")
    test_valid_auth_token_passes()
    print("PASS: test_valid_auth_token_passes")
    print("\nALL AUTHENTICATION TESTS PASSED!")
