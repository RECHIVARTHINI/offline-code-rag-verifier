"""
Test Suite: Git Host Allowlist and Local-Only Mode Security
Validates:
1. Rejection of unapproved remote Git hosts (e.g. internal IP / unapproved domain).
2. Acceptance of approved hosts (github.com, gitlab.com, bitbucket.org or configured).
3. Local-only mode enforcement (RAG_LOCAL_ONLY_MODE=1 / RAG_DISABLE_REMOTE_CLONE=1).
4. Endpoint HTTP 403 response on unapproved hosts or local-only mode violation.
"""
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.git_service import extract_git_host, is_git_host_allowed, clone_repo
from main import app

client = TestClient(app)

def test_extract_git_host():
    assert extract_git_host("https://github.com/org/repo.git") == "github.com"
    assert extract_git_host("http://gitlab.com/group/project") == "gitlab.com"
    assert extract_git_host("git@github.com:org/repo.git") == "github.com"
    assert extract_git_host("ssh://git@internal-git.corp.net/repo.git") == "internal-git.corp.net"
    assert extract_git_host("http://192.168.1.50/repo.git") == "192.168.1.50"

def test_host_allowlist_default():
    os.environ.pop("RAG_ALLOWED_GIT_HOSTS", None)
    os.environ.pop("RAG_LOCAL_ONLY_MODE", None)
    os.environ.pop("RAG_DISABLE_REMOTE_CLONE", None)

    assert is_git_host_allowed("https://github.com/org/repo") is True
    assert is_git_host_allowed("git@gitlab.com:org/repo.git") is True
    assert is_git_host_allowed("https://bitbucket.org/org/repo") is True
    
    # Internal or unapproved hosts must be rejected
    assert is_git_host_allowed("http://192.168.1.100/secret-repo.git") is False
    assert is_git_host_allowed("http://localhost:8080/repo.git") is False
    assert is_git_host_allowed("https://evil-host.com/payload.git") is False

def test_custom_host_allowlist():
    os.environ["RAG_ALLOWED_GIT_HOSTS"] = "git.internal-corp.com,github.com"
    try:
        assert is_git_host_allowed("https://git.internal-corp.com/repo") is True
        assert is_git_host_allowed("https://github.com/repo") is True
        assert is_git_host_allowed("https://gitlab.com/repo") is False
    finally:
        os.environ.pop("RAG_ALLOWED_GIT_HOSTS", None)

def test_local_only_mode():
    os.environ["RAG_LOCAL_ONLY_MODE"] = "1"
    try:
        assert is_git_host_allowed("https://github.com/repo") is False
        try:
            clone_repo("https://github.com/repo")
            assert False, "Should have raised PermissionError in local-only mode"
        except PermissionError as e:
            assert "local-only mode" in str(e).lower()
    finally:
        os.environ.pop("RAG_LOCAL_ONLY_MODE", None)

def test_endpoint_unapproved_host_returns_403():
    os.environ["RAG_AUTH_TOKEN"] = "test-token"
    os.environ["RAG_ALLOWED_GIT_HOSTS"] = "github.com"
    headers = {"Authorization": "Bearer test-token"}
    
    res = client.post(
        "/api/v1/upload",
        json={"repo_url": "http://192.168.1.100/unapproved-repo.git"},
        headers=headers
    )
    assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"
    assert "not in the approved Git host allowlist" in res.json()["detail"]

if __name__ == "__main__":
    test_extract_git_host()
    print("PASS: test_extract_git_host")
    test_host_allowlist_default()
    print("PASS: test_host_allowlist_default")
    test_custom_host_allowlist()
    print("PASS: test_custom_host_allowlist")
    test_local_only_mode()
    print("PASS: test_local_only_mode")
    test_endpoint_unapproved_host_returns_403()
    print("PASS: test_endpoint_unapproved_host_returns_403")
    print("\nALL GIT HOST ALLOWLIST TESTS PASSED!")
