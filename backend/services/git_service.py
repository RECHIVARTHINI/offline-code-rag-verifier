import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
from git import Repo

def is_path_allowlisted(local_path: Path) -> bool:
    """
    Validates if local path is within explicitly allowlisted directories.
    Configurable via RAG_ALLOWED_LOCAL_REPO_ROOTS (comma-separated).
    Defaults to current workspace root.
    """
    allowed_roots_env = os.getenv("RAG_ALLOWED_LOCAL_REPO_ROOTS")
    resolved_path = local_path.resolve()
    
    if allowed_roots_env:
        roots = [Path(r.strip()).resolve() for r in allowed_roots_env.split(",") if r.strip()]
        for root in roots:
            try:
                resolved_path.relative_to(root)
                return True
            except ValueError:
                continue
        return False
    else:
        # Default allowlist: allow only within current workspace directory
        cwd = Path(os.getcwd()).resolve()
        try:
            resolved_path.relative_to(cwd)
            return True
        except ValueError:
            return False

import re
from urllib.parse import urlparse

def is_local_only_mode() -> bool:
    """Checks if remote cloning is disabled via environment variable."""
    val = os.getenv("RAG_LOCAL_ONLY_MODE", "").strip().lower()
    val_disable = os.getenv("RAG_DISABLE_REMOTE_CLONE", "").strip().lower()
    return val in ("1", "true", "yes") or val_disable in ("1", "true", "yes")

def extract_git_host(repo_url: str) -> Optional[str]:
    """Extracts the hostname from an HTTP(S), SSH, or SCP-style git URL."""
    url = repo_url.strip()
    if url.startswith(("http://", "https://", "ssh://")):
        parsed = urlparse(url)
        return parsed.hostname.lower() if parsed.hostname else None
    
    # SCP-style: git@github.com:owner/repo.git or user@host.com:path
    match = re.match(r'^(?:[^@]+@)?([^:/]+)(?::|/)', url)
    if match:
        return match.group(1).lower()
    return None

def is_git_host_allowed(repo_url: str) -> bool:
    """
    Validates if the remote Git host is in the allowlist.
    Configurable via RAG_ALLOWED_GIT_HOSTS (comma-separated).
    Defaults to github.com, gitlab.com, bitbucket.org.
    """
    if is_local_only_mode():
        return False
        
    allowed_hosts_env = os.getenv("RAG_ALLOWED_GIT_HOSTS")
    if allowed_hosts_env is not None:
        allowed_hosts = [h.strip().lower() for h in allowed_hosts_env.split(",") if h.strip()]
    else:
        allowed_hosts = ["github.com", "gitlab.com", "bitbucket.org"]
        
    if not allowed_hosts:
        return False
        
    if "*" in allowed_hosts:
        return True
        
    host = extract_git_host(repo_url)
    if not host:
        return False
        
    for allowed in allowed_hosts:
        if host == allowed or host.endswith("." + allowed):
            return True
    return False

def clone_repo(repo_url: str) -> Path:
    repo_url_clean = repo_url.strip()
    local_path = Path(repo_url_clean)
    
    # 1. Local path validation with strict allowlist check
    if local_path.exists() and local_path.is_dir():
        if not is_path_allowlisted(local_path):
            raise PermissionError(
                f"Local repository path '{repo_url_clean}' is not in the allowlisted directories. "
                f"Configure RAG_ALLOWED_LOCAL_REPO_ROOTS or place repositories in the workspace directory."
            )
        return local_path

    # Check local-only mode
    if is_local_only_mode():
        raise PermissionError(
            "Remote repository cloning is disabled in local-only mode (RAG_LOCAL_ONLY_MODE=1). "
            "Only allowlisted local repository paths are permitted."
        )

    # 2. Remote Git Host Allowlist Validation
    host = extract_git_host(repo_url_clean)
    if not is_git_host_allowed(repo_url_clean):
        allowed_hosts_display = os.getenv("RAG_ALLOWED_GIT_HOSTS", "github.com, gitlab.com, bitbucket.org")
        raise PermissionError(
            f"Remote Git host '{host or 'unknown'}' is not in the approved Git host allowlist: [{allowed_hosts_display}]. "
            f"Configure RAG_ALLOWED_GIT_HOSTS to allow this host."
        )
        
    # 3. Remote Git Clone (Strict: no arbitrary temp globbing on failure)
    temp_dir = Path(tempfile.mkdtemp(prefix="codebase-rag-"))
    try:
        Repo.clone_from(repo_url_clean, temp_dir)
        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Git clone failed for repository '{repo_url_clean}': {str(e)}"
        ) from e
