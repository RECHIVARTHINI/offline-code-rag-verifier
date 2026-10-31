"""
Local User Authentication Service (Zero-Trust PBKDF2).
Supports User Registration, Login, Session Token Issuance, and Role-Based Access Control (Manager vs Developer).
Stores credentials securely in storage/users.json with salted PBKDF2-SHA256 hashes.
"""
import os
import json
import secrets
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
USERS_FILE = STORAGE_DIR / "users.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        iterations=100000
    )
    return key.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a candidate password against stored salt and hash."""
    candidate_hash, _ = _hash_password(password, salt)
    return secrets.compare_digest(candidate_hash, stored_hash)


def load_users() -> Dict[str, Any]:
    """Loads all registered users from storage/users.json."""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Initialize with default manager and developer accounts
    initial_users = {}
    
    # 1. Default Manager Account
    mgr_hash, mgr_salt = _hash_password("admin123")
    initial_users["manager"] = {
        "id": "usr_manager_001",
        "username": "manager",
        "email": "manager@company.internal",
        "role": "manager",
        "password_hash": mgr_hash,
        "salt": mgr_salt,
        "token": "rag_token_mgr_" + secrets.token_hex(16),
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }

    # 2. Default Developer Account
    dev_hash, dev_salt = _hash_password("dev123")
    initial_users["dev"] = {
        "id": "usr_dev_002",
        "username": "dev",
        "email": "developer@company.internal",
        "role": "developer",
        "password_hash": dev_hash,
        "salt": dev_salt,
        "token": "rag_token_dev_" + secrets.token_hex(16),
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }

    save_users(initial_users)
    return initial_users


def save_users(users: Dict[str, Any]):
    """Persists users to storage/users.json."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, email: Optional[str] = None, role: str = "developer") -> Dict[str, Any]:
    """Registers a new user with role ('manager' or 'developer')."""
    clean_username = username.strip().lower()
    if len(clean_username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if not clean_username.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Username can only contain alphanumeric characters, underscores, and hyphens.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    clean_role = role.strip().lower()
    if clean_role not in {"manager", "developer"}:
        clean_role = "developer"

    clean_email = email.strip().lower() if email else f"{clean_username}@company.internal"

    users = load_users()
    if clean_username in users:
        raise ValueError(f"Username '{clean_username}' is already taken.")

    # Check email uniqueness
    for u in users.values():
        if u.get("email") == clean_email:
            raise ValueError(f"Email '{clean_email}' is already registered.")

    p_hash, salt = _hash_password(password)
    user_token = f"rag_token_{clean_username}_{secrets.token_hex(16)}"
    user_id = f"usr_{secrets.token_hex(6)}"

    new_user = {
        "id": user_id,
        "username": clean_username,
        "email": clean_email,
        "role": clean_role,
        "password_hash": p_hash,
        "salt": salt,
        "token": user_token,
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }

    users[clean_username] = new_user
    save_users(users)

    return {
        "status": "SUCCESS",
        "message": f"User '{clean_username}' registered successfully.",
        "token": user_token,
        "user": {
            "id": user_id,
            "username": clean_username,
            "email": clean_email,
            "role": clean_role
        }
    }


def login_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticates user credentials and generates a fresh session token."""
    clean_username = username.strip().lower()
    users = load_users()

    # Search by username or email
    user = None
    target_key = None
    for k, u in users.items():
        if k == clean_username or u.get("email") == clean_username:
            user = u
            target_key = k
            break

    if not user:
        raise ValueError("Invalid username or password.")

    if not _verify_password(password, user["password_hash"], user["salt"]):
        raise ValueError("Invalid username or password.")

    # Issue fresh or verify active token
    new_token = f"rag_token_{user['username']}_{secrets.token_hex(16)}"
    user["token"] = new_token
    user["last_login"] = datetime.now().isoformat()
    users[target_key] = user
    save_users(users)

    return {
        "status": "SUCCESS",
        "message": f"Welcome back, {user['username']}!",
        "token": new_token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    }


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Resolves active user dict from Bearer token."""
    if not token or not token.strip():
        return None

    clean_token = token.strip()
    
    # Check if system token matches RAG_AUTH_TOKEN
    system_token = os.getenv("RAG_AUTH_TOKEN")
    if system_token and clean_token == system_token.strip():
        return {
            "id": "usr_system_root",
            "username": "admin",
            "email": "admin@system.internal",
            "role": "manager"
        }

    users = load_users()
    for u in users.values():
        if u.get("token") == clean_token:
            return {
                "id": u["id"],
                "username": u["username"],
                "email": u["email"],
                "role": u["role"]
            }

    return None
