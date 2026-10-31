"""
FastAPI Server with Bearer-Token Authentication (HTTPBearer),
Multi-Tenant Repository Isolation, Truly Offline Models,
Fail-Closed Zero-Trust Verification, Project Manager Studio,
and Incremental Git Code Update Mechanism.
"""
import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.services.git_service import clone_repo
from backend.services.scanner import scanner
from backend.services.chunker import chunker
from backend.services.embedding import generate_embeddings, query_embedding
from backend.services.storage import (
    get_qdrant_client,
    recreate_isolated_collection,
    store_isolated_chunks,
    search_isolated_query,
    get_repo_collection_name
)
from backend.services.reranker import rerank_results
from backend.services.ollama_service import generate_code_answer
from backend.services.verification import VerificationEngine
from backend.services.project_manager import (
    get_all_projects,
    get_project,
    get_project_files_and_symbols,
    delete_project,
    sync_project_incremental,
    ingest_new_project,
    register_project_ingestion,
    load_projects_registry
)
from backend.services.auth_service import (
    register_user,
    login_user,
    get_user_by_token,
    load_users
)

app = FastAPI(
    title="Evidence-Backed Local Code RAG with Deterministic Verification",
    version="2.1.0"
)

security = HTTPBearer(auto_error=False)
REPO_ENGINES: Dict[str, VerificationEngine] = {}

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


def verify_bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """
    Strict Bearer Token Authentication.
    Validates either the root RAG_AUTH_TOKEN environment secret or an active registered user session token.
    Returns HTTP 401 if missing or invalid.
    """
    expected_token = os.getenv("RAG_AUTH_TOKEN")
    if not expected_token or not expected_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Server configuration error: RAG_AUTH_TOKEN environment variable is not configured.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    token_str = credentials.credentials.strip()

    # 1. Check system root token (for automated tests, CI, and system admin)
    if token_str == expected_token.strip():
        return {
            "id": "usr_system_root",
            "username": "admin",
            "email": "admin@system.internal",
            "role": "manager"
        }

    # 2. Check registered user session token
    user = get_user_by_token(token_str)
    if user:
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid Bearer authentication token.",
        headers={"WWW-Authenticate": "Bearer"}
    )


def verify_manager_role(current_user: Dict[str, Any] = Depends(verify_bearer_token)) -> Dict[str, Any]:
    """
    Enforces Role-Based Access Control (RBAC):
    Only users with 'manager' role (or system admin) are authorized to ingest, sync, or delete repositories.
    Developers have read-only access for querying and inspecting codebase evidence.
    """
    if current_user.get("role") != "manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Only users with 'manager' role are permitted to add, sync, or delete repositories."
        )
    return current_user


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=4, max_length=100, description="Password")
    email: Optional[str] = Field(None, description="Optional email address")
    role: Optional[str] = Field("developer", description="Role: 'manager' or 'developer'")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class IngestionRequest(BaseModel):
    repo_url: str = Field(..., description="Git URL or allowlisted local repository path")
    name: Optional[str] = Field(None, description="Optional display name for project")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Technical query regarding the codebase")
    repo_id: Optional[str] = Field(None, description="Repository identifier (defaults to active repo)")


# ================= USER AUTHENTICATION API =================

@app.post('/api/v1/auth/register', status_code=201)
def api_register_user(req: RegisterRequest):
    """Registers a new user and issues an authenticated session token."""
    try:
        res = register_user(
            username=req.username,
            password=req.password,
            email=req.email,
            role=req.role or "developer"
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post('/api/v1/auth/login')
def api_login_user(req: LoginRequest):
    """Authenticates user credentials and returns session token."""
    try:
        res = login_user(username=req.username, password=req.password)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.get('/api/v1/auth/me')
def api_get_current_user(current_user: Dict[str, Any] = Depends(verify_bearer_token)):
    """Returns profile for currently authenticated user."""
    return {
        "status": "SUCCESS",
        "user": current_user
    }


# ================= WEB UI ROUTES =================

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0"
}

@app.get('/', response_class=HTMLResponse)
def root_endpoint(request: Request):
    """
    Serves the modern Developer Chat & Manager Web UI for browser requests,
    while returning JSON health status for API callers / automated tests.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        if INDEX_HTML_PATH.exists():
            return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"), headers=NO_CACHE_HEADERS)
    
    # Return JSON health if requested via curl, TestClient, or application/json
    return {
        "status": "online",
        "system": "Evidence-Backed Local Code RAG with Deterministic Verification",
        "isolated_repositories": list(REPO_ENGINES.keys())
    }


@app.get('/manager', response_class=HTMLResponse)
def manager_view():
    """Direct route for Manager Dashboard."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"), headers=NO_CACHE_HEADERS)
    return HTMLResponse("<h1>Manager Dashboard Loading...</h1>", headers=NO_CACHE_HEADERS)


@app.get('/chat', response_class=HTMLResponse)
def chat_view():
    """Direct route for Developer Chat."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"), headers=NO_CACHE_HEADERS)
    return HTMLResponse("<h1>Developer Chat Loading...</h1>", headers=NO_CACHE_HEADERS)


# ================= PROJECT MANAGEMENT & GIT SYNC API =================

@app.get('/api/v1/projects')
def list_projects():
    """Returns all managed repositories, commit metadata, and AST chunk statistics."""
    return get_all_projects()


@app.get('/api/v1/projects/{repo_id}')
def get_project_by_id(repo_id: str):
    """Retrieves full metadata for a specific repository project."""
    proj = get_project(repo_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{repo_id}' not found.")
    return proj


@app.get('/api/v1/projects/{repo_id}/files')
def get_project_files(repo_id: str):
    """Returns indexed files, symbols, and line numbers for a project."""
    return get_project_files_and_symbols(repo_id)


@app.post('/api/v1/projects', status_code=201, dependencies=[Depends(verify_manager_role)])
def create_project_endpoint(req: IngestionRequest):
    """Clones or scans repository, AST-chunks, embeds, and registers project in studio."""
    try:
        res = ingest_new_project(
            repo_url=req.repo_url,
            display_name=req.name,
            repo_engines=REPO_ENGINES
        )
        return res
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Project ingestion failed: {str(e)}")


@app.post('/api/v1/projects/{repo_id}/sync', dependencies=[Depends(verify_manager_role)])
def sync_project_endpoint(repo_id: str):
    """
    Incremental Code Update Mechanism:
    Pulls latest git commits, diffs changed files, deletes old points,
    and embeds only modified/added code files without requiring full re-cloning.
    """
    try:
        res = sync_project_incremental(repo_id, repo_engines=REPO_ENGINES)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Incremental sync failed: {str(e)}")


@app.delete('/api/v1/projects/{repo_id}', dependencies=[Depends(verify_manager_role)])
def delete_project_endpoint(repo_id: str):
    """Deletes project: purges Qdrant collection, deletes local clone, and unregisters."""
    success = delete_project(repo_id, repo_engines=REPO_ENGINES)
    if not success:
        raise HTTPException(status_code=404, detail=f"Project '{repo_id}' not found or could not be deleted.")
    return {"status": "SUCCESS", "message": f"Project '{repo_id}' successfully deleted."}


# ================= REPOSITORY INGESTION (LEGACY & COMPATIBILITY) =================

@app.post('/api/v1/upload', status_code=201, dependencies=[Depends(verify_manager_role)])
def upload_and_index_repository(req: IngestionRequest):
    repo_url = req.repo_url.strip()
    if not repo_url.startswith(("http://", "https://", "git@", "ssh://", "/")) and not re.match(r'^[a-zA-Z]:\\', repo_url):
        raise HTTPException(status_code=400, detail="Invalid repository URL/Path format.")

    collection_name = get_repo_collection_name(repo_url)
    
    try:
        # 1. Clone & Scan
        temp_path = clone_repo(repo_url)
        documents = scanner(temp_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No parsable code files (.py, .js) found in repository.")

        # 2. Tree-sitter AST Chunking with Parameter & FQN Extraction
        all_chunks = []
        for doc in documents:
            chunks = chunker(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="Repository contains no functions or routes to index.")

        # 3. Dense Embeddings & Fresh Isolated Vector Storage
        embeddings = generate_embeddings(all_chunks)
        recreate_isolated_collection(collection_name)
        store_isolated_chunks(collection_name, all_chunks, embeddings)

        # 4. Instantiate Isolated Verification Engine
        engine = VerificationEngine(chunks=all_chunks, repo_id=collection_name)
        REPO_ENGINES[collection_name] = engine

        # 5. Register in Project Manager Studio
        try:
            register_project_ingestion(
                collection_name=collection_name,
                repo_url=repo_url,
                local_path=Path(temp_path),
                chunks=all_chunks,
                engine=engine,
                display_name=req.name
            )
        except Exception as reg_err:
            print(f"[main] Notice: Project registration warning: {reg_err}")

        return {
            "status": "SUCCESS",
            "msg": "Repo cloned, scanned, embedded into isolated Qdrant collection, and indexed into Verification Engine successfully",
            "repo_id": collection_name,
            "indexed_chunks": len(all_chunks),
            "qualified_symbols": len(engine.valid_fqns),
            "call_graph_nodes": engine.call_graph.number_of_nodes(),
            "call_graph_edges": engine.call_graph.number_of_edges()
        }
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


def get_or_restore_engine(collection_name: str) -> Optional[VerificationEngine]:
    if collection_name in REPO_ENGINES:
        return REPO_ENGINES[collection_name]
    
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        points = []
        next_page_offset = None
        while True:
            res, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=250,
                offset=next_page_offset,
                with_payload=True
            )
            points.extend(res)
            if next_page_offset is None:
                break
        chunks = [p.payload for p in points if p.payload]
        if chunks:
            engine = VerificationEngine(chunks=chunks, repo_id=collection_name)
            REPO_ENGINES[collection_name] = engine
            return engine
    return None


# ================= DEVELOPER QUERY ENDPOINT (4-LAYER ZERO-TRUST) =================

@app.post('/api/v1/userquery', dependencies=[Depends(verify_bearer_token)])
async def query_codebase(req: QueryRequest):
    client = get_qdrant_client()
    all_collections = [c.name for c in client.get_collections().collections if c.name.startswith("repo_")]

    if req.repo_id:
        collection_name = req.repo_id if req.repo_id.startswith("repo_") else get_repo_collection_name(req.repo_id)
    elif REPO_ENGINES:
        collection_name = list(REPO_ENGINES.keys())[-1]
    elif all_collections:
        # Default to repo_3e9c5a1753fa if present, else latest collection
        if "repo_3e9c5a1753fa" in all_collections:
            collection_name = "repo_3e9c5a1753fa"
        else:
            collection_name = all_collections[-1]
    else:
        raise HTTPException(
            status_code=400,
            detail="No repository is currently indexed. Call /api/v1/upload or /api/v1/projects first."
        )

    engine = get_or_restore_engine(collection_name)
    if not engine:
        raise HTTPException(
            status_code=404,
            detail=f"Repository collection '{collection_name}' not found. Upload repository first."
        )

    # 1. Embed query & Retrieve Top 20 from Isolated Qdrant Collection
    query_vec = query_embedding(req.query)
    candidates = search_isolated_query(collection_name, query_vec, limit=20)
    
    if not candidates:
        return {
            "query": req.query,
            "repo_id": collection_name,
            "answer": "No relevant code found in this repository.",
            "trust_decision": "REJECTED_NO_EVIDENCE",
            "trust_score": 0.0,
            "trust_percentage": 0,
            "badge_color": "red",
            "verification_layers": {},
            "rejection_reason": "No candidate chunks found in vector search.",
            "extracted_claims": {},
            "evidence_chunks": []
        }

    # 2. Candidate Selection (Dense Top 5 by default, Cross-Encoder Reranker if RERANK_ENABLED=true)
    rerank_enabled = os.getenv("RERANK_ENABLED", "false").strip().lower() in ("true", "1", "yes")
    if rerank_enabled:
        reranked = rerank_results(req.query, candidates)[:5]
    else:
        reranked = [
            {"point": c, "score": c.score}
            for c in candidates[:5]
        ]

    # Check top rerank score against configurable RERANK_MIN_SCORE (disabled by default)
    rerank_min_score_env = os.getenv("RERANK_MIN_SCORE")
    if rerank_min_score_env is not None and rerank_min_score_env.strip() != "":
        rerank_min_score = float(rerank_min_score_env)
        if not reranked or reranked[0]["score"] < rerank_min_score:
            top_score = reranked[0]["score"] if reranked else None
            reason = (
                f"Top rerank score ({top_score:.3f}) fell below minimum evidence threshold ({rerank_min_score})."
                if top_score is not None
                else "No candidate chunks found after reranking."
            )
            return {
                "query": req.query,
                "repo_id": collection_name,
                "answer": "No relevant code evidence found for this query in the repository.",
                "trust_decision": "REJECTED_NO_EVIDENCE",
                "trust_score": 0.0,
                "trust_percentage": 0,
                "badge_color": "red",
                "verification_layers": {},
                "rejection_reason": reason,
                "extracted_claims": {},
                "evidence_chunks": []
            }

    # 3. Ollama Offline Generation
    llm_output = await generate_code_answer(req.query, reranked)
    ollama_errored = "error" in llm_output or not llm_output.get("explanation")
    claims = llm_output.get("claims", {})

    # 4. Fail-Closed 4-Layer Deterministic Verification
    trust_report = engine.verify(
        claims=claims,
        retrieved_chunks=reranked,
        ollama_failed=ollama_errored,
        answer_text=llm_output.get("explanation")
    )

    # 5. Build Complete Evidence Bundle
    evidence_bundle = [
        {
            "file_path": item["point"].payload.get("file_path"),
            "symbol": item["point"].payload.get("symbol"),
            "fqn": item["point"].payload.get("fqn"),
            "symbol_type": item["point"].payload.get("symbol_type"),
            "parent": item["point"].payload.get("parent"),
            "lines": f"{item['point'].payload.get('start_line')}-{item['point'].payload.get('end_line')}",
            "rerank_score": item["score"],
            "source_code": item["point"].payload.get("content")
        }
        for item in reranked
    ]

    return {
        "query": req.query,
        "repo_id": collection_name,
        "answer": llm_output.get("explanation"),
        "trust_decision": trust_report["decision"],
        "trust_score": trust_report["trust_score"],
        "trust_percentage": trust_report["trust_percentage"],
        "badge_color": trust_report["badge_color"],
        "verification_layers": trust_report.get("layers", {}),
        "rejection_reason": trust_report.get("rejection_reason"),
        "extracted_claims": claims,
        "evidence_chunks": evidence_bundle
    }
