"""
FastAPI Server with Bearer-Token Authentication (HTTPBearer),
Multi-Tenant Repository Isolation, Truly Offline Models,
and Fail-Closed Zero-Trust Verification.
"""
import os
import re
from fastapi import FastAPI, HTTPException, Depends, status
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

app = FastAPI(
    title="Evidence-Backed Local Code RAG with Deterministic Verification",
    version="2.0.0"
)

security = HTTPBearer(auto_error=False)
REPO_ENGINES: Dict[str, VerificationEngine] = {}

def verify_bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """
    Strict Bearer Token Authentication.
    Reads token strictly from RAG_AUTH_TOKEN environment variable.
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
        
    if credentials.credentials.strip() != expected_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

class IngestionRequest(BaseModel):
    repo_url: str = Field(..., description="Git URL or allowlisted local repository path")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Technical query regarding the codebase")
    repo_id: Optional[str] = Field(None, description="Repository identifier (defaults to active repo)")

@app.get('/')
def health():
    return {
        "status": "online",
        "system": "Evidence-Backed Local Code RAG with Deterministic Verification",
        "isolated_repositories": list(REPO_ENGINES.keys())
    }

@app.post('/api/v1/upload', status_code=201, dependencies=[Depends(verify_bearer_token)])
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

        # 3. Dense Embeddings & Fresh Isolated Vector Storage (Purges Stale Vectors)
        embeddings = generate_embeddings(all_chunks)
        recreate_isolated_collection(collection_name)
        store_isolated_chunks(collection_name, all_chunks, embeddings)

        # 4. Instantiate Isolated Verification Engine with Qualified Symbols
        engine = VerificationEngine(chunks=all_chunks, repo_id=collection_name)
        REPO_ENGINES[collection_name] = engine

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

@app.post('/api/v1/userquery', dependencies=[Depends(verify_bearer_token)])
async def query_codebase(req: QueryRequest):
    client = get_qdrant_client()
    all_collections = [c.name for c in client.get_collections().collections]

    if req.repo_id:
        collection_name = req.repo_id if req.repo_id.startswith("repo_") else get_repo_collection_name(req.repo_id)
    elif REPO_ENGINES:
        collection_name = list(REPO_ENGINES.keys())[-1]
    elif all_collections:
        collection_name = all_collections[-1]
    else:
        raise HTTPException(
            status_code=400,
            detail="No repository is currently indexed. Call /api/v1/upload first."
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
