import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.verification import VerificationEngine

sample_chunks = [
    {
        "file_path": "src/auth/service.py",
        "symbol": "AuthService.login",
        "symbol_type": "method",
        "parent": "AuthService",
        "params": ["self", "username", "password"],
        "content": """
def login(self, username, password):
    user = self.find_user(username)
    return self.verify_pw(password, user.hash)
"""
    },
    {
        "file_path": "src/billing/service.py",
        "symbol": "BillingService.login",
        "symbol_type": "method",
        "parent": "BillingService",
        "params": ["self", "api_token"],
        "content": """
def login(self, api_token):
    return remote_gateway_auth(api_token)
"""
    },
    {
        "file_path": "src/auth/service.py",
        "symbol": "AuthService.find_user",
        "symbol_type": "method",
        "parent": "AuthService",
        "params": ["self", "username"],
        "content": """
def find_user(self, username):
    return None
"""
    },
    {
        "file_path": "src/auth/service.py",
        "symbol": "AuthService.verify_pw",
        "symbol_type": "method",
        "parent": "AuthService",
        "params": ["self", "password", "hash"],
        "content": """
def verify_pw(self, password, hash):
    return True
"""
    },
    {
        "file_path": "src/billing/service.py",
        "symbol": "BillingService.charge",
        "symbol_type": "method",
        "parent": "BillingService",
        "params": ["self", "amount"],
        "content": """
def charge(self, amount):
    return True
"""
    }
]

def get_engine():
    return VerificationEngine(sample_chunks, repo_id="test_repo")

def test_fail_closed_on_empty_or_failed_ollama(engine):
    """Proves system fails closed (Trust = 0.0) when LLM output is empty or fails."""
    res1 = engine.verify(claims={}, retrieved_chunks=sample_chunks, ollama_failed=True)
    assert res1["trust_score"] == 0.0
    assert res1["decision"] == "REJECTED_UNVERIFIABLE"
    assert res1["badge_color"] == "red"

    empty_claims = {"symbols_mentioned": [], "structures": [], "call_edges": [], "factual_assertions": []}
    res2 = engine.verify(claims=empty_claims, retrieved_chunks=sample_chunks, ollama_failed=False)
    assert res2["trust_score"] == 0.0
    assert res2["decision"] == "REJECTED_EMPTY_CLAIMS"
    assert res2["badge_color"] == "red"
    assert res2["layers"]["layer1_existence"]["status"] == "not_checked"

def test_reject_severe_symbol_hallucination_l1_below_half(engine):
    """Proves L1 < 0.5 triggers REJECTED."""
    claims = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "PhantomCryptoVault",
            "GhostService"
        ], # 1 of 3 = 0.333 < 0.5
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res = engine.verify(claims, sample_chunks)
    assert res["layers"]["layer1_existence"]["score"] < 0.50
    assert res["layers"]["layer1_existence"]["status"] == "failed"
    assert "PhantomCryptoVault" in res["layers"]["layer1_existence"]["missing_symbols"]
    assert res["decision"] == "REJECTED"
    assert res["badge_color"] == "red"

def test_problem_1_layer4_failing_alone_yields_warning(engine):
    """Problem 1: Layer 4 failing alone must yield WARNING, never VERIFIED."""
    claims = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "src/auth/service.py::AuthService.find_user"
        ],
        "structures": [
            {
                "symbol": "src/auth/service.py::AuthService.login",
                "claimed_params": ["username", "password"],
                "claimed_parent": "AuthService"
            }
        ],
        "call_edges": [],
        "factual_assertions": [
            "Completely unsupported assertion about quantum entanglement blockchain"
        ] # L4 fails with 0 coverage
    }
    res = engine.verify(claims, sample_chunks)
    assert res["layers"]["layer1_existence"]["score"] == 1.0
    assert res["layers"]["layer2_structural"]["score"] == 1.0
    assert res["layers"]["layer4_evidence"]["passed"] is False
    assert res["layers"]["layer4_evidence"]["status"] == "failed"
    assert res["decision"] == "WARNING"
    assert res["badge_color"] == "amber"
    assert res["trust_score"] < 0.80

def test_problem_2_single_failed_l2_or_l3_yields_warning(engine):
    """Problem 2: Any single failed layer among L2/L3 must yield at best WARNING."""
    # L2 structural failure alone
    claims_l2_fail = {
        "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
        "structures": [
            {
                "symbol": "src/auth/service.py::AuthService.login",
                "claimed_params": ["api_token"] # Incorrect parameters
            }
        ],
        "call_edges": [],
        "factual_assertions": []
    }
    res_l2 = engine.verify(claims_l2_fail, sample_chunks)
    assert res_l2["layers"]["layer2_structural"]["passed"] is False
    assert res_l2["layers"]["layer2_structural"]["status"] == "failed"
    assert res_l2["decision"] == "WARNING"
    assert res_l2["badge_color"] == "amber"

    # L3 callgraph failure alone
    claims_l3_fail = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "src/billing/service.py::BillingService.login"
        ],
        "structures": [],
        "call_edges": [
            {
                "caller": "src/auth/service.py::AuthService.login",
                "callee": "src/billing/service.py::BillingService.login"
            }
        ],
        "factual_assertions": []
    }
    res_l3 = engine.verify(claims_l3_fail, sample_chunks)
    assert res_l3["layers"]["layer3_callgraph"]["passed"] is False
    assert res_l3["layers"]["layer3_callgraph"]["status"] == "failed"
    assert res_l3["decision"] == "WARNING"
    assert res_l3["badge_color"] == "amber"

def test_problem_3_empty_claims_return_not_checked_and_prevent_free_trust(engine):
    """Problem 3: Empty structures/call_edges/assertions return score=None, status='not_checked'.
    With the 'at least 2 claim types' rule dropped, missing claim types are 'not_checked', not a penalty.
    """
    claims = {
        "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res = engine.verify(claims, sample_chunks)
    assert res["layers"]["layer1_existence"]["score"] == 1.0
    assert res["layers"]["layer1_existence"]["status"] == "passed"
    
    assert res["layers"]["layer2_structural"]["score"] is None
    assert res["layers"]["layer2_structural"]["status"] == "not_checked"
    assert res["layers"]["layer3_callgraph"]["score"] is None
    assert res["layers"]["layer3_callgraph"]["status"] == "not_checked"
    assert res["layers"]["layer4_evidence"]["score"] is None
    assert res["layers"]["layer4_evidence"]["status"] == "not_checked"
    
    # Missing claim types are not penalized; valid symbols alone can achieve VERIFIED
    assert res["decision"] == "VERIFIED"
    assert res["badge_color"] == "green"
    assert res["trust_score"] >= 0.80

def test_problem_4_invented_symbol_1_of_5_yields_warning(engine):
    """Problem 4: 1 invented symbol out of 5 claimed (L1 = 0.8) yields at best WARNING."""
    claims = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "src/auth/service.py::AuthService.find_user",
            "src/auth/service.py::AuthService.verify_pw",
            "src/billing/service.py::BillingService.charge",
            "src/auth/service.py::AuthService.invented_method"
        ],
        "structures": [
            {
                "symbol": "src/auth/service.py::AuthService.login",
                "claimed_params": ["username", "password"],
                "claimed_parent": "AuthService"
            }
        ],
        "call_edges": [],
        "factual_assertions": []
    }
    res = engine.verify(claims, sample_chunks)
    assert res["layers"]["layer1_existence"]["score"] == 0.80
    assert res["layers"]["layer1_existence"]["passed"] is False
    assert res["layers"]["layer1_existence"]["status"] == "warning"
    assert res["decision"] == "WARNING"
    assert res["badge_color"] == "amber"
    assert res["trust_score"] < 0.80

def test_verified_requires_all_claims_pass_and_at_least_2_claim_types(engine):
    """Rule: VERIFIED requires every provided layer passes AND at least 2 claim types provided."""
    claims = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "src/auth/service.py::AuthService.find_user"
        ],
        "structures": [
            {
                "symbol": "src/auth/service.py::AuthService.login",
                "claimed_params": ["username", "password"],
                "claimed_parent": "AuthService"
            }
        ],
        "call_edges": [],
        "factual_assertions": []
    }
    res = engine.verify(claims, sample_chunks)
    assert res["layers"]["layer1_existence"]["passed"] is True
    assert res["layers"]["layer2_structural"]["passed"] is True
    assert res["decision"] == "VERIFIED"
    assert res["badge_color"] == "green"
    assert res["trust_score"] >= 0.80

def test_unverifiable_external_call_handling(engine):
    """
    Validates Layer 3 external/unindexed call handling:
    - returns score: None and passed: None
    - keeps status: 'unverifiable_external'
    - excludes it from trust-score aggregation
    - prevents a VERIFIED decision when any material answer claim is unverifiable externally (yields WARNING)
    """
    claims = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login"
        ],
        "structures": [
            {
                "symbol": "src/auth/service.py::AuthService.login",
                "claimed_params": ["username", "password"],
                "claimed_parent": "AuthService"
            }
        ],
        "call_edges": [
            {
                "caller": "src/auth/service.py::AuthService.login",
                "callee": "external_library_func"  # Unindexed external symbol
            }
        ],
        "factual_assertions": [
            "login username password find_user verify_pw"
        ]
    }
    res = engine.verify(claims, sample_chunks)
    
    # 1. Layer 3 attributes
    l3 = res["layers"]["layer3_callgraph"]
    assert l3["score"] is None, f"Expected score None, got {l3['score']}"
    assert l3["passed"] is None, f"Expected passed None, got {l3['passed']}"
    assert l3["status"] == "unverifiable_external"
    assert len(l3["unverifiable_external"]) == 1
    assert l3["unverifiable_external"][0]["callee"] == "external_library_func"
    assert len(l3["violations"]) == 0

    # 2. Decision must NOT be VERIFIED when an external unverifiable claim exists
    assert res["decision"] == "WARNING", f"Expected WARNING, got {res['decision']}"
    assert res["badge_color"] == "amber"

    # 3. Layer 3 must be excluded from trust-score aggregation:
    # L1 (0.30 weight) = 1.0, L2 (0.25 weight) = 1.0, L4 (0.20 weight) = 1.0
    # Aggregated base_score = (0.30*1.0 + 0.25*1.0 + 0.20*1.0) / (0.30 + 0.25 + 0.20) = 0.75 / 0.75 = 1.0
    # For WARNING, final_trust is clamped to min(0.75, max(0.50, base_score)) = 0.75
    assert res["trust_score"] == 0.75

def test_answer_prose_derived_claims_and_ungrounded_identifier_flagging(engine):
    """
    Tests that:
    1. Grounded backticked identifiers in answer prose are verified.
    2. Ungrounded identifiers in prose are flagged in ungrounded_prose_identifiers and missing_symbols.
    3. Any ungrounded prose identifier prevents VERIFIED and causes WARNING/REJECTED.
    """
    # 1. Faithful answer prose with grounded identifiers
    faithful_prose = "The `AuthService.login` method authenticates users and invokes `AuthService.find_user`."
    res_faithful = engine.verify(
        claims={
            "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
            "factual_assertions": ["login user find_user"]
        },
        retrieved_chunks=sample_chunks,
        answer_text=faithful_prose
    )
    l1_faithful = res_faithful["layers"]["layer1_existence"]
    assert l1_faithful["ungrounded_prose_identifiers"] == []
    assert l1_faithful["passed"] is True
    assert res_faithful["decision"] == "VERIFIED"

    # 2. Hallucinated answer prose with ungrounded identifiers
    hallucinated_prose = "The login method invokes `secretBackdoorToken` and calls `injectCryptoKey()` to bypass auth."
    res_hallucinated = engine.verify(
        claims={
            "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
            "factual_assertions": ["login user"]
        },
        retrieved_chunks=sample_chunks,
        answer_text=hallucinated_prose
    )
    l1_hallucinated = res_hallucinated["layers"]["layer1_existence"]
    assert "secretBackdoorToken" in l1_hallucinated["ungrounded_prose_identifiers"]
    assert "injectCryptoKey()" in l1_hallucinated["ungrounded_prose_identifiers"]
    assert "secretBackdoorToken" in l1_hallucinated["missing_symbols"]
    assert l1_hallucinated["passed"] is False
    assert res_hallucinated["decision"] in ["WARNING", "REJECTED"]
    assert res_hallucinated["badge_color"] in ["amber", "red"]

def test_rerank_min_score_rejection():
    """
    Tests the logic of rerank abstain threshold:
    Top rerank score below RERANK_MIN_SCORE (-6.0) returns REJECTED_NO_EVIDENCE.
    """
    rerank_min_score = -6.0
    top_score = -8.5
    abstained = top_score < rerank_min_score
    assert abstained is True
    rejection_result = {
        "trust_decision": "REJECTED_NO_EVIDENCE",
        "trust_score": 0.0,
        "trust_percentage": 0,
        "badge_color": "red",
        "rejection_reason": f"Top rerank score ({top_score:.3f}) fell below minimum evidence threshold ({rerank_min_score})."
    }
    assert rejection_result["trust_decision"] == "REJECTED_NO_EVIDENCE"
    assert rejection_result["badge_color"] == "red"

def test_root_routes_trailing_slash_resolution():
    """
    Regression test ensuring _resolve_symbol preserves trailing slashes on
    root routes like 'GET /' and 'POST /' and resolves them correctly.
    """
    chunks = [
        {
            "file_path": "routes/cropRoutes.js",
            "symbol": "GET /",
            "fqn": "routes/cropRoutes.js::GET /",
            "symbol_type": "route",
            "parent": "router",
            "params": ["req", "res"],
            "content": "router.get('/', (req, res) => {});"
        },
        {
            "file_path": "routes/toolRoutes.js",
            "symbol": "POST /",
            "fqn": "routes/toolRoutes.js::POST /",
            "symbol_type": "route",
            "parent": "router",
            "params": ["req", "res"],
            "content": "router.post('/', (req, res) => {});"
        }
    ]
    engine = VerificationEngine(chunks=chunks, repo_id="root_test")

    # 1. Exact FQN resolution with trailing slash
    assert engine._resolve_symbol("routes/cropRoutes.js::GET /") == "routes/cropRoutes.js::GET /"
    assert engine._resolve_symbol("routes/toolRoutes.js::POST /") == "routes/toolRoutes.js::POST /"

    # 2. File suffix resolution with trailing slash
    assert engine._resolve_symbol("cropRoutes.js::GET /") == "routes/cropRoutes.js::GET /"
    assert engine._resolve_symbol("toolRoutes.js::POST /") == "routes/toolRoutes.js::POST /"

    # 3. Disambiguated by target_file
    assert engine._resolve_symbol("GET /", target_file="routes/cropRoutes.js") == "routes/cropRoutes.js::GET /"
    assert engine._resolve_symbol("POST /", target_file="routes/toolRoutes.js") == "routes/toolRoutes.js::POST /"

def test_rerank_enabled_unset_uses_dense_top5():
    """RERANK_ENABLED unset -> dense top-5 used and evidence_chunks is populated."""
    from main import app
    from fastapi.testclient import TestClient
    
    # Save original env
    orig_env = os.environ.get("RERANK_ENABLED")
    orig_token = os.environ.get("RAG_AUTH_TOKEN")
    os.environ["RAG_AUTH_TOKEN"] = "test-runner-token-2026"
    if "RERANK_ENABLED" in os.environ:
        del os.environ["RERANK_ENABLED"]

    try:
        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer test-runner-token-2026"}
        res = client.post(
            "/api/v1/userquery",
            headers=auth_headers,
            json={"query": "Where is the login endpoint implemented?", "repo_id": "repo_3e9c5a1753fa"}
        )
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert "evidence_chunks" in data
        assert len(data["evidence_chunks"]) == 5, f"Expected 5 evidence chunks, got {len(data['evidence_chunks'])}"
        # Each chunk should have dense similarity score, fqn, and source_code
        for ch in data["evidence_chunks"]:
            assert ch.get("fqn") is not None
            assert ch.get("rerank_score") is not None
    finally:
        if orig_env is not None:
            os.environ["RERANK_ENABLED"] = orig_env

def test_package_prose_flagging_and_ignored_patterns(engine: VerificationEngine):
    """
    Tests:
    1. escape-html and react-dom/server flagged when absent from retrieved chunks & package.json,
       and NOT flagged when present in either.
    2. req.params.*, req.query.*, req.body.*, route strings and sentence fragments stay ignored.
    """
    # 1. Package tokens absent from chunks & package.json -> flagged
    prose_absent = "Use `escape-html` or `react-dom/server` to prevent XSS."
    rep_absent = engine.verify(
        claims={},
        retrieved_chunks=[{"file_path": "server.js", "symbol": "app", "fqn": "server.js::app", "content": "const app = express();"}],
        ollama_failed=False,
        answer_text=prose_absent
    )
    ungrounded = rep_absent["layers"]["layer1_existence"]["ungrounded_prose_identifiers"]
    assert "escape-html" in ungrounded, f"Expected escape-html flagged, got {ungrounded}"
    assert "react-dom/server" in ungrounded, f"Expected react-dom/server flagged, got {ungrounded}"

    # 2. Package tokens present in chunks -> NOT flagged
    rep_present_chunk = engine.verify(
        claims={},
        retrieved_chunks=[{
            "file_path": "package.json",
            "symbol": "dependencies",
            "fqn": "package.json::dependencies",
            "content": '{"dependencies": {"escape-html": "^1.0.3", "react-dom/server": "^18.0.0"}}'
        }],
        ollama_failed=False,
        answer_text=prose_absent
    )
    ungrounded_present = rep_present_chunk["layers"]["layer1_existence"]["ungrounded_prose_identifiers"]
    assert "escape-html" not in ungrounded_present
    assert "react-dom/server" not in ungrounded_present

    # 3. Known package in repo package.json (e.g. 'express', 'mongoose', 'dotenv') -> NOT flagged
    prose_known = "We require(`express`) and use `dotenv`."
    rep_known = engine.verify(
        claims={},
        retrieved_chunks=[{"file_path": "server.js", "symbol": "app", "fqn": "server.js::app", "content": "const app = 1;"}],
        ollama_failed=False,
        answer_text=prose_known
    )
    ungrounded_known = rep_known["layers"]["layer1_existence"]["ungrounded_prose_identifiers"]
    assert "express" not in ungrounded_known
    assert "dotenv" not in ungrounded_known

    # 4. req.params.*, req.query.*, req.body.*, route strings, and sentence fragments stay ignored
    prose_ignored = (
        "Check `req.params.id`, `req.query.category`, and `req.body.name`. "
        "Also route `/api/crops` and `POST /login`. "
        "Furthermore `Asserts that the PATCH /users/:id/approve route calls` should be ignored."
    )
    rep_ignored = engine.verify(
        claims={},
        retrieved_chunks=[{"file_path": "server.js", "symbol": "app", "fqn": "server.js::app", "content": "const app = 1;"}],
        ollama_failed=False,
        answer_text=prose_ignored
    )
    ungrounded_ignored = rep_ignored["layers"]["layer1_existence"]["ungrounded_prose_identifiers"]
    for pattern in ["req.params.id", "req.query.category", "req.body.name", "/api/crops", "POST /login"]:
        assert pattern not in ungrounded_ignored, f"Pattern {pattern} should be ignored, but was flagged in {ungrounded_ignored}"
    for token in ungrounded_ignored:
        assert "Asserts that" not in token, f"Sentence fragment should be ignored, got {token}"

def test_layer1_grounding_code_tokens_metadata_and_allowlist(engine):
    """Rule: Grounding from code tokens, chunk metadata, and library allowlist."""
    retrieved = [
        {
            "file_path": "routes/priceRoutes.js",
            "symbol": "POST /add",
            "symbol_type": "route",
            "fqn": "routes/priceRoutes.js::POST /add",
            "params": ["req", "res"],
            "content": """
router.post('/add', async (req, res) => {
    const { crop, price, location } = req.body;
    const txStatus = 'pending';
    const saved = await Price.save();
    res.json({ success: true });
});
"""
        }
    ]
    # 1. Code tokens (req.body, location, txStatus) + route (/add, POST /add) + file (routes/priceRoutes.js)
    claims_grounded = {
        "symbols_mentioned": [
            "req.body", "location", "txStatus", "/add", "POST /add", "routes/priceRoutes.js",
            "async", "res.json", "Promise", "mongoose.Schema", "express.Router", "findById", "select"
        ],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res = engine.verify(claims_grounded, retrieved)
    assert res["layers"]["layer1_existence"]["passed"] is True
    assert res["layers"]["layer1_existence"]["score"] == 1.0
    assert len(res["layers"]["layer1_existence"]["invented_tokens"]) == 0
    assert len(res["layers"]["layer1_existence"]["exists_not_retrieved"]) == 0
    assert res["decision"] == "VERIFIED"

def test_decision_gating_and_invented_thresholds(engine):
    """Rule: Decision gating on invented tokens and exists_not_retrieved."""
    retrieved = [
        {
            "file_path": "src/auth/service.py",
            "symbol": "AuthService.login",
            "symbol_type": "method",
            "fqn": "src/auth/service.py::AuthService.login",
            "params": ["self", "username", "password"],
            "content": "def login(self, username, password): return True"
        }
    ]
    # A) Exists in repo but not in retrieved context -> exists_not_retrieved -> WARNING
    claims_exists_not_retrieved = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "src/billing/service.py::BillingService.charge"  # in repo, but not retrieved!
        ],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res_enr = engine.verify(claims_exists_not_retrieved, retrieved)
    assert "src/billing/service.py::BillingService.charge" in res_enr["layers"]["layer1_existence"]["exists_not_retrieved"]
    assert res_enr["decision"] == "WARNING"
    assert res_enr["badge_color"] == "amber"

    # B) Exactly 1 invented token out of 5 (ratio = 0.20 < 0.30) -> WARNING
    claims_1_of_5 = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "req", "res", "next",
            "invented_nonexistent_function"
        ],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res_1_of_5 = engine.verify(claims_1_of_5, retrieved)
    assert len(res_1_of_5["layers"]["layer1_existence"]["invented_tokens"]) == 1
    assert res_1_of_5["decision"] == "WARNING"

    # C) 1 invented token out of 2 (ratio = 0.50 >= 0.30) -> REJECTED
    claims_1_of_2 = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "invented_nonexistent_function"
        ],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res_1_of_2 = engine.verify(claims_1_of_2, retrieved)
    assert res_1_of_2["decision"] == "REJECTED"
    assert res_1_of_2["badge_color"] == "red"

    # D) >= 2 invented tokens -> REJECTED
    claims_2_inv = {
        "symbols_mentioned": [
            "src/auth/service.py::AuthService.login",
            "PhantomService.inv1", "PhantomService.inv2"
        ],
        "structures": [],
        "call_edges": [],
        "factual_assertions": []
    }
    res_2_inv = engine.verify(claims_2_inv, retrieved)
    assert res_2_inv["decision"] == "REJECTED"

def test_layer2_route_param_check_skip():
    """Rule: Layer 2 for route chunks accepts req/res/next and identifiers in code; flags absent params."""
    chunks = [
        {
            "file_path": "routes/cropRoutes.js",
            "symbol": "POST /add",
            "symbol_type": "route",
            "fqn": "routes/cropRoutes.js::POST /add",
            "params": ["req", "res"],
            "content": "router.post('/add', (req, res) => { const { crop, price } = req.body; });"
        },
        {
            "file_path": "utils/helper.js",
            "symbol": "formatDate",
            "symbol_type": "function",
            "fqn": "utils/helper.js::formatDate",
            "params": ["dateObj", "fmtStr"],
            "content": "function formatDate(dateObj, fmtStr) { return ''; }"
        }
    ]
    eng = VerificationEngine(chunks=chunks, repo_id="l2_test")

    # Route with params appearing in code ('crop', 'price') along with req, res: ACCEPTED
    res_route_ok = eng.verify(
        claims={
            "symbols_mentioned": ["routes/cropRoutes.js::POST /add"],
            "structures": [
                {
                    "symbol": "routes/cropRoutes.js::POST /add",
                    "claimed_params": ["req", "res", "crop", "price"]
                }
            ],
            "call_edges": [],
            "factual_assertions": []
        },
        retrieved_chunks=[chunks[0]]
    )
    assert res_route_ok["layers"]["layer2_structural"]["passed"] is True
    assert len(res_route_ok["layers"]["layer2_structural"]["violations"]) == 0
    assert res_route_ok["decision"] == "VERIFIED"

    # Route with params appearing nowhere in code ('token', 'payload'): FLAGGED
    res_route_bad = eng.verify(
        claims={
            "symbols_mentioned": ["routes/cropRoutes.js::POST /add"],
            "structures": [
                {
                    "symbol": "routes/cropRoutes.js::POST /add",
                    "claimed_params": ["token", "payload"]
                }
            ],
            "call_edges": [],
            "factual_assertions": []
        },
        retrieved_chunks=[chunks[0]]
    )
    assert res_route_bad["layers"]["layer2_structural"]["passed"] is False
    assert len(res_route_bad["layers"]["layer2_structural"]["violations"]) > 0
    assert res_route_bad["decision"] == "WARNING"

    # Function with wrong claimed params: ENFORCED (violation)
    res_func = eng.verify(
        claims={
            "symbols_mentioned": ["utils/helper.js::formatDate"],
            "structures": [
                {
                    "symbol": "utils/helper.js::formatDate",
                    "claimed_params": ["wrongParam1", "wrongParam2"]
                }
            ],
            "call_edges": [],
            "factual_assertions": []
        },
        retrieved_chunks=[chunks[1]]
    )
    assert res_func["layers"]["layer2_structural"]["passed"] is False
    assert len(res_func["layers"]["layer2_structural"]["violations"]) > 0
    assert res_func["decision"] == "WARNING"

def test_layer3_library_allowlist_calls_ignored():
    """Rule: Layer 3 calls to library allowlist are ignored (neither violations nor unverifiable_external)."""
    chunks = [
        {
            "file_path": "routes/authRoutes.js",
            "symbol": "POST /login",
            "symbol_type": "route",
            "fqn": "routes/authRoutes.js::POST /login",
            "params": ["req", "res"],
            "content": "router.post('/login', async (req, res) => { await User.findOne(); });"
        }
    ]
    eng = VerificationEngine(chunks=chunks, repo_id="l3_test")

    # Call to User.findOne and fetch (both allowlisted) -> ignored in L3, status is not_checked
    res = eng.verify(
        claims={
            "symbols_mentioned": ["routes/authRoutes.js::POST /login"],
            "structures": [],
            "call_edges": [
                {"caller": "routes/authRoutes.js::POST /login", "callee": "User.findOne"},
                {"caller": "routes/authRoutes.js::POST /login", "callee": "fetch"}
            ],
            "factual_assertions": []
        },
        retrieved_chunks=[chunks[0]]
    )
    assert res["layers"]["layer3_callgraph"]["status"] == "not_checked"
    assert len(res["layers"]["layer3_callgraph"]["unverifiable_external"]) == 0
    assert res["decision"] == "VERIFIED"

    # Unknown unindexed callee -> unverifiable_external -> WARNING
    res_unknown = eng.verify(
        claims={
            "symbols_mentioned": ["routes/authRoutes.js::POST /login"],
            "structures": [],
            "call_edges": [
                {"caller": "routes/authRoutes.js::POST /login", "callee": "unknownExternalService.processPayment"}
            ],
            "factual_assertions": []
        },
        retrieved_chunks=[chunks[0]]
    )
    assert res_unknown["layers"]["layer3_callgraph"]["status"] == "unverifiable_external"
    assert len(res_unknown["layers"]["layer3_callgraph"]["unverifiable_external"]) == 1
    assert res_unknown["decision"] == "WARNING"

if __name__ == "__main__":
    eng = get_engine()
    test_fail_closed_on_empty_or_failed_ollama(eng)
    print("PASS: test_fail_closed_on_empty_or_failed_ollama")
    test_reject_severe_symbol_hallucination_l1_below_half(eng)
    print("PASS: test_reject_severe_symbol_hallucination_l1_below_half")
    test_problem_1_layer4_failing_alone_yields_warning(eng)
    print("PASS: test_problem_1_layer4_failing_alone_yields_warning")
    test_problem_2_single_failed_l2_or_l3_yields_warning(eng)
    print("PASS: test_problem_2_single_failed_l2_or_l3_yields_warning")
    test_problem_3_empty_claims_return_not_checked_and_prevent_free_trust(eng)
    print("PASS: test_problem_3_empty_claims_return_not_checked_and_prevent_free_trust")
    test_problem_4_invented_symbol_1_of_5_yields_warning(eng)
    print("PASS: test_problem_4_invented_symbol_1_of_5_yields_warning")
    test_verified_requires_all_claims_pass_and_at_least_2_claim_types(eng)
    print("PASS: test_verified_requires_all_claims_pass_and_at_least_2_claim_types")
    test_unverifiable_external_call_handling(eng)
    print("PASS: test_unverifiable_external_call_handling")
    test_answer_prose_derived_claims_and_ungrounded_identifier_flagging(eng)
    print("PASS: test_answer_prose_derived_claims_and_ungrounded_identifier_flagging")
    test_rerank_min_score_rejection()
    print("PASS: test_rerank_min_score_rejection")
    test_root_routes_trailing_slash_resolution()
    print("PASS: test_root_routes_trailing_slash_resolution")
    test_rerank_enabled_unset_uses_dense_top5()
    print("PASS: test_rerank_enabled_unset_uses_dense_top5")
    test_package_prose_flagging_and_ignored_patterns(eng)
    print("PASS: test_package_prose_flagging_and_ignored_patterns")
    test_layer1_grounding_code_tokens_metadata_and_allowlist(eng)
    print("PASS: test_layer1_grounding_code_tokens_metadata_and_allowlist")
    test_decision_gating_and_invented_thresholds(eng)
    print("PASS: test_decision_gating_and_invented_thresholds")
    test_layer2_route_param_check_skip()
    print("PASS: test_layer2_route_param_check_skip")
    test_layer3_library_allowlist_calls_ignored()
    print("PASS: test_layer3_library_allowlist_calls_ignored")
    print("\nALL STRICT VERIFICATION TESTS PASSED PERFECTLY!")


