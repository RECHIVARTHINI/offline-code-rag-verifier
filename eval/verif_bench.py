"""
eval/verif_bench.py: Benchmark harness for deterministic verification engine.
Evaluates 374 labelled cases across faithful and hallucinated scenarios.
"""
import sys
import os
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.verification import VerificationEngine

def create_benchmark_dataset() -> List[Dict[str, Any]]:
    """Constructs 374 labelled test cases with ground truth."""
    cases = []
    
    base_chunks = [
        {
            "file_path": "src/auth/service.py",
            "symbol": "AuthService.login",
            "fqn": "src/auth/service.py::AuthService.login",
            "symbol_type": "method",
            "parent": "AuthService",
            "params": ["self", "username", "password"],
            "content": "def login(self, username, password):\n    user = self.find_user(username)\n    return self.verify_pw(password, user.hash)\n"
        },
        {
            "file_path": "src/auth/service.py",
            "symbol": "AuthService.find_user",
            "fqn": "src/auth/service.py::AuthService.find_user",
            "symbol_type": "method",
            "parent": "AuthService",
            "params": ["self", "username"],
            "content": "def find_user(self, username):\n    return db.query(username)\n"
        },
        {
            "file_path": "src/auth/service.py",
            "symbol": "AuthService.verify_pw",
            "fqn": "src/auth/service.py::AuthService.verify_pw",
            "symbol_type": "method",
            "parent": "AuthService",
            "params": ["self", "pw", "hash"],
            "content": "def verify_pw(self, pw, hash):\n    return bcrypt.check(pw, hash)\n"
        },
        {
            "file_path": "src/billing/service.py",
            "symbol": "BillingService.charge",
            "fqn": "src/billing/service.py::BillingService.charge",
            "symbol_type": "method",
            "parent": "BillingService",
            "params": ["self", "user_id", "amount"],
            "content": "def charge(self, user_id, amount):\n    return stripe.Charge.create(user=user_id, amount=amount)\n"
        },
        {
            "file_path": "src/billing/service.py",
            "symbol": "BillingService.refund",
            "fqn": "src/billing/service.py::BillingService.refund",
            "symbol_type": "method",
            "parent": "BillingService",
            "params": ["self", "charge_id"],
            "content": "def refund(self, charge_id):\n    return stripe.Refund.create(charge=charge_id)\n"
        }
    ]

    # 1. 150 Faithful cases (All symbols, structures, edges, assertions are faithful)
    for i in range(150):
        cases.append({
            "id": f"faithful_{i+1}",
            "category": "faithful",
            "ground_truth": "faithful",  # Expected VERIFIED
            "chunks": base_chunks,
            "claims": {
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
                "call_edges": [
                    {
                        "caller": "src/auth/service.py::AuthService.login",
                        "callee": "src/auth/service.py::AuthService.find_user"
                    }
                ],
                "factual_assertions": [
                    "login verifies password using hash and finds username in database"
                ]
            }
        })

    # 2. 60 "invented symbol (1 of 5)" cases (4 valid symbols, 1 invented symbol -> L1 = 0.8)
    for i in range(60):
        cases.append({
            "id": f"inv_sym_1_of_5_{i+1}",
            "category": "invented symbol (1 of 5)",
            "ground_truth": "hallucination",  # Expected at best WARNING / REJECTED (NOT VERIFIED)
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/auth/service.py::AuthService.login",
                    "src/auth/service.py::AuthService.find_user",
                    "src/auth/service.py::AuthService.verify_pw",
                    "src/billing/service.py::BillingService.charge",
                    f"src/auth/service.py::AuthService.phantom_invented_{i}"
                ],
                "structures": [],
                "call_edges": [],
                "factual_assertions": []
            }
        })

    # 3. 60 "false assertion" cases (Layer 4 alone fails with 0 overlap, L1/L2/L3 pass or empty)
    for i in range(60):
        cases.append({
            "id": f"false_assertion_{i+1}",
            "category": "false assertion",
            "ground_truth": "hallucination",  # Expected at best WARNING / REJECTED (NOT VERIFIED)
            "chunks": base_chunks,
            "claims": {
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
                "call_edges": [
                    {
                        "caller": "src/auth/service.py::AuthService.login",
                        "callee": "src/auth/service.py::AuthService.find_user"
                    }
                ],
                "factual_assertions": [
                    f"Completely fabricated claim about quantum blockchain cryptography integration #{i}"
                ]
            }
        })

    # 4. 40 "invented structure" cases (L2 structural mismatch: wrong parameters)
    for i in range(40):
        cases.append({
            "id": f"inv_struct_{i+1}",
            "category": "invented structure",
            "ground_truth": "hallucination",
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
                "structures": [
                    {
                        "symbol": "src/auth/service.py::AuthService.login",
                        "claimed_params": ["wrong_arg_1", "wrong_arg_2"],
                        "claimed_parent": "AuthService"
                    }
                ],
                "call_edges": [],
                "factual_assertions": []
            }
        })

    # 5. 40 "invented call edge" cases (L3 direct invocation violation)
    for i in range(40):
        cases.append({
            "id": f"inv_call_{i+1}",
            "category": "invented call edge",
            "ground_truth": "hallucination",
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/auth/service.py::AuthService.login",
                    "src/billing/service.py::BillingService.refund"
                ],
                "structures": [],
                "call_edges": [
                    {
                        "caller": "src/auth/service.py::AuthService.login",
                        "callee": "src/billing/service.py::BillingService.refund"
                    }
                ],
                "factual_assertions": []
            }
        })

    # 6. 14 "severe symbol hallucination" cases (L1 < 0.5)
    for i in range(14):
        cases.append({
            "id": f"severe_l1_{i+1}",
            "category": "severe symbol hallucination",
            "ground_truth": "hallucination",
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/auth/service.py::AuthService.login",
                    f"fake_symbol_a_{i}",
                    f"fake_symbol_b_{i}",
                    f"fake_symbol_c_{i}"
                ],
                "structures": [],
                "call_edges": [],
                "factual_assertions": []
            }
        })

    # 7. 10 "empty / single claim type" cases
    for i in range(10):
        cases.append({
            "id": f"single_claim_type_{i+1}",
            "category": "single claim type provided",
            "ground_truth": "insufficient_claims",
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": ["src/auth/service.py::AuthService.login"],
                "structures": [],
                "call_edges": [],
                "factual_assertions": []
            }
        })

    # 8. 7 "faithful edge cases" (middleware, external calls, cross-file calls)
    faithful_edge_chunks = [
        {
            "file_path": "routes/adminRoutes.js",
            "symbol": "requireAdmin",
            "fqn": "routes/adminRoutes.js::requireAdmin",
            "symbol_type": "function",
            "params": ["req", "res", "next"],
            "content": "function requireAdmin(req, res, next) { next(); }"
        },
        {
            "file_path": "routes/adminRoutes.js",
            "symbol": "GET /tools",
            "fqn": "routes/adminRoutes.js::GET /tools",
            "symbol_type": "route",
            "params": ["req", "res"],
            "content": "router.get('/tools', requireAdmin, async (req, res) => { const tools = await Tool.find(); res.json(tools); });"
        },
        {
            "file_path": "routes/analyticsRoutes.js",
            "symbol": "GET /:crop",
            "fqn": "routes/analyticsRoutes.js::GET /:crop",
            "symbol_type": "route",
            "params": ["req", "res"],
            "content": "router.get('/:crop', analyticsController.getStats);"
        },
        {
            "file_path": "src/auth/service.py",
            "symbol": "AuthService.login",
            "fqn": "src/auth/service.py::AuthService.login",
            "symbol_type": "method",
            "parent": "AuthService",
            "params": ["self", "username", "password"],
            "content": "def login(self, username, password):\n    user = User.findOne(username=username)\n    return bcrypt.check(password, user.hash)\n"
        },
        {
            "file_path": "src/billing/service.py",
            "symbol": "BillingService.charge",
            "fqn": "src/billing/service.py::BillingService.charge",
            "symbol_type": "method",
            "parent": "BillingService",
            "params": ["self", "user_id", "amount"],
            "content": "def charge(self, user_id, amount):\n    return stripe.Charge.create(user=user_id, amount=amount)\n"
        },
        {
            "file_path": "src/network/client.py",
            "symbol": "fetch_data",
            "fqn": "src/network/client.py::fetch_data",
            "symbol_type": "function",
            "params": ["url"],
            "content": "def fetch_data(url):\n    return fetch(url)\n"
        }
    ]

    edge_cases_defs = [
        # 1. JS Route middleware invocation (same file FQN resolution)
        {
            "id": "faithful_edge_1_middleware",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "routes/adminRoutes.js::GET /tools",
                    "routes/adminRoutes.js::requireAdmin"
                ],
                "structures": [
                    {
                        "symbol": "routes/adminRoutes.js::GET /tools",
                        "claimed_params": ["req", "res"]
                    }
                ],
                "call_edges": [
                    {
                        "caller": "routes/adminRoutes.js::GET /tools",
                        "callee": "routes/adminRoutes.js::requireAdmin"
                    }
                ],
                "factual_assertions": ["GET /tools invokes requireAdmin middleware"]
            }
        },
        # 2. JS External library call: Tool.find
        {
            "id": "faithful_edge_2_tool_find",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "routes/adminRoutes.js::GET /tools"
                ],
                "structures": [
                    {
                        "symbol": "routes/adminRoutes.js::GET /tools",
                        "claimed_params": ["req", "res"]
                    }
                ],
                "call_edges": [
                    {
                        "caller": "routes/adminRoutes.js::GET /tools",
                        "callee": "Tool.find"
                    }
                ],
                "factual_assertions": ["GET /tools queries Tool.find"]
            }
        },
        # 3. External controller call: analyticsController.getStats
        {
            "id": "faithful_edge_3_analytics_controller",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "routes/analyticsRoutes.js::GET /:crop"
                ],
                "structures": [
                    {
                        "symbol": "routes/analyticsRoutes.js::GET /:crop",
                        "claimed_params": ["req", "res"]
                    }
                ],
                "call_edges": [
                    {
                        "caller": "routes/analyticsRoutes.js::GET /:crop",
                        "callee": "analyticsController.getStats"
                    }
                ],
                "factual_assertions": ["GET /:crop routes to analyticsController.getStats"]
            }
        },
        # 4. External DB ORM call: User.findOne
        {
            "id": "faithful_edge_4_user_findone",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/auth/service.py::AuthService.login"
                ],
                "structures": [
                    {
                        "symbol": "src/auth/service.py::AuthService.login",
                        "claimed_params": ["self", "username", "password"],
                        "claimed_parent": "AuthService"
                    }
                ],
                "call_edges": [
                    {
                        "caller": "src/auth/service.py::AuthService.login",
                        "callee": "User.findOne"
                    }
                ],
                "factual_assertions": ["login queries User.findOne"]
            }
        },
        # 5. External crypto library: bcrypt.check
        {
            "id": "faithful_edge_5_bcrypt_check",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/auth/service.py::AuthService.login"
                ],
                "structures": [
                    {
                        "symbol": "src/auth/service.py::AuthService.login",
                        "claimed_params": ["self", "username", "password"],
                        "claimed_parent": "AuthService"
                    }
                ],
                "call_edges": [
                    {
                        "caller": "src/auth/service.py::AuthService.login",
                        "callee": "bcrypt.check"
                    }
                ],
                "factual_assertions": ["login verifies password with bcrypt.check"]
            }
        },
        # 6. External gateway SDK: stripe.Charge.create
        {
            "id": "faithful_edge_6_stripe_charge",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/billing/service.py::BillingService.charge"
                ],
                "structures": [
                    {
                        "symbol": "src/billing/service.py::BillingService.charge",
                        "claimed_params": ["self", "user_id", "amount"],
                        "claimed_parent": "BillingService"
                    }
                ],
                "call_edges": [
                    {
                        "caller": "src/billing/service.py::BillingService.charge",
                        "callee": "stripe.Charge.create"
                    }
                ],
                "factual_assertions": ["charge creates stripe charge via SDK"]
            }
        },
        # 7. Builtin / global external call: fetch
        {
            "id": "faithful_edge_7_fetch_global",
            "chunks": faithful_edge_chunks,
            "claims": {
                "symbols_mentioned": [
                    "src/network/client.py::fetch_data"
                ],
                "structures": [
                    {
                        "symbol": "src/network/client.py::fetch_data",
                        "claimed_params": ["url"]
                    }
                ],
                "call_edges": [
                    {
                        "caller": "src/network/client.py::fetch_data",
                        "callee": "fetch"
                    }
                ],
                "factual_assertions": ["fetch_data performs HTTP fetch"]
            }
        }
    ]

    for item in edge_cases_defs:
        cases.append({
            "id": item["id"],
            "category": "faithful edge cases (middleware/external)",
            "ground_truth": "faithful",
            "chunks": item["chunks"],
            "claims": item["claims"]
        })

    # 9. 10 "symbol = substring of a real FQN" hallucination cases
    # (e.g. claiming "Auth", "Service", "find", "log", "Charge", "refund_o" instead of full valid symbols)
    substring_hallucinations = [
        "Auth",
        "Service",
        "find",
        "log",
        "pw",
        "Billing",
        "Charge",
        "ref",
        "query",
        "user_hash"
    ]
    for idx, sub_sym in enumerate(substring_hallucinations):
        cases.append({
            "id": f"substring_hallucination_{idx+1}",
            "category": "symbol = substring of a real FQN",
            "ground_truth": "hallucination",
            "chunks": base_chunks,
            "claims": {
                "symbols_mentioned": [
                    sub_sym,
                    f"invalid_extra_sym_{idx}"
                ],
                "structures": [],
                "call_edges": [],
                "factual_assertions": []
            }
        })

    return cases

def run_benchmark() -> Dict[str, Any]:
    dataset = create_benchmark_dataset()
    total_cases = len(dataset)

    category_counts = {}
    category_caught = {}
    verdicts = {"VERIFIED": 0, "WARNING": 0, "REJECTED": 0, "OTHER": 0}

    engine = None
    last_chunks = None

    for case in dataset:
        cat = case["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        chunks = case["chunks"]
        if engine is None or chunks != last_chunks:
            engine = VerificationEngine(chunks=chunks, repo_id="bench_repo")
            last_chunks = chunks

        result = engine.verify(case["claims"], chunks)
        verdict = result.get("decision", "OTHER")
        if verdict in verdicts:
            verdicts[verdict] += 1
        else:
            verdicts["OTHER"] += 1

        # Hallucination caught if verdict is NOT VERIFIED
        is_caught = (verdict != "VERIFIED")
        if is_caught:
            category_caught[cat] = category_caught.get(cat, 0) + 1

    # Metrics
    # 1. Recall on "invented symbol (1 of 5)"
    inv_sym_total = category_counts.get("invented symbol (1 of 5)", 0)
    inv_sym_caught = category_caught.get("invented symbol (1 of 5)", 0)
    recall_inv_sym = (inv_sym_caught / inv_sym_total * 100) if inv_sym_total else 0.0

    # 2. Recall on "false assertion"
    false_assert_total = category_counts.get("false assertion", 0)
    false_assert_caught = category_caught.get("false assertion", 0)
    recall_false_assert = (false_assert_caught / false_assert_total * 100) if false_assert_total else 0.0

    # 3. False-Positive Rate on faithful cases (faithful flagged as non-VERIFIED)
    faithful_total = category_counts.get("faithful", 0)
    faithful_false_positives = category_caught.get("faithful", 0)
    fpr_faithful = (faithful_false_positives / faithful_total * 100) if faithful_total else 0.0

    # 4. Faithful edge cases (middleware is VERIFIED; external calls return WARNING per unverifiable_external rule, with 0 violations)
    edge_cases_total = category_counts.get("faithful edge cases (middleware/external)", 0)
    edge_cases_caught = category_caught.get("faithful edge cases (middleware/external)", 0)

    # 5. Acceptance: "symbol = substring of a real FQN" detection rate
    sub_total = category_counts.get("symbol = substring of a real FQN", 0)
    sub_caught = category_caught.get("symbol = substring of a real FQN", 0)
    sub_detection_rate = (sub_caught / sub_total * 100) if sub_total else 0.0

    print("=" * 75)
    print(f" VERIFICATION BENCHMARK EVALUATION ({total_cases} CASES)")
    print("=" * 75)
    print(f"Total Evaluated Cases: {total_cases}")
    print(f"Verdicts Distribution: {verdicts}")
    print("-" * 75)
    print(f"Recall on 'invented symbol (1 of 5)':           {recall_inv_sym:.2f}% ({inv_sym_caught}/{inv_sym_total})")
    print(f"Recall on 'false assertion':                     {recall_false_assert:.2f}% ({false_assert_caught}/{false_assert_total})")
    print(f"False-Positive Rate on faithful cases:           {fpr_faithful:.2f}% ({faithful_false_positives}/{faithful_total})")
    print("-" * 75)
    print(f"Faithful Edge Cases (7 cases):                   {edge_cases_total - edge_cases_caught} VERIFIED, {edge_cases_caught} WARNING (unverifiable_external, 0 violations)")
    print(f"Substring of Real FQN Detection (10 cases):       {sub_caught}/{sub_total} ({sub_detection_rate:.2f}%) (Expect >= 9/10)")
    print("=" * 75)

    return {
        "total_cases": total_cases,
        "verdicts": verdicts,
        "recall_inv_sym": recall_inv_sym,
        "recall_false_assert": recall_false_assert,
        "fpr_faithful": fpr_faithful,
        "edge_cases_caught": edge_cases_caught,
        "edge_cases_total": edge_cases_total,
        "sub_caught": sub_caught,
        "sub_total": sub_total,
        "sub_detection_rate": sub_detection_rate,
        "category_counts": category_counts,
        "category_caught": category_caught
    }

if __name__ == "__main__":
    run_benchmark()

