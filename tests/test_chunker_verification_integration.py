"""
Integration Test Suite: Real Tree-sitter Chunker -> VerificationEngine
Validates that Tree-sitter extracts deterministic FQNs, parameter lists, and parent classes.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.chunker import chunker
from backend.services.verification import VerificationEngine

py_doc = {
    "path": "src/services/payment_service.py",
    "extension": ".py",
    "content": """
class PaymentProcessor:
    def __init__(self, api_key, environment="sandbox"):
        self.api_key = api_key
        self.env = environment

    def process_transaction(self, user_id, amount, currency="USD"):
        token = self.generate_token(user_id)
        return self.dispatch_gateway(token, amount)

    def generate_token(self, user_id):
        return f"tok_{user_id}"

def refund_order(order_id, reason):
    return f"refunded_{order_id}"
"""
}

js_doc = {
    "path": "routes/billingRoutes.js",
    "extension": ".js",
    "content": """
const express = require('express');
const router = express.Router();

router.post('/checkout', async (req, res) => {
    return res.json({ status: 'ok' });
});

router.get('/history/:userId', billingController.getHistory);

function validateBillingData(payload) {
    return payload != null;
}
"""
}

def test_py_chunker_fqns_and_parameters():
    chunks = chunker(py_doc)
    assert len(chunks) == 5 # Class + 3 methods + 1 function
    
    # Check Class chunk
    cls_chunk = next(c for c in chunks if c["symbol"] == "PaymentProcessor")
    assert cls_chunk["fqn"] == "src/services/payment_service.py::PaymentProcessor"
    assert cls_chunk["symbol_type"] == "class"

    # Check Method chunk with parameters
    tx_chunk = next(c for c in chunks if c["symbol"] == "PaymentProcessor.process_transaction")
    assert tx_chunk["fqn"] == "src/services/payment_service.py::PaymentProcessor.process_transaction"
    assert tx_chunk["symbol_type"] == "method"
    assert tx_chunk["parent"] == "PaymentProcessor"
    assert "user_id" in tx_chunk["params"]
    assert "amount" in tx_chunk["params"]
    assert "currency" in tx_chunk["params"]

    # Check standalone function
    refund_chunk = next(c for c in chunks if c["symbol"] == "refund_order")
    assert refund_chunk["fqn"] == "src/services/payment_service.py::refund_order"
    assert refund_chunk["symbol_type"] == "function"
    assert refund_chunk["params"] == ["order_id", "reason"]

def test_js_chunker_routes_and_functions():
    chunks = chunker(js_doc)
    assert len(chunks) == 3 # 2 routes + 1 function
    
    post_route = next(c for c in chunks if c["symbol"] == "POST /checkout")
    assert post_route["fqn"] == "routes/billingRoutes.js::POST /checkout"
    assert post_route["symbol_type"] == "route"
    assert post_route["route_method"] == "POST"

    get_route = next(c for c in chunks if c["symbol"] == "GET /history/:userId")
    assert get_route["fqn"] == "routes/billingRoutes.js::GET /history/:userId"

    func_chunk = next(c for c in chunks if c["symbol"] == "validateBillingData")
    assert func_chunk["fqn"] == "routes/billingRoutes.js::validateBillingData"
    assert func_chunk["params"] == ["payload"]

def test_chunker_output_with_verification_engine():
    all_chunks = chunker(py_doc) + chunker(js_doc)
    engine = VerificationEngine(all_chunks, repo_id="integration_test")
    
    # 1. Valid Claim Test
    valid_claims = {
        "symbols_mentioned": [
            "src/services/payment_service.py::PaymentProcessor.process_transaction",
            "routes/billingRoutes.js::POST /checkout"
        ],
        "structures": [
            {
                "symbol": "src/services/payment_service.py::PaymentProcessor.process_transaction",
                "claimed_params": ["self", "user_id", "amount", "currency"],
                "claimed_parent": "PaymentProcessor"
            }
        ],
        "call_edges": [
            {
                "caller": "src/services/payment_service.py::PaymentProcessor.process_transaction",
                "callee": "generate_token"
            }
        ],
        "factual_assertions": ["process_transaction generates token and dispatches gateway."]
    }
    res = engine.verify(valid_claims, all_chunks)
    assert res["trust_score"] >= 0.80
    assert res["decision"] == "VERIFIED"

    # 2. Structural Mismatch Test (Wrong parameters)
    bad_claims = {
        "symbols_mentioned": ["src/services/payment_service.py::PaymentProcessor.process_transaction"],
        "structures": [
            {
                "symbol": "src/services/payment_service.py::PaymentProcessor.process_transaction",
                "claimed_params": ["wrong_param_1", "wrong_param_2"]
            }
        ],
        "call_edges": [],
        "factual_assertions": []
    }
    bad_res = engine.verify(bad_claims, all_chunks)
    assert bad_res["layers"]["layer2_structural"]["score"] == 0.0
    assert len(bad_res["layers"]["layer2_structural"]["violations"]) > 0

if __name__ == "__main__":
    test_py_chunker_fqns_and_parameters()
    print("PASS: test_py_chunker_fqns_and_parameters")
    test_js_chunker_routes_and_functions()
    print("PASS: test_js_chunker_routes_and_functions")
    test_chunker_output_with_verification_engine()
    print("PASS: test_chunker_output_with_verification_engine")
    print("\nALL CHUNKER & VERIFICATION INTEGRATION TESTS PASSED!")
