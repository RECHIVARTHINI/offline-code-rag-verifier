import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.chunker import chunker


class ChunkerMetadataTests(unittest.TestCase):
    def test_javascript_express_route_metadata(self):
        doc = {
            "path": "src/routes/auth.js",
            "content": "router.post('/login', async (req, res) => {\n  const user = await findUser(req.body.email);\n  return res.json({ ok: true });\n});\n",
            "extension": ".js",
        }

        chunks = chunker(doc)
        route = next((c for c in chunks if c["symbol_type"] == "route"), None)

        self.assertIsNotNone(route)
        self.assertEqual(route["symbol"], "POST /login")
        self.assertEqual(route["symbol_type"], "route")
        self.assertEqual(route["route_method"], "POST")
        self.assertEqual(route["route_path"], "/login")
        self.assertEqual(route["parent"], "router")
        self.assertEqual(route["file_name"], "auth.js")
        self.assertEqual(route["file_path"], "src/routes/auth.js")
        self.assertIn("router.post('/login'", route["content"])

    def test_python_method_parent_and_function_metadata(self):
        doc = {
            "path": "services/auth.py",
            "content": "class AuthService:\n    def login(self):\n        return 'ok'\n\ndef create_user():\n    return {}\n",
            "extension": ".py",
        }

        chunks = chunker(doc)

        method = next((c for c in chunks if c["symbol"] in {"login", "AuthService.login"}), None)
        func = next((c for c in chunks if c["symbol"] == "create_user"), None)

        self.assertIsNotNone(method)
        self.assertEqual(method["symbol_type"], "method")
        self.assertEqual(method["parent"], "AuthService")
        self.assertIsNotNone(func)
        self.assertEqual(func["symbol_type"], "function")
        self.assertIsNone(func["parent"])

    def test_javascript_class_and_function_metadata(self):
        doc = {
            "path": "src/auth.js",
            "content": "class AuthService {\n  login() {\n    return true;\n  }\n}\n\nfunction createUser() {\n  return true;\n}\n",
            "extension": ".js",
        }

        chunks = chunker(doc)

        method = next((c for c in chunks if c["symbol"] in {"login", "AuthService.login"}), None)
        func = next((c for c in chunks if c["symbol"] == "createUser"), None)

        self.assertIsNotNone(method)
        self.assertEqual(method["symbol_type"], "method")
        self.assertEqual(method["parent"], "AuthService")
        self.assertIsNotNone(func)
        self.assertEqual(func["symbol_type"], "function")

    def test_mongoose_schema_definition(self):
        """1. Mongoose new mongoose.Schema({...}) definitions (symbol = model name, symbol_type='model')."""
        doc = {
            "path": "models/Crop.js",
            "content": """
const mongoose = require('mongoose');

const cropSchema = new mongoose.Schema({
    cropName: { type: String, required: true },
    quantity: { type: Number, required: true }
});

module.exports = mongoose.model('Crop', cropSchema);
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        schema_chunk = next((c for c in chunks if c["symbol"] == "Crop"), None)
        self.assertIsNotNone(schema_chunk, "Expected model chunk with symbol 'Crop'")
        self.assertEqual(schema_chunk["symbol"], "Crop")
        self.assertEqual(schema_chunk["symbol_type"], "model")
        self.assertEqual(schema_chunk["fqn"], "models/Crop.js::Crop")

        export_chunk = next((c for c in chunks if c["symbol"] == "module.exports"), None)
        self.assertIsNone(export_chunk, "Expected module.exports duplicate chunk to be dropped")

    def test_exports_arrow_and_function_handler(self):
        """1. exports.name = (async) arrow/function."""
        doc = {
            "path": "controllers/analyticsController.js",
            "content": """
exports.getStats = async (req, res) => {
    return res.json({ stats: true });
};

exports.helper = function(val) {
    return val * 2;
};
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        stats_chunk = next((c for c in chunks if c["symbol"] == "getStats"), None)
        self.assertIsNotNone(stats_chunk, "Expected exports.getStats chunk")
        self.assertEqual(stats_chunk["symbol_type"], "function")
        self.assertEqual(stats_chunk["params"], ["req", "res"])
        self.assertEqual(stats_chunk["fqn"], "controllers/analyticsController.js::getStats")

        helper_chunk = next((c for c in chunks if c["symbol"] == "helper"), None)
        self.assertIsNotNone(helper_chunk, "Expected exports.helper chunk")
        self.assertEqual(helper_chunk["symbol_type"], "function")
        self.assertEqual(helper_chunk["params"], ["val"])

    def test_const_arrow_and_function_declarations(self):
        """1. const name = (async) (...) => {} / = function."""
        doc = {
            "path": "utils/math.js",
            "content": """
const add = (a, b) => a + b;
const multiply = async (x, y) => { return x * y; };
const divide = function(num, den) { return num / den; };
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        add_chunk = next((c for c in chunks if c["symbol"] == "add"), None)
        self.assertIsNotNone(add_chunk)
        self.assertEqual(add_chunk["symbol_type"], "function")
        self.assertEqual(add_chunk["params"], ["a", "b"])

        mult_chunk = next((c for c in chunks if c["symbol"] == "multiply"), None)
        self.assertIsNotNone(mult_chunk)
        self.assertEqual(mult_chunk["symbol_type"], "function")
        self.assertEqual(mult_chunk["params"], ["x", "y"])

        div_chunk = next((c for c in chunks if c["symbol"] == "divide"), None)
        self.assertIsNotNone(div_chunk)
        self.assertEqual(div_chunk["symbol_type"], "function")
        self.assertEqual(div_chunk["params"], ["num", "den"])

    def test_axios_get_and_map_get_not_treated_as_routes(self):
        """2. axios.get('/api/x') and map.get('key') / cache.get('key') must NOT become fake routes."""
        doc = {
            "path": "client/api.js",
            "content": """
const response = axios.get('/api/users');
const item = map.get('user_id');
const val = cache.get('token');
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        route_chunks = [c for c in chunks if c["symbol_type"] == "route"]
        self.assertEqual(len(route_chunks), 0, "axios.get and map.get must not become route chunks")

    def test_merged_route_map_chunk_for_use_lines(self):
        """3. Do not emit one chunk per app.use line; merge them into a single route map chunk."""
        doc = {
            "path": "server.js",
            "content": """
const express = require('express');
const app = express();

app.use('/api/tools', toolRoutes);
app.use('/api/users', userRoutes);
app.use('/api/crops', cropRoutes);

app.get('/health', (req, res) => res.json({ ok: true }));
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        route_map = next((c for c in chunks if c["symbol"] == "route map"), None)
        self.assertIsNotNone(route_map, "Expected merged route map chunk")
        self.assertEqual(route_map["symbol_type"], "route_map")
        self.assertIn("app.use('/api/tools'", route_map["content"])
        self.assertIn("app.use('/api/users'", route_map["content"])
        self.assertIn("app.use('/api/crops'", route_map["content"])

        # No individual "USE ..." route chunk should exist
        use_chunks = [c for c in chunks if c.get("route_method") == "USE"]
        self.assertEqual(len(use_chunks), 0, "Must not emit individual USE route chunks")

        # Standard GET route is still preserved
        health_route = next((c for c in chunks if c["symbol"] == "GET /health"), None)
        self.assertIsNotNone(health_route)

    def test_no_nested_routes_inside_functions(self):
        """4. Do not create route chunks nested inside function chunks."""
        doc = {
            "path": "routes/setup.js",
            "content": """
function registerDynamicRoutes(router) {
    router.get('/dynamic-test', (req, res) => res.send('ok'));
}
""",
            "extension": ".js",
        }

        chunks = chunker(doc)
        route_chunks = [c for c in chunks if c["symbol_type"] == "route"]
        self.assertEqual(len(route_chunks), 0, "Routes inside function declarations must not become route chunks")
        fn_chunk = next((c for c in chunks if c["symbol"] == "registerDynamicRoutes"), None)
        self.assertIsNotNone(fn_chunk, "Function chunk should still be created")


if __name__ == "__main__":
    unittest.main()

