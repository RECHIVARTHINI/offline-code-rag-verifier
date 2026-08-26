import unittest

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

        method = next((c for c in chunks if c["symbol"] == "login"), None)
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

        method = next((c for c in chunks if c["symbol"] == "login"), None)
        func = next((c for c in chunks if c["symbol"] == "createUser"), None)

        self.assertIsNotNone(method)
        self.assertEqual(method["symbol_type"], "method")
        self.assertEqual(method["parent"], "AuthService")
        self.assertIsNotNone(func)
        self.assertEqual(func["symbol_type"], "function")


if __name__ == "__main__":
    unittest.main()
