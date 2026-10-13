"""
Deterministic Verification Engine with Strict Fail-Closed Semantics,
Exact Fully Qualified Name (FQN) Resolution, and Direct Call Edge Enforcement.
"""
import ast
import os
import json
from pathlib import Path
import re
import networkx as nx
from typing import Dict, Any, List, Set, Optional


ALLOWLIST_BUILTINS = {
    "req", "res", "next", "process", "json", "promise", "string", "number",
    "date", "math", "console", "fetch", "async", "await",
    "find", "findone", "findbyid", "findbyidandupdate", "findbyidanddelete",
    "save", "countdocuments", "select", "sort"
}


def _is_code_shaped(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if t.endswith("()"):
        t = t[:-2].strip()

    # Dotted names: User.select, req.body, a.b
    if "." in t:
        return True

    # Hyphenated package names: escape-html, react-dom
    if "-" in t:
        return True

    # snake_case: user_id, order_id
    if "_" in t:
        return True

    # Scoped package or path with extension
    if t.startswith("@") or re.search(r'\.[a-zA-Z0-9]+$', t):
        return True

    # Compound paths with slash: if it's like "async/await", ignore it unless it's a file path
    if "/" in t:
        if re.search(r'\.[a-zA-Z0-9]+$', t) or t.startswith(("/", "./", "../")):
            return True
        parts = t.split("/")
        if all(re.match(r'^[a-zA-Z]+$', p) for p in parts):
            return False
        return True

    # camelCase or PascalCase with multiple words (deleteCrop, PaymentService, UserCard)
    if re.search(r'[a-z][A-Z]', t):
        return True

    return False


def _is_allowlisted_token(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if t.endswith("()"):
        t = t[:-2].strip()
    t_lower = t.lower()
    if t_lower in ALLOWLIST_BUILTINS:
        return True
    for pfx in ("mongoose.", "express.", "req.", "res.", "process.", "console.", "json.", "promise."):
        if t_lower.startswith(pfx):
            return True
    return False


def _is_callee_allowlisted(callee: str) -> bool:
    if not callee:
        return False
    c = callee.strip()
    if c.endswith("()"):
        c = c[:-2].strip()
    c_lower = c.lower()
    if c_lower in ALLOWLIST_BUILTINS:
        return True
    for pfx in ("mongoose.", "express.", "req.", "res.", "console.", "json.", "promise.", "process."):
        if c_lower.startswith(pfx):
            return True
    if "." in c:
        method_name = c.split(".")[-1].lower()
        if method_name in ALLOWLIST_BUILTINS or method_name in {
            "populate", "exec", "then", "catch"
        }:
            return True
    return False


def _build_evidence_set(retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence_fqns: Set[str] = set()
    evidence_symbols: Set[str] = set()
    evidence_files: Set[str] = set()
    evidence_routes: Set[str] = set()
    evidence_tokens: Set[str] = set()
    raw_texts: List[str] = []

    for c in retrieved_chunks:
        if not c:
            continue
        p = c
        if hasattr(c.get("point"), "payload") and isinstance(c.get("point").payload, dict):
            p = c.get("point").payload
        elif isinstance(c.get("payload"), dict):
            p = c.get("payload")

        fp = (p.get("file_path") or "").replace("\\", "/").strip("/")
        if fp:
            evidence_files.add(fp)
            evidence_files.add(Path(fp).name)
            evidence_files.add(Path(fp).stem)
            for part in fp.split("/"):
                if part:
                    evidence_files.add(part)

        sym = p.get("symbol") or ""
        if sym:
            evidence_symbols.add(sym)

        fqn = p.get("fqn") or ""
        if fqn:
            evidence_fqns.add(fqn)
        elif fp and sym:
            evidence_fqns.add(f"{fp}::{sym}")

        # Route matching
        for s in (sym, fqn):
            if "::" in s:
                s = s.split("::", 1)[1]
            m = re.match(r'^(GET|POST|PUT|PATCH|DELETE|USE)\s+(/.+)$', s)
            if m:
                method, rpath = m.groups()
                evidence_routes.add(f"{method} {rpath}")
                evidence_routes.add(rpath)
                evidence_routes.add(rpath.lstrip("/"))
                clean_rpath = re.sub(r'/:[a-zA-Z0-9_]+', '', rpath)
                if clean_rpath:
                    evidence_routes.add(clean_rpath)

        content = p.get("content") or ""
        if content:
            raw_texts.append(content)
            content_routes = re.findall(r'(?:router|app)\.(?:get|post|put|patch|delete|use)\s*\(\s*[\'"]([^\'"]+)[\'"]', content)
            for cr in content_routes:
                evidence_routes.add(cr)
                evidence_routes.add(cr.lstrip("/"))

            tokens = re.findall(r'[A-Za-z0-9_@][\w./-]*', content)
            for t in tokens:
                evidence_tokens.add(t)
                if "." in t:
                    parts = t.split(".")
                    for part in parts:
                        if part:
                            evidence_tokens.add(part)
                    for i in range(1, len(parts)):
                        evidence_tokens.add(".".join(parts[:i]))
                        evidence_tokens.add(".".join(parts[i:]))
                if "/" in t:
                    for part in t.split("/"):
                        if part:
                            evidence_tokens.add(part)
                if "-" in t:
                    evidence_tokens.add(t)

            words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', content)
            for w in words:
                evidence_tokens.add(w)

    return {
        "evidence_fqns": evidence_fqns,
        "evidence_symbols": evidence_symbols,
        "evidence_files": evidence_files,
        "evidence_routes": evidence_routes,
        "evidence_tokens": evidence_tokens,
        "raw_text": " ".join(raw_texts)
    }


class VerificationEngine:
    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None, repo_id: str = "default"):
        self.repo_id = repo_id
        # Registry mapping FQN (file_path::Symbol) -> SymbolMetadata
        self.symbol_registry: Dict[str, Dict[str, Any]] = {}
        # Fast lookup set of all valid exact FQNs
        self.valid_fqns: Set[str] = set()
        # Mapping bare_name -> Set[FQN] for unambiguous resolution
        self.bare_to_fqn: Dict[str, Set[str]] = {}
        # All indexed repo files
        self.repo_files: Set[str] = set()
        # Deterministic Directed Call Graph
        self.call_graph = nx.DiGraph()

        if chunks:
            self.index_chunks(chunks)

    def _make_fqn(self, file_path: str, symbol: str) -> str:
        clean_file = file_path.replace("\\", "/").strip("/")
        return f"{clean_file}::{symbol}"

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes symbols with exact FQNs and builds deterministic call graph."""
        # Pass 1: Index all valid FQNs, repo files, and symbol registries first
        for c in chunks:
            file_path = c.get("file_path", "")
            symbol = c.get("symbol", "")
            if not file_path or not symbol:
                continue

            clean_fp = file_path.replace("\\", "/").strip("/")
            self.repo_files.add(clean_fp)
            self.repo_files.add(Path(clean_fp).name)
            self.repo_files.add(Path(clean_fp).stem)

            fqn = c.get("fqn") or self._make_fqn(file_path, symbol)
            self.valid_fqns.add(fqn)

            if symbol not in self.bare_to_fqn:
                self.bare_to_fqn[symbol] = set()
            self.bare_to_fqn[symbol].add(fqn)

            # If symbol is Class.method, also index bare method name
            if "." in symbol:
                bare_method = symbol.split(".")[-1]
                if bare_method not in self.bare_to_fqn:
                    self.bare_to_fqn[bare_method] = set()
                self.bare_to_fqn[bare_method].add(fqn)

            params = c.get("params", [])
            self.symbol_registry[fqn] = {
                "fqn": fqn,
                "symbol": symbol,
                "file_path": file_path,
                "symbol_type": c.get("symbol_type", "function"),
                "parent": c.get("parent"),
                "params": params,
                "content": c.get("content", ""),
                "start_line": c.get("start_line"),
                "end_line": c.get("end_line"),
            }

        # Pass 2: Extract deterministic calls ensuring callee nodes resolve to FQNs
        for c in chunks:
            file_path = c.get("file_path", "")
            symbol = c.get("symbol", "")
            if not file_path or not symbol:
                continue

            fqn = c.get("fqn") or self._make_fqn(file_path, symbol)
            content = c.get("content", "")
            if content:
                self._extract_deterministic_calls(content, fqn, file_path)

    def _resolve_callee_fqn(self, callee_name: str, file_path: str, owner_class: Optional[str] = None) -> Optional[str]:
        """Resolves a callee to a real FQN in the same file first, then to a unique repo-wide symbol."""
        clean_file = file_path.replace("\\", "/").strip("/")

        # 1. Same-file method in same class (if owner_class known)
        if owner_class:
            cand = f"{clean_file}::{owner_class}.{callee_name}"
            if cand in self.valid_fqns:
                return cand

        # 2. Same-file function or symbol
        cand = f"{clean_file}::{callee_name}"
        if cand in self.valid_fqns:
            return cand

        # 3. Same-file check across all valid_fqns
        same_file_matches = [
            f for f in self.valid_fqns
            if f.startswith(f"{clean_file}::") and (
                f.endswith(f".{callee_name}") or f.endswith(f"::{callee_name}")
            )
        ]
        if len(same_file_matches) == 1:
            return same_file_matches[0]

        # 4. Unique repo-wide symbol
        if callee_name in self.bare_to_fqn:
            matches = self.bare_to_fqn[callee_name]
            if len(matches) == 1:
                return next(iter(matches))

        # Check Class.method repo-wide
        if "." in callee_name:
            repo_matches = [
                f for f in self.valid_fqns
                if f.endswith(f"::{callee_name}") or f.endswith(f".{callee_name}")
            ]
            if len(repo_matches) == 1:
                return repo_matches[0]

        return None

    def _extract_deterministic_calls(self, code_content: str, caller_fqn: str, file_path: str):
        clean_file = file_path.replace("\\", "/").strip("/")
        self.call_graph.add_node(caller_fqn, file=clean_file)
        
        owner_class = None
        if "::" in caller_fqn:
            sym_part = caller_fqn.split("::", 1)[1]
            if "." in sym_part:
                owner_class = sym_part.split(".", 1)[0]

        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    callee_name = None
                    if isinstance(node.func, ast.Name):
                        callee_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        callee_name = node.func.attr
                    
                    if callee_name and callee_name not in ["print", "len", "range", "isinstance", "getattr"]:
                        resolved_callee = self._resolve_callee_fqn(callee_name, file_path, owner_class)
                        if resolved_callee:
                            self.call_graph.add_node(resolved_callee)
                            self.call_graph.add_edge(caller_fqn, resolved_callee, type="direct_call")
        except Exception:
            calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
            for callee in set(calls):
                if callee not in ["if", "for", "while", "catch", "function", "return", "require", "import"]:
                    resolved_callee = self._resolve_callee_fqn(callee, file_path, owner_class)
                    if resolved_callee:
                        self.call_graph.add_node(resolved_callee)
                        self.call_graph.add_edge(caller_fqn, resolved_callee, type="direct_call")
            
            # Extract JS route callback handlers: router.get('/path', controller.handler) or middleware
            route_callbacks = re.findall(r'(?:router|app)\.(?:get|post|put|patch|delete|use)\s*\([^,]+,\s*([a-zA-Z0-9_.]+)', code_content)
            for cb in route_callbacks:
                cb_clean = cb.strip()
                # Try resolving the callback as a symbol (same-file first, then repo-wide)
                resolved_cb = self._resolve_callee_fqn(cb_clean, file_path, owner_class)
                if not resolved_cb and "." in cb_clean:
                    method_part = cb_clean.split(".")[-1]
                    resolved_cb = self._resolve_callee_fqn(method_part, file_path, owner_class)
                if resolved_cb:
                    self.call_graph.add_node(resolved_cb)
                    self.call_graph.add_edge(caller_fqn, resolved_cb, type="direct_call")

    def _resolve_symbol(self, claimed_symbol: str, target_file: Optional[str] = None) -> Optional[str]:
        """
        Resolves a claimed symbol into an exact deterministic FQN.
        Strictly allows only:
        1. Exact FQN
        2. File-suffix + exact symbol
        3. Exact bare name (unambiguous, or disambiguated by target_file)
        4. Class.method suffix
        Ambiguous matches return None. No substring matching fallbacks.
        """
        raw = claimed_symbol.replace("\\", "/")
        if "::" in raw:
            file_part, sym_part = raw.split("::", 1)
            file_part = file_part.lstrip("/")
            clean_sym = f"{file_part}::{sym_part}"
        else:
            clean_sym = raw.lstrip("/")

        clean_tf = target_file.replace("\\", "/").lstrip("/") if target_file else None
        
        # 1. Exact FQN Match
        if clean_sym in self.valid_fqns:
            return clean_sym

        # 2. File-suffix + exact symbol (e.g. "analyticsRoutes.js::GET /:crop" or "service.py::AuthService.login")
        if "::" in clean_sym:
            file_part, sym_part = clean_sym.split("::", 1)
            matches = []
            for fqn in self.valid_fqns:
                if "::" in fqn:
                    fqn_file, fqn_sym = fqn.split("::", 1)
                    file_matched = (
                        fqn_file == file_part or
                        fqn_file.endswith("/" + file_part)
                    )
                    if file_matched and fqn_sym == sym_part:
                        matches.append(fqn)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                if clean_tf:
                    tf_matches = [m for m in matches if m.split("::", 1)[0].endswith(clean_tf)]
                    if len(tf_matches) == 1:
                        return tf_matches[0]
                return None  # Ambiguous match returns None

        # Helper check: If target_file provided, exact target_file::symbol
        if clean_tf:
            cand = f"{clean_tf}::{clean_sym}"
            if cand in self.valid_fqns:
                return cand

        # 3. Exact bare name (unambiguous, or disambiguated by target_file)
        if clean_sym in self.bare_to_fqn:
            matches = list(self.bare_to_fqn[clean_sym])
            if clean_tf:
                filtered = [m for m in matches if m.split("::", 1)[0].endswith(clean_tf)]
                if len(filtered) == 1:
                    return filtered[0]
                elif len(filtered) > 1:
                    return None
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                return None

        # 4. Class.method suffix (e.g. "AuthService.login" or "service.py" context)
        if "." in clean_sym:
            matches = [
                fqn for fqn in self.valid_fqns
                if fqn.endswith(f"::{clean_sym}")
            ]
            if clean_tf:
                filtered = [m for m in matches if m.split("::", 1)[0].endswith(clean_tf)]
                if len(filtered) == 1:
                    return filtered[0]
                elif len(filtered) > 1:
                    return None
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                return None

        return None

    def verify(
        self,
        claims: Optional[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        ollama_failed: bool = False,
        answer_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes 4-Layer Verification with Explicit Gating Rules.
        FAIL CLOSED: Returns trust_score = 0.0 if claims are absent, malformed, or LLM errored.
        """
        if ollama_failed or (claims is None and answer_text is None) or (claims is not None and not isinstance(claims, dict)):
            return {
                "trust_score": 0.0,
                "trust_percentage": 0,
                "decision": "REJECTED_UNVERIFIABLE",
                "badge_color": "red",
                "rejection_reason": "Ollama failed or output lacked valid machine-verifiable <claims> schema.",
                "layers": {
                    "layer1_existence": {"score": 0.0, "passed": False, "status": "failed", "reason": "Claims absent", "missing_symbols": [], "ungrounded_prose_identifiers": [], "total_checked": 0},
                    "layer2_structural": {"score": 0.0, "passed": False, "status": "failed", "reason": "Claims absent", "violations": []},
                    "layer3_callgraph": {"score": 0.0, "passed": False, "status": "failed", "reason": "Claims absent", "violations": [], "unverifiable_external": []},
                    "layer4_evidence": {"score": 0.0, "passed": False, "status": "failed", "reason": "Claims absent", "unsupported": []}
                }
            }

        safe_claims = claims if isinstance(claims, dict) else {}
        symbols = list(safe_claims.get("symbols_mentioned", []))
        structures = safe_claims.get("structures", [])
        call_edges = safe_claims.get("call_edges", [])
        assertions = safe_claims.get("factual_assertions", [])

        # Extract and verify identifiers from answer prose if provided
        ungrounded_prose = []
        if answer_text:
            context_text = " ".join([
                (c.get("point").payload.get("content", "") + " " + 
                 c.get("point").payload.get("file_path", "") + " " + 
                 c.get("point").payload.get("symbol", "") + " " + 
                 c.get("point").payload.get("fqn", ""))
                if hasattr(c.get("point"), "payload")
                else (c.get("content", "") + " " + c.get("file_path", "") + " " + c.get("symbol", "") + " " + c.get("fqn", ""))
                for c in retrieved_chunks if c
            ])

            # Find package dependencies in repo's package.json
            repo_packages = set()
            try:
                candidate_pkg_paths = []
                temp_dir = os.environ.get("TEMP")
                if temp_dir:
                    candidate_pkg_paths.extend(Path(temp_dir).glob("codebase*/package.json"))
                # Also check local project / workspace
                candidate_pkg_paths.extend(Path(".").glob("**/package.json"))
                for pkg_p in candidate_pkg_paths:
                    if pkg_p.is_file():
                        try:
                            data = json.loads(pkg_p.read_text(encoding="utf-8"))
                            for dep_key in ("dependencies", "devDependencies", "peerDependencies"):
                                deps = data.get(dep_key, {})
                                if isinstance(deps, dict):
                                    repo_packages.update(deps.keys())
                        except Exception:
                            pass
            except Exception:
                pass

            # Only consider backticked strings matching ^[A-Za-z_][\w.]*(\(\))?$ (no spaces)
            # OR package-like tokens (hyphens, slashes, e.g. escape-html, react-dom/server) in backticks or require()/import context
            backticked = re.findall(r'`([^`\n]+)`', answer_text)

            # require() or import ... from ... context
            req_matches = re.findall(r'require\s*\(\s*[\'"]([^\'"\r\n]+)[\'"]\s*\)', answer_text)
            imp_matches = re.findall(r'(?:import\s+.*?from\s+|from\s+)[\'"]([^\'"\r\n]+)[\'"]', answer_text)

            candidates = set()
            package_candidates = set()

            for b in backticked:
                b_clean = b.strip()
                # Package-like tokens (containing '-' or '/') e.g. escape-html, react-dom/server
                if re.match(r'^[A-Za-z0-9_@][\w./-]*$', b_clean) and ('-' in b_clean or '/' in b_clean):
                    package_candidates.add(b_clean)
                    continue

                # 1. Standard identifiers must match ^[A-Za-z_][\w.]*(\(\))?$ (no spaces)
                if not re.match(r'^[A-Za-z_][\w.]*(?:\(\))?$', b_clean):
                    continue
                # 2. Ignore req.params.*, req.query.*, req.body.*
                if b_clean.startswith(("req.params", "req.query", "req.body")):
                    continue
                # 3. Ignore single capitalized words (e.g. ID, General, User, Post)
                if re.match(r'^[A-Z][a-zA-Z0-9]*$', b_clean) and "." not in b_clean and "(" not in b_clean:
                    continue

                candidates.add(b_clean)

            for req in req_matches + imp_matches:
                req_clean = req.strip()
                if re.match(r'^[A-Za-z0-9_@][\w./-]*$', req_clean):
                    package_candidates.add(req_clean)

            for cand in sorted(candidates):
                base_name = cand[:-2] if cand.endswith("()") else cand
                cand_core = base_name.split("::")[-1].split("/")[-1].split(".")[-1]
                if (base_name in context_text) or (cand_core in context_text) or (base_name in repo_packages) or (cand_core in repo_packages):
                    if cand not in symbols:
                        symbols.append(cand)
                else:
                    ungrounded_prose.append(cand)

            for pkg in sorted(package_candidates):
                # Valid if package appears in retrieved chunks OR repo's package.json
                pkg_base = pkg.split("/")[0] if not pkg.startswith("@") else "/".join(pkg.split("/")[:2])
                if (pkg in context_text) or (pkg in repo_packages) or (pkg_base in repo_packages):
                    if pkg not in symbols:
                        symbols.append(pkg)
                else:
                    ungrounded_prose.append(pkg)

        if not symbols and not structures and not call_edges and not assertions and not ungrounded_prose:
            return {
                "trust_score": 0.0,
                "trust_percentage": 0,
                "decision": "REJECTED_EMPTY_CLAIMS",
                "badge_color": "red",
                "rejection_reason": "No verifiable code claims extracted from LLM output.",
                "layers": {
                    "layer1_existence": {"score": None, "passed": None, "status": "not_checked", "missing_symbols": [], "ungrounded_prose_identifiers": [], "total_checked": 0},
                    "layer2_structural": {"score": None, "passed": None, "status": "not_checked", "violations": []},
                    "layer3_callgraph": {"score": None, "passed": None, "status": "not_checked", "violations": [], "unverifiable_external": []},
                    "layer4_evidence": {"score": None, "passed": None, "status": "not_checked", "unsupported": []}
                }
            }

        context_files = {
            (c.get("point").payload.get("file_path") if hasattr(c.get("point"), "payload") 
             else c.get("file_path", "")).replace("\\", "/").strip("/")
            for c in retrieved_chunks if c
        }

        l1 = self._verify_layer1(symbols, context_files, retrieved_chunks=retrieved_chunks, ungrounded_prose=ungrounded_prose)
        l2 = self._verify_layer2(structures, context_files)
        l3 = self._verify_layer3(call_edges, context_files)
        l4 = self._verify_layer4(assertions, retrieved_chunks)

        # Collect counts from Layer 1
        invented_tokens = l1.get("invented_tokens", [])
        exists_not_retrieved = l1.get("exists_not_retrieved", [])
        num_invented = len(invented_tokens)
        has_exists_not_retrieved = len(exists_not_retrieved) > 0
        total_checked = l1.get("total_checked", 0)
        invented_ratio = (num_invented / total_checked) if total_checked > 0 else 0.0

        # Check for unverifiable external claims in Layer 3
        has_unverifiable_external = bool(l3.get("unverifiable_external"))

        # Layer pass statuses
        l1_passed = (l1["status"] == "not_checked") or (num_invented == 0 and not has_exists_not_retrieved)
        l2_passed = (l2["status"] == "not_checked") or (l2.get("passed") is True)
        l3_passed = (l3["status"] == "not_checked") or (l3.get("passed") is True)
        l4_passed = (l4["status"] == "not_checked") or (l4.get("passed") is True)

        all_checked_passed = l1_passed and l2_passed and l3_passed and l4_passed

        # Explicit Gating Rules:
        # - VERIFIED: no INVENTED tokens, no exists_not_retrieved, and no L2/L3/L4 failure.
        # - WARNING: exists_not_retrieved, unknown unindexed callees (unverifiable_external), OR exactly 1 INVENTED token.
        # - REJECTED: >= 2 INVENTED tokens OR >= 30% of tokens are INVENTED.
        # (Drop "at least 2 claim types" rule: missing claim type is "not_checked", not a penalty)

        if num_invented >= 2 or (total_checked > 0 and invented_ratio >= 0.30):
            decision = "REJECTED"
            badge = "red"
        elif (has_exists_not_retrieved or 
              has_unverifiable_external or 
              num_invented == 1 or 
              not all_checked_passed):
            decision = "WARNING"
            badge = "amber"
        else:
            decision = "VERIFIED"
            badge = "green"

        # Keep trust_score as a number; exclude layers with score=None from trust-score aggregation
        total_weight = 0.0
        weighted_score_sum = 0.0
        weights_map = {0: 0.30, 1: 0.25, 2: 0.25, 3: 0.20}
        layers_list = [l1, l2, l3, l4]

        for idx, layer in enumerate(layers_list):
            if layer["score"] is not None:
                w = weights_map[idx]
                total_weight += w
                weighted_score_sum += w * layer["score"]

        base_score = (weighted_score_sum / total_weight) if total_weight > 0 else 0.0

        if decision == "VERIFIED":
            final_trust = round(max(0.80, min(1.0, base_score)), 3)
        elif decision == "WARNING":
            final_trust = round(min(0.75, max(0.50, base_score)), 3)
        else:
            final_trust = round(min(0.40, max(0.0, base_score * 0.40)), 3)

        return {
            "trust_score": final_trust,
            "trust_percentage": int(final_trust * 100),
            "decision": decision,
            "badge_color": badge,
            "layers": {
                "layer1_existence": l1,
                "layer2_structural": l2,
                "layer3_callgraph": l3,
                "layer4_evidence": l4
            }
        }

    def _verify_layer1(
        self,
        symbols: List[str],
        context_files: Set[str],
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        ungrounded_prose: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        ungrounded = list(ungrounded_prose) if ungrounded_prose else []
        all_syms = list(dict.fromkeys(symbols + ungrounded))
        if not all_syms:
            return {
                "score": None,
                "passed": None,
                "status": "not_checked",
                "missing_symbols": [],
                "grounded_tokens": [],
                "exists_not_retrieved": [],
                "invented_tokens": [],
                "ungrounded_prose_identifiers": [],
                "total_checked": 0
            }

        ev = _build_evidence_set(retrieved_chunks or [])
        ev_fqns = ev["evidence_fqns"]
        ev_symbols = ev["evidence_symbols"]
        ev_files = ev["evidence_files"]
        ev_routes = ev["evidence_routes"]
        ev_tokens = ev["evidence_tokens"]
        ev_raw = ev["raw_text"]

        grounded = []
        exists_not_retrieved = []
        invented = []

        for sym in all_syms:
            s_clean = sym.strip()
            if not s_clean:
                continue
            s_base = s_clean[:-2].strip() if s_clean.endswith("()") else s_clean
            s_lower = s_base.lower()

            # Library allowlist is never INVENTED, treated as grounded/allowed
            if _is_allowlisted_token(s_base):
                grounded.append(sym)
                continue

            # (a) Resolves via _resolve_symbol to an evidence chunk, or matches evidence FQN / symbol
            resolved_in_evidence = False
            if s_base in ev_fqns or s_base in ev_symbols:
                resolved_in_evidence = True
            else:
                for f in context_files:
                    r_fqn = self._resolve_symbol(s_base, f)
                    if r_fqn and (r_fqn in ev_fqns or not ev_fqns):
                        resolved_in_evidence = True
                        break
                if not resolved_in_evidence:
                    r_fqn = self._resolve_symbol(s_base)
                    if r_fqn and (r_fqn in ev_fqns or not ev_fqns):
                        resolved_in_evidence = True

            if resolved_in_evidence:
                grounded.append(sym)
                continue

            # (b) File path / URL / route string present in evidence or metadata
            if s_base in ev_files or s_lower in [f.lower() for f in ev_files]:
                grounded.append(sym)
                continue
            if s_base in ev_routes or s_lower in [r.lower() for r in ev_routes]:
                grounded.append(sym)
                continue
            if s_base.startswith("/") or s_base.startswith(("GET ", "POST ", "PUT ", "PATCH ", "DELETE ", "USE ")):
                if s_base in ev_raw or any(s_base in r for r in ev_routes):
                    grounded.append(sym)
                    continue

            # (c) Appears as a token in the evidence code
            if s_base in ev_tokens or s_lower in [t.lower() for t in ev_tokens]:
                grounded.append(sym)
                continue
            if len(s_base) >= 2:
                if re.match(r'^[A-Za-z0-9_]+$', s_base):
                    if re.search(r'\b' + re.escape(s_base) + r'\b', ev_raw, re.IGNORECASE):
                        grounded.append(sym)
                        continue
                elif s_base in ev_raw or s_lower in ev_raw.lower():
                    grounded.append(sym)
                    continue

            # Check if file path matches ANY indexed file in repo (not only retrieved ones)
            # If indexed in repo but not retrieved -> exists_not_retrieved, never INVENTED
            if (s_base in self.repo_files) or (s_lower in [rf.lower() for rf in self.repo_files]):
                exists_not_retrieved.append(sym)
                continue

            # Not grounded in evidence. Check if it exists elsewhere in repo symbol table
            repo_res = None
            for f in context_files:
                repo_res = self._resolve_symbol(s_base, f)
                if repo_res:
                    break
            if not repo_res:
                repo_res = self._resolve_symbol(s_base)

            if repo_res or (s_base in self.valid_fqns) or (s_base in self.bare_to_fqn):
                exists_not_retrieved.append(sym)
                continue

            if "." in s_base:
                bare_sub = s_base.split(".")[-1]
                if bare_sub in self.bare_to_fqn:
                    exists_not_retrieved.append(sym)
                    continue

            # Neither grounded nor in repo nor allowlisted:
            # INVENTED must only apply to code-shaped tokens: camelCase, snake_case, dotted
            # names, hyphenated or scoped package names, or paths/names with a file extension.
            # Plain lowercase English words (fields, title, description, claims, entities,
            # async/await, Scripting) are ignored, never INVENTED.
            if _is_code_shaped(s_base):
                invented.append(sym)
            else:
                # Plain words ignored, never INVENTED
                pass

        total_checked = len(grounded) + len(exists_not_retrieved) + len(invented)
        score = (len(grounded) / total_checked) if total_checked > 0 else 1.0
        passed = (len(invented) == 0 and len(exists_not_retrieved) == 0)
        invented_ratio = (len(invented) / total_checked) if total_checked > 0 else 0.0

        if passed:
            status = "passed"
        elif len(invented) <= 1 and invented_ratio < 0.30:
            status = "warning"
        else:
            status = "failed"

        missing_symbols = invented + exists_not_retrieved
        ungrounded_prose_flagged = [t for t in ungrounded if t in invented or t in exists_not_retrieved]

        return {
            "score": round(score, 3),
            "passed": passed,
            "status": status,
            "missing_symbols": missing_symbols,
            "grounded_tokens": grounded,
            "exists_not_retrieved": exists_not_retrieved,
            "invented_tokens": invented,
            "ungrounded_prose_identifiers": ungrounded_prose_flagged,
            "total_checked": total_checked
        }

    def _verify_layer2(self, structures: List[Dict[str, Any]], context_files: Set[str]) -> Dict[str, Any]:
        if not structures:
            return {
                "score": None,
                "passed": None,
                "status": "not_checked",
                "violations": []
            }
        
        violations = []
        matched = 0
        for st in structures:
            raw_sym = st.get("symbol", "")
            resolved_fqn = None
            for f in context_files:
                resolved_fqn = self._resolve_symbol(raw_sym, f)
                if resolved_fqn:
                    break
            if not resolved_fqn:
                resolved_fqn = self._resolve_symbol(raw_sym)

            if not resolved_fqn:
                violations.append(f"Unresolvable structure: {raw_sym}")
                continue

            ref = self.symbol_registry.get(resolved_fqn, {})
            # Validate parent
            if st.get("claimed_parent") and ref.get("parent"):
                if st["claimed_parent"] != ref["parent"]:
                    violations.append(f"Class hierarchy mismatch: {raw_sym} expected parent '{ref['parent']}', got '{st['claimed_parent']}'")
                    continue

            # Validate parameters
            if "claimed_params" in st:
                claimed_params = st["claimed_params"]
                if ref.get("symbol_type") == "route":
                    # Layer 2 for route chunks: do not skip entirely.
                    # Accept req/res/next/self/cls, and accept a claimed param if it appears as an
                    # identifier in that chunk's own code (e.g. destructured from req.body).
                    # Flag claimed params that appear nowhere in the chunk's code.
                    chunk_code = ref.get("content", "")
                    code_identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', chunk_code))
                    standard_route_params = {"req", "res", "next", "self", "cls"}
                    unrecognized_params = [
                        p for p in claimed_params
                        if p not in standard_route_params and p not in code_identifiers
                    ]
                    if unrecognized_params:
                        violations.append(
                            f"Signature mismatch on route {raw_sym}: params {unrecognized_params} appear nowhere in route code"
                        )
                        continue
                else:
                    # Keep checking functions and methods as before
                    if ref.get("params"):
                        claimed_p = [p for p in claimed_params if p not in ["self", "cls"]]
                        actual_p = [p for p in ref["params"] if p not in ["self", "cls"]]
                        if claimed_p and actual_p and set(claimed_p) != set(actual_p):
                            violations.append(f"Signature mismatch on {raw_sym}: claimed {claimed_p}, actual {actual_p}")
                            continue

            matched += 1

        score = matched / len(structures)
        passed = (score >= 0.80 and len(violations) == 0)
        status = "passed" if passed else "failed"
        return {
            "score": round(score, 3),
            "passed": passed,
            "status": status,
            "violations": violations
        }

    def _verify_layer3(self, call_edges: List[Dict[str, str]], context_files: Set[str]) -> Dict[str, Any]:
        if not call_edges:
            return {
                "score": None,
                "passed": None,
                "status": "not_checked",
                "violations": [],
                "unverifiable_external": []
            }

        violations = []
        unverifiable_external = []
        valid_calls = 0

        for edge in call_edges:
            caller = edge.get("caller", "")
            callee = edge.get("callee", "")

            # Calls to the library allowlist are ignored (neither violations nor unverifiable_external)
            if _is_callee_allowlisted(callee) or _is_callee_allowlisted(caller):
                continue

            # Resolve caller using context_files first, then repo-wide
            c_fqn = None
            for f in context_files:
                c_fqn = self._resolve_symbol(caller, f)
                if c_fqn:
                    break
            if not c_fqn:
                c_fqn = self._resolve_symbol(caller)

            # Resolve callee using context_files first, then repo-wide
            e_fqn = None
            for f in context_files:
                e_fqn = self._resolve_symbol(callee, f)
                if e_fqn:
                    break
            if not e_fqn:
                e_fqn = self._resolve_symbol(callee)

            # Check if caller or callee is external/unindexed library code
            if not c_fqn or not e_fqn:
                unverifiable_external.append({
                    "caller": caller,
                    "callee": callee,
                    "resolved_caller": c_fqn,
                    "resolved_callee": e_fqn
                })
                continue

            # Both ends resolve to indexed symbols: strict direct edge check
            if self.call_graph.has_edge(c_fqn, e_fqn):
                valid_calls += 1
            else:
                violations.append(f"Direct invocation violation: '{caller}' does not statically invoke '{callee}'")

        # Total internal verified edges evaluated
        internal_edges_count = valid_calls + len(violations)
        if internal_edges_count > 0:
            if unverifiable_external:
                score = None
                passed = None
                status = "unverifiable_external"
            else:
                score = valid_calls / internal_edges_count
                passed = (score >= 0.80 and len(violations) == 0)
                status = "passed" if passed else "failed"
        elif unverifiable_external:
            score = None
            passed = None
            status = "unverifiable_external"
        else:
            # All calls were ignored (allowlist calls) or no edges
            score = None
            passed = None
            status = "not_checked"

        return {
            "score": round(score, 3) if score is not None else None,
            "passed": passed,
            "status": status,
            "violations": violations,
            "unverifiable_external": unverifiable_external
        }

    def _verify_layer4(self, assertions: List[str], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not assertions:
            return {
                "score": None,
                "passed": None,
                "status": "not_checked",
                "unsupported": []
            }
        
        evidence_text = " ".join([
            (c.get("point").payload.get("content", "") if hasattr(c.get("point"), "payload") else c.get("content", ""))
            for c in chunks if c
        ]).lower()

        if not evidence_text:
            return {
                "score": 0.0,
                "passed": False,
                "status": "failed",
                "unsupported": [{"claim": a, "token_coverage": 0.0} for a in assertions]
            }

        stopwords = {
            "the", "is", "at", "which", "on", "and", "a", "this", "function", "method",
            "returns", "calls", "for", "asserts", "that", "in", "to", "with", "from"
        }
        unsupported = []
        valid = 0

        for a in assertions:
            clean_a = re.sub(r'^[#\*\-\s]+', '', a).strip()
            toks = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', clean_a.lower())) - stopwords
            if not toks:
                valid += 1
                continue
            found = sum(1 for t in toks if t in evidence_text)
            coverage = found / len(toks)
            if coverage >= 0.35:
                valid += 1
            else:
                unsupported.append({"claim": clean_a, "token_coverage": round(coverage, 2)})

        score = valid / len(assertions)
        passed = (score >= 0.70 and len(unsupported) == 0)
        status = "passed" if passed else "failed"
        return {
            "score": round(score, 3),
            "passed": passed,
            "status": status,
            "unsupported": unsupported
        }
