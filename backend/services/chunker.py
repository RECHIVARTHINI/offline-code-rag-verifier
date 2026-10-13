import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from tree_sitter import Language, Parser
import tree_sitter_javascript
import tree_sitter_python

JS_language = Language(tree_sitter_javascript.language())
PYTHON_language = Language(tree_sitter_python.language())

js_parser = Parser(JS_language)
py_parser = Parser(PYTHON_language)


def _node_text(source_bytes: bytes, node) -> Optional[str]:
    if not node:
        return None
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _clean_file_path(path_str: str) -> str:
    return path_str.replace("\\", "/").strip("/")


def _extract_py_parameters(func_node, source_bytes: bytes) -> List[str]:
    """Extracts formal parameter names from a Python function_definition node."""
    params_node = func_node.child_by_field_name("parameters")
    if not params_node:
        return []
    
    params = []
    for child in params_node.children:
        if child.type == "identifier":
            param_name = _node_text(source_bytes, child)
            if param_name:
                params.append(param_name)
        elif child.type in {"typed_parameter", "default_parameter", "typed_default_parameter"}:
            name_node = child.child_by_field_name("name")
            if not name_node:
                for sub in child.children:
                    if sub.type == "identifier":
                        name_node = sub
                        break
            param_name = _node_text(source_bytes, name_node) if name_node else None
            if param_name:
                params.append(param_name)
        elif child.type in {"list_splat_pattern", "dictionary_splat_pattern"}:
            name_node = None
            for sub in child.children:
                if sub.type == "identifier":
                    name_node = sub
                    break
            param_name = _node_text(source_bytes, name_node) if name_node else None
            if param_name:
                params.append(param_name)
    return params


def _extract_js_parameters(func_node, source_bytes: bytes) -> List[str]:
    """Extracts parameter names from a JavaScript function/method/arrow node."""
    params_node = func_node.child_by_field_name("parameters")
    if not params_node:
        for child in func_node.children:
            if child.type == "formal_parameters":
                params_node = child
                break
    if not params_node:
        # Check single identifier parameter in arrow functions: x => x * 2
        param_n = func_node.child_by_field_name("parameter")
        if param_n and param_n.type == "identifier":
            p = _node_text(source_bytes, param_n)
            return [p] if p else []
        for child in func_node.children:
            if child.type == "identifier":
                p = _node_text(source_bytes, child)
                return [p] if p else []
            if child.type == "=>":
                break
        return []
    
    params = []
    for child in params_node.children:
        if child.type == "identifier":
            p = _node_text(source_bytes, child)
            if p:
                params.append(p)
        elif child.type in {"formal_parameter", "required_parameter", "optional_parameter"}:
            name_node = child.child_by_field_name("name") or (child.children[0] if child.children else None)
            p = _node_text(source_bytes, name_node) if name_node else None
            if p:
                params.append(p)
    return params


def _extract_router_variables(root_node, source_bytes: bytes) -> set:
    """Finds all variable names assigned from express.Router() or Router() or express()."""
    router_vars = {"router", "app", "server"}
    
    def check_node(n):
        if n.type in {"lexical_declaration", "variable_declaration"}:
            for child in n.children:
                if child.type == "variable_declarator":
                    name_n = child.child_by_field_name("name")
                    val_n = child.child_by_field_name("value")
                    if name_n and val_n:
                        var_name = _node_text(source_bytes, name_n)
                        val_text = _node_text(source_bytes, val_n) or ""
                        if any(pat in val_text for pat in ["express.Router()", "Router()", "express()"]):
                            if var_name:
                                router_vars.add(var_name)
        for c in n.children:
            check_node(c)

    check_node(root_node)
    return router_vars


def _extract_model_names_map(root_node, source_bytes: bytes) -> Dict[str, str]:
    """Finds all mongoose.model('ModelName', schemaVar) calls and maps schemaVar -> ModelName."""
    models_map = {}
    
    def check_node(n):
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            fn_text = _node_text(source_bytes, fn) or ""
            if "model" in fn_text:
                args = n.child_by_field_name("arguments")
                if args:
                    arg_nodes = [c for c in args.children if c.type not in {"(", ")", ","}]
                    if len(arg_nodes) >= 2:
                        raw_model = _node_text(source_bytes, arg_nodes[0]) or ""
                        raw_schema = _node_text(source_bytes, arg_nodes[1]) or ""
                        model_name = raw_model.strip("\"'")
                        schema_var = raw_schema.strip()
                        if model_name and schema_var:
                            models_map[schema_var] = model_name
        for c in n.children:
            check_node(c)

    check_node(root_node)
    return models_map


def _extract_js_route(node, source_bytes: bytes, allowed_receivers: Optional[set] = None) -> Optional[Dict[str, Any]]:
    if node.type != "call_expression":
        return None

    member_expression = None
    for child in node.children:
        if child.type == "member_expression":
            member_expression = child
            break

    if not member_expression:
        return None

    object_name = None
    method_name = None
    for child in member_expression.children:
        if child.type in {"identifier", "this"}:
            if object_name is None:
                object_name = _node_text(source_bytes, child)
        elif child.type in {"property_identifier", "identifier"}:
            if method_name is None:
                method_name = _node_text(source_bytes, child)

    if not method_name or not object_name:
        return None

    # Only accept call receivers named router/app/server or assigned from express.Router()
    valid_receivers = allowed_receivers if allowed_receivers is not None else {"router", "app", "server"}
    if object_name not in valid_receivers:
        return None

    route_method = method_name.lower()
    if route_method not in {"get", "post", "put", "patch", "delete", "use"}:
        return None

    arguments_node = None
    for child in node.children:
        if child.type == "arguments":
            arguments_node = child
            break

    if not arguments_node:
        return None

    route_path = None
    route_callback = None
    for child in arguments_node.children:
        if route_path is None and child.type in {"string", "template_string", "literal"}:
            route_path = _node_text(source_bytes, child)
        elif route_callback is None and child.type in {"arrow_function", "function_expression"}:
            route_callback = child

    if route_path is None:
        return None

    route_path = route_path.strip()
    if route_path.startswith(("'", '"')) and route_path.endswith(("'", '"')):
        route_path = route_path[1:-1]

    return {
        "method": route_method.upper(),
        "path": route_path,
        "parent": object_name or "router",
        "callback": route_callback,
    }


def js_chunker(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_bytes = document["content"].encode("utf-8")
    tree = js_parser.parse(source_bytes)
    chunks = []
    clean_file = _clean_file_path(document["path"])
    file_name = Path(document["path"]).name

    allowed_receivers = _extract_router_variables(tree.root_node, source_bytes)
    models_map = _extract_model_names_map(tree.root_node, source_bytes)
    use_route_lines = []

    def visit(node, parent=None, current_class=None, in_function=False):
        nonlocal use_route_lines

        # 1. Routes (Never create route chunks nested inside function chunks)
        if node.type == "call_expression" and not in_function:
            route = _extract_js_route(node, source_bytes, allowed_receivers)
            if route:
                if route["method"] == "USE":
                    # Collect app.use / router.use for merged "route map" chunk
                    use_line = _node_text(source_bytes, node)
                    if use_line:
                        use_route_lines.append(use_line)
                    return
                else:
                    raw_symbol = f"{route['method']} {route['path']}"
                    fqn = f"{clean_file}::{raw_symbol}"
                    chunks.append({
                        "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                        "file_path": document["path"],
                        "file_name": file_name,
                        "language": "javascript",
                        "symbol": raw_symbol,
                        "fqn": fqn,
                        "symbol_type": "route",
                        "route_method": route["method"],
                        "route_path": route["path"],
                        "parent": route["parent"],
                        "params": ["req", "res"],
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                    })
                    return

        # 2. exports.name = ... or module.exports = ...
        if node.type == "expression_statement":
            for child in node.children:
                if child.type == "assignment_expression":
                    left = child.child_by_field_name("left")
                    right = child.child_by_field_name("right")
                    left_text = _node_text(source_bytes, left) if left else ""
                    if left_text:
                        # Case: exports.name = (async) arrow/function
                        if left_text.startswith("exports."):
                            fn_name = left_text.split(".", 1)[1].strip()
                            if right and right.type in {"arrow_function", "function_expression"}:
                                params = _extract_js_parameters(right, source_bytes)
                                fqn = f"{clean_file}::{fn_name}"
                                chunks.append({
                                    "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                                    "file_path": document["path"],
                                    "file_name": file_name,
                                    "language": "javascript",
                                    "symbol": fn_name,
                                    "fqn": fqn,
                                    "symbol_type": "function",
                                    "parent": parent,
                                    "params": params,
                                    "start_line": node.start_point.row + 1,
                                    "end_line": node.end_point.row + 1,
                                })
                                return
                        # Case: module.exports = ...
                        elif left_text == "module.exports":
                            content_str = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                            # Do not index chunks whose body is only `module.exports = <identifier>` or `module.exports = router`.
                            if re.match(r"^module\.exports\s*=\s*[A-Za-z0-9_$]+;?$", content_str.strip()):
                                return
                            # Also drop duplicate model exports (module.exports = mongoose.model(...))
                            if "mongoose.model" in content_str or (right and right.type == "call_expression" and "model" in (_node_text(source_bytes, right.child_by_field_name("function")) or "")):
                                return
                            params = _extract_js_parameters(right, source_bytes) if (right and right.type in {"arrow_function", "function_expression"}) else []
                            sym_type = "function" if (right and right.type in {"arrow_function", "function_expression"}) else "export"
                            fqn = f"{clean_file}::module.exports"
                            chunks.append({
                                "content": content_str,
                                "file_path": document["path"],
                                "file_name": file_name,
                                "language": "javascript",
                                "symbol": "module.exports",
                                "fqn": fqn,
                                "symbol_type": sym_type,
                                "parent": parent,
                                "params": params,
                                "start_line": node.start_point.row + 1,
                                "end_line": node.end_point.row + 1,
                            })
                            return

        # 3. const name = (async) (...) => {} or = function, and Mongoose Schema
        if node.type in {"lexical_declaration", "variable_declaration"}:
            for child in node.children:
                if child.type == "variable_declarator":
                    name_n = child.child_by_field_name("name")
                    val_n = child.child_by_field_name("value")
                    var_name = _node_text(source_bytes, name_n)
                    if var_name and val_n:
                        # Mongoose Schema: new mongoose.Schema({...})
                        if val_n.type == "new_expression":
                            constructor_n = val_n.child_by_field_name("constructor")
                            c_text = _node_text(source_bytes, constructor_n) or ""
                            if "Schema" in c_text:
                                # Determine model name from models_map, or fallback from var_name
                                model_name = models_map.get(var_name)
                                if not model_name:
                                    if var_name.lower().endswith("schema") and len(var_name) > 6:
                                        model_name = var_name[:-6]
                                        model_name = model_name[0].upper() + model_name[1:]
                                    else:
                                        model_name = var_name
                                fqn = f"{clean_file}::{model_name}"
                                chunks.append({
                                    "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                                    "file_path": document["path"],
                                    "file_name": file_name,
                                    "language": "javascript",
                                    "symbol": model_name,
                                    "fqn": fqn,
                                    "symbol_type": "model",
                                    "parent": parent,
                                    "params": [],
                                    "start_line": node.start_point.row + 1,
                                    "end_line": node.end_point.row + 1,
                                })
                                return

                        # const name = (async) (...) => {} / = function
                        if val_n.type in {"arrow_function", "function_expression"}:
                            params = _extract_js_parameters(val_n, source_bytes)
                            fqn = f"{clean_file}::{var_name}"
                            chunks.append({
                                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                                "file_path": document["path"],
                                "file_name": file_name,
                                "language": "javascript",
                                "symbol": var_name,
                                "fqn": fqn,
                                "symbol_type": "function",
                                "parent": parent,
                                "params": params,
                                "start_line": node.start_point.row + 1,
                                "end_line": node.end_point.row + 1,
                            })
                            return

        # 4. Standard function declaration
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            function_name = _node_text(source_bytes, name_node) or "anonymous"
            fqn = f"{clean_file}::{function_name}"
            params = _extract_js_parameters(node, source_bytes)
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": file_name,
                "language": "javascript",
                "symbol": function_name,
                "fqn": fqn,
                "symbol_type": "function",
                "parent": parent,
                "params": params,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            for child in node.children:
                visit(child, function_name, current_class, in_function=True)
            return

        # 5. Method definition
        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            method_name = _node_text(source_bytes, name_node) or "anonymous"
            owner_class = current_class or parent
            symbol = f"{owner_class}.{method_name}" if owner_class else method_name
            fqn = f"{clean_file}::{symbol}"
            params = _extract_js_parameters(node, source_bytes)
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": file_name,
                "language": "javascript",
                "symbol": symbol,
                "fqn": fqn,
                "symbol_type": "method",
                "parent": owner_class,
                "params": params,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            for child in node.children:
                visit(child, method_name, owner_class, in_function=True)
            return

        # 6. Class declaration
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source_bytes, name_node) or "anonymous"
            fqn = f"{clean_file}::{class_name}"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": file_name,
                "language": "javascript",
                "symbol": class_name,
                "fqn": fqn,
                "symbol_type": "class",
                "parent": parent,
                "params": [],
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            for child in node.children:
                visit(child, class_name, class_name, in_function=in_function)
            return

        # Recurse through children
        is_func_now = in_function or (node.type in {"arrow_function", "function_expression"})
        for child in node.children:
            visit(child, parent, current_class, in_function=is_func_now)

    visit(tree.root_node, None, None, in_function=False)

    # 7. Merge collected app.use / router.use calls into a single "route map" chunk per file
    if use_route_lines:
        content_use = "\n".join(use_route_lines)
        chunks.append({
            "content": content_use,
            "file_path": document["path"],
            "file_name": file_name,
            "language": "javascript",
            "symbol": "route map",
            "fqn": f"{clean_file}::route map",
            "symbol_type": "route_map",
            "parent": None,
            "params": [],
            "start_line": 1,
            "end_line": tree.root_node.end_point.row + 1,
        })

    return chunks


def py_chunker(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_bytes = document["content"].encode("utf-8")
    tree = py_parser.parse(source_bytes)
    chunks = []
    clean_file = _clean_file_path(document["path"])

    def visit(node, parent=None, current_class=None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source_bytes, name_node) or "anonymous"
            fqn = f"{clean_file}::{class_name}"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "python",
                "symbol": class_name,
                "fqn": fqn,
                "symbol_type": "class",
                "parent": parent,
                "params": [],
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_class = class_name
            current_parent = class_name

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            func_name = _node_text(source_bytes, name_node) or "anonymous"
            params = _extract_py_parameters(node, source_bytes)
            
            if current_class is not None:
                symbol = f"{current_class}.{func_name}"
                symbol_type = "method"
                owner = current_class
            else:
                symbol = func_name
                symbol_type = "function"
                owner = parent

            fqn = f"{clean_file}::{symbol}"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "python",
                "symbol": symbol,
                "fqn": fqn,
                "symbol_type": symbol_type,
                "parent": owner,
                "params": params,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_parent = func_name

        else:
            current_parent = parent
            current_class = current_class

        for child in node.children:
            visit(child, current_parent, current_class)

    visit(tree.root_node, None, None)
    return chunks


def chunker(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    extension = document.get("extension", "").lower()

    if extension in {".js", ".jsx"}:
        return js_chunker(document)

    if extension == ".py":
        return py_chunker(document)

    return []
