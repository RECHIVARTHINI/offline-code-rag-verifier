from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_javascript
import tree_sitter_python

JS_language = Language(tree_sitter_javascript.language())
PYTHON_language = Language(tree_sitter_python.language())

js_parser = Parser(JS_language)
py_parser = Parser(PYTHON_language)


def _node_text(source_bytes, node):
    if not node:
        return None
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _make_chunk(document, node, symbol, symbol_type, parent, language, **extra):
    chunk = {
        "content": _node_text(document["content"].encode("utf-8"), node),
        "file_path": document["path"],
        "file_name": Path(document["path"]).name,
        "language": language,
        "symbol": symbol,
        "symbol_type": symbol_type,
        "parent": parent,
        "start_line": node.start_point.row + 1,
        "end_line": node.end_point.row + 1,
    }
    chunk.update(extra)
    return chunk


def _js_object_name(node):
    if not node:
        return None
    for child in node.children:
        if child.type in {"identifier", "property_identifier", "private_identifier"}:
            return _node_text(node.parent if False else b"", child) if False else None
    return None


def _extract_js_route(node, source_bytes):
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

    if not method_name:
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


def js_chunker(document):
    source_bytes = document["content"].encode("utf-8")
    tree = js_parser.parse(source_bytes)
    chunks = []

    def visit(node, parent=None, current_class=None):
        if node.type == "call_expression":
            route = _extract_js_route(node, source_bytes)
            if route:
                chunks.append({
                    "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                    "file_path": document["path"],
                    "file_name": Path(document["path"]).name,
                    "language": "javascript",
                    "symbol": f"{route['method']} {route['path']}",
                    "symbol_type": "route",
                    "route_method": route["method"],
                    "route_path": route["path"],
                    "parent": route["parent"],
                    "start_line": node.start_point.row + 1,
                    "end_line": node.end_point.row + 1,
                })
                return

        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            function_name = _node_text(source_bytes, name_node) or "anonymous"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "javascript",
                "symbol": function_name,
                "symbol_type": "function",
                "parent": parent,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_parent = function_name

        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            function_name = _node_text(source_bytes, name_node) or "anonymous"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "javascript",
                "symbol": function_name,
                "symbol_type": "method",
                "parent": current_class or parent,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_parent = function_name

        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source_bytes, name_node) or "anonymous"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "javascript",
                "symbol": class_name,
                "symbol_type": "class",
                "parent": parent,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_class = class_name
            current_parent = class_name

        else:
            current_parent = parent
            current_class = current_class

        for child in node.children:
            visit(child, current_parent, current_class)

    visit(tree.root_node, None, None)
    return chunks


def py_chunker(document):
    source_bytes = document["content"].encode("utf-8")
    tree = py_parser.parse(source_bytes)
    chunks = []

    def visit(node, parent=None, current_class=None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source_bytes, name_node) or "anonymous"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "python",
                "symbol": class_name,
                "symbol_type": "class",
                "parent": parent,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_class = class_name
            current_parent = class_name

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            function_name = _node_text(source_bytes, name_node) or "anonymous"
            symbol_type = "method" if current_class is not None else "function"
            chunks.append({
                "content": source_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                "file_path": document["path"],
                "file_name": Path(document["path"]).name,
                "language": "python",
                "symbol": function_name,
                "symbol_type": symbol_type,
                "parent": current_class or parent,
                "start_line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
            })
            current_parent = function_name

        else:
            current_parent = parent
            current_class = current_class

        for child in node.children:
            visit(child, current_parent, current_class)

    visit(tree.root_node, None, None)
    return chunks


def chunker(document):
    extension = document["extension"].lower()

    if extension in {".js", ".jsx"}:
        return js_chunker(document)

    if extension == ".py":
        return py_chunker(document)

    return []