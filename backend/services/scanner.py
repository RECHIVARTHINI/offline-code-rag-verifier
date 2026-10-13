from pathlib import Path
def scanner(repo_path:Path):
    documents=[]
    allowed_extensions = {".js", ".jsx", ".py"}
    ignored_dir={".git",
    "node_modules",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next"}
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(directory in file_path.parts
               for directory in ignored_dir):
            continue
        if file_path.suffix.lower() not in allowed_extensions:
            continue
        try:
            content=file_path.read_text(encoding="utf-8")
        except(UnicodeDecodeError,OSError):
            continue
        documents.append({
            "path":str(file_path.relative_to(repo_path)),
            "content":content,
            "extension":file_path.suffix.lower()
        })
    return documents