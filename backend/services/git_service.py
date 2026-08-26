from git import Repo
import tempfile
from pathlib import Path
def clone_repo(repo_url:str):
    temp_dir=Path(tempfile.mkdtemp(prefix="codebase-rag"))
    Repo.clone_from(repo_url,temp_dir)
    return temp_dir