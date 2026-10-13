import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import get_or_restore_engine
from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query

repo_id = "repo_3e9c5a1753fa"
engine = get_or_restore_engine(repo_id)

answers_raw = json.load(open("eval/results/answers_raw.json", encoding="utf-8"))
target_ids = [4, 17, 20]

print("=" * 80)
print("FROZEN ANSWERS VERIFICATION (IDs 4, 17, 20)")
print("=" * 80)

for item in answers_raw:
    idx = item.get("idx")
    if idx not in target_ids:
        continue
    
    q = item.get("query", "")
    ans = item.get("explanation", "")
    claims = item.get("claims", {})
    
    # Retrieve top 5
    cands = search_isolated_query(repo_id, query_embedding(q), limit=5)
    reranked = [{"point": c, "score": c.score} for c in cands]
    
    rep = engine.verify(
        claims=claims,
        retrieved_chunks=reranked,
        ollama_failed=False,
        answer_text=ans
    )
    
    dec = rep["decision"]
    score = rep["trust_score"]
    l1 = rep["layers"]["layer1_existence"]
    invented = l1.get("invented_tokens", [])
    exists_nr = l1.get("exists_not_retrieved", [])
    ungrounded_prose = l1.get("ungrounded_prose_identifiers", [])
    
    print(f"\n[Answer {idx}] Query: {q}")
    print(f"Decision: {dec} | Trust Score: {score}")
    print(f"Invented Tokens: {invented}")
    print(f"Exists Not Retrieved: {exists_nr}")
    print(f"Ungrounded Prose Identifiers: {ungrounded_prose}")
    
    if idx == 4:
        has_user_select = "User.select" in invented or "User.select" in exists_nr
        print(f"-> 'User.select' flagged: {has_user_select}")
    elif idx == 17:
        has_escape_html = any("escape-html" in t for t in (invented + exists_nr + ungrounded_prose))
        has_react_dom = any("react-dom" in t for t in (invented + exists_nr + ungrounded_prose))
        print(f"-> 'escape-html' flagged: {has_escape_html}")
        print(f"-> 'react-dom/server' flagged: {has_react_dom}")
    elif idx == 20:
        has_delete_crop = "deleteCrop" in invented or "deleteCrop" in exists_nr
        print(f"-> 'deleteCrop' flagged: {has_delete_crop}")
    
    assert dec != "VERIFIED", f"Answer {idx} should NOT be VERIFIED, got {dec}"

print("\n" + "=" * 80)
print("CONFIRMATION: None of the 3 frozen answers is VERIFIED!")
print("=" * 80)
