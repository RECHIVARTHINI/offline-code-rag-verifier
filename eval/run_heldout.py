import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import get_or_restore_engine
from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.ollama_service import generate_code_answer

repo_id = "repo_3e9c5a1753fa"
engine = get_or_restore_engine(repo_id)

heldout_path = "eval/heldout_queries.json"
out_path = "eval/results/heldout.json"

with open(heldout_path, "r", encoding="utf-8") as f:
    heldout_data = json.load(f)

answerable_queries = heldout_data.get("answerable", [])
bait_queries = heldout_data.get("bait", [])

async def run_queries(query_list, group_name):
    group_results = []
    print(f"\n--- Running {len(query_list)} {group_name} queries ---")
    for i, q in enumerate(query_list, 1):
        # 1. Dense top-5
        cands = search_isolated_query(repo_id, query_embedding(q), limit=5)
        reranked = [{"point": c, "score": c.score} for c in cands]
        
        # 2. Ollama generation
        llm_out = await generate_code_answer(q, reranked)
        answer = llm_out.get("explanation", "")
        claims = llm_out.get("claims", {})
        ollama_failed = "error" in llm_out or not answer
        
        # 3. Verification
        rep = engine.verify(
            claims=claims,
            retrieved_chunks=reranked,
            ollama_failed=ollama_failed,
            answer_text=answer
        )
        
        invented = rep.get("layers", {}).get("layer1_existence", {}).get("invented_tokens", [])
        dec = rep["decision"]
        sc = rep["trust_score"]
        
        entry = {
            "group": group_name,
            "idx": i,
            "query": q,
            "answer": answer,
            "claims": claims,
            "decision": dec,
            "trust_score": sc,
            "invented_tokens": invented,
            "per_layer": rep.get("layers", {})
        }
        group_results.append(entry)
        print(f"[{group_name} {i:02d}] {dec:<8} | Invented: {invented} | {q}")
    return group_results

async def main():
    ans_results = await run_queries(answerable_queries, "answerable")
    bait_results = await run_queries(bait_queries, "bait")
    
    all_results = {
        "answerable": ans_results,
        "bait": bait_results
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
