import os, sys, json, asyncio
sys.path.insert(0, os.getcwd())

from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.reranker import rerank_results
from backend.services.ollama_service import generate_code_answer
from backend.services.verification import VerificationEngine

async def run_prose_fpr_benchmark():
    gold = json.load(open('eval/gold_v2.json', encoding='utf-8'))
    queries = gold['answerable'][:20]
    repo_id = 'repo_3e9c5a1753fa'

    # Load engine with chunks from chunks_dump_v2
    dump_data = json.load(open('eval/chunks_dump_v2.json', encoding='utf-8'))
    chunks = [x['payload'] for x in dump_data[repo_id]]
    engine = VerificationEngine(chunks=chunks, repo_id=repo_id)

    results = []
    print("=" * 80)
    print(f" MEASURING PROSE-IDENTIFIER FALSE POSITIVE RATE (20 REAL OLLAMA OUTPUTS)")
    print("=" * 80)

    for i, item in enumerate(queries, 1):
        q = item['q']
        print(f"\n[{i}/20] Query: {q}")
        
        # 1. Retrieve & Rerank
        cands = search_isolated_query(repo_id, query_embedding(q), limit=20)
        reranked = rerank_results(q, cands)[:5]

        # 2. Generate with Ollama
        llm_output = await generate_code_answer(q, reranked)
        explanation = llm_output.get("explanation", "")
        claims = llm_output.get("claims", {})

        # 3. Run verification with answer_text
        trust_report = engine.verify(
            claims=claims,
            retrieved_chunks=reranked,
            ollama_failed=False,
            answer_text=explanation
        )

        l1 = trust_report["layers"].get("layer1_existence", {})
        ungrounded = l1.get("ungrounded_prose_identifiers", [])

        # Check evidence text to inspect flagged items
        context_text = " ".join([
            (c.get("point").payload.get("content", "") + " " +
             c.get("point").payload.get("file_path", "") + " " +
             c.get("point").payload.get("symbol", "") + " " +
             c.get("point").payload.get("fqn", ""))
            if hasattr(c.get("point"), "payload")
            else (c.get("content", "") + " " + c.get("file_path", "") + " " + c.get("symbol", "") + " " + c.get("fqn", ""))
            for c in reranked if c
        ])

        results.append({
            "idx": i,
            "query": q,
            "explanation": explanation[:300] + "..." if len(explanation) > 300 else explanation,
            "claims": claims,
            "decision": trust_report["decision"],
            "trust_score": trust_report["trust_score"],
            "flagged_prose_identifiers": ungrounded
        })

        print(f"    Decision: {trust_report['decision']} (Score: {trust_report['trust_score']})")
        if ungrounded:
            print(f"    Flagged prose identifiers: {ungrounded}")
        else:
            print("    Flagged prose identifiers: None (0)")

    with open('eval/results/prose_fpr_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    all_flagged = [sym for r in results for sym in r["flagged_prose_identifiers"]]
    queries_with_flags = sum(1 for r in results if r["flagged_prose_identifiers"])

    print("\n" + "=" * 80)
    print(" SUMMARY OF PROSE IDENTIFIER FALSE-POSITIVE AUDIT")
    print("=" * 80)
    print(f"Total evaluated answerable queries : {len(results)}")
    print(f"Queries with flagged prose symbols : {queries_with_flags} / {len(results)} ({queries_with_flags/len(results)*100:.1f}%)")
    print(f"Total flagged identifiers          : {len(all_flagged)}")
    print(f"List of flagged identifiers        : {sorted(list(set(all_flagged)))}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_prose_fpr_benchmark())
