import os, sys, json
sys.path.insert(0, os.getcwd())

from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.reranker import rerank_results

gold = json.load(open('eval/gold_v2.json', encoding='utf-8'))
repo_id = 'repo_3e9c5a1753fa'

def get_top_score(q):
    cands = search_isolated_query(repo_id, query_embedding(q), limit=20)
    rr = rerank_results(q, cands)
    return rr[0]["score"] if rr else -999.0

answerable_scores = []
for item in gold['answerable']:
    q = item['q']
    score = get_top_score(q)
    answerable_scores.append({"q": q, "score": round(float(score), 3), "type": "answerable"})

unanswerable_scores = []
for item in gold['unanswerable']:
    q = item['q']
    score = get_top_score(q)
    unanswerable_scores.append({"q": q, "score": round(float(score), 3), "type": "unanswerable"})

all_results = {
    "answerable": answerable_scores,
    "unanswerable": unanswerable_scores
}

with open('eval/results/rerank_scores.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)

print("=" * 80)
print(f"{'TYPE':<14} | {'TOP SCORE':<10} | QUERY")
print("=" * 80)
for r in answerable_scores:
    print(f"{r['type']:<14} | {r['score']:<10.3f} | {r['q']}")
print("-" * 80)
for r in unanswerable_scores:
    print(f"{r['type']:<14} | {r['score']:<10.3f} | {r['q']}")
print("=" * 80)
