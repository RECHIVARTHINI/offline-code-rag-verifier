import os, sys, json, math, re, collections
sys.path.insert(0, os.getcwd())

from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.reranker import rerank_results

# Load gold_v2
gold = json.load(open('eval/gold_v2.json', encoding='utf-8'))
repo_id = 'repo_3e9c5a1753fa'

# Load chunks for BM25
dump_data = json.load(open('eval/chunks_dump_v2.json', encoding='utf-8'))
chunks = [x['payload'] for x in dump_data[repo_id]]

def tok(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())

def fqn_of(payload):
    return payload.get("fqn") or f"{payload.get('file_path', '').replace(chr(92), '/')}::{payload.get('symbol')}"

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [collections.Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(self.len)
        df = collections.Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                if t in d:
                    f = d[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out

docs_bm25 = [tok(p["content"] + f" {p['file_path']} {p['symbol']}") for p in chunks]
bm25 = BM25(docs_bm25)
bm25_fqns = [fqn_of(p) for p in chunks]

def metrics_for_ranked(ranked_fqns, relevant):
    k = 5
    top5 = ranked_fqns[:k]
    hit5 = [f for f in top5 if f in relevant]
    p5 = len(hit5) / k
    r5 = len(set(hit5)) / len(relevant)
    f1_5 = (2 * p5 * r5 / (p5 + r5)) if (p5 + r5) else 0.0
    hit1 = 1.0 if ranked_fqns and ranked_fqns[0] in relevant else 0.0
    hit5_val = 1.0 if hit5 else 0.0
    mrr = next((1.0 / (idx + 1) for idx, f in enumerate(ranked_fqns) if f in relevant), 0.0)
    return p5, r5, f1_5, hit1, hit5_val, mrr

# Evaluation loop
variants = {"A": [], "B": [], "C": [], "D": []}
queries_lost_by_B = []

for item in gold['answerable']:
    q = item['q']
    rel = set(item['relevant'])

    # Dense retrieval (Top 20)
    cands = search_isolated_query(repo_id, query_embedding(q), limit=20)
    dense_fqns = [fqn_of(p.payload) for p in cands]

    # Variant A: dense only
    ranked_A = dense_fqns

    # Variant B: dense + cross-encoder
    rr = rerank_results(q, cands)
    ranked_B = [fqn_of(r["point"].payload) for r in rr]

    # Variant C: RRF of dense rank and cross-encoder rank (k=60)
    rrf_c = collections.defaultdict(float)
    for r_dense, fqn in enumerate(dense_fqns, start=1):
        rrf_c[fqn] += 1.0 / (60.0 + r_dense)
    for r_ce, fqn in enumerate(ranked_B, start=1):
        rrf_c[fqn] += 1.0 / (60.0 + r_ce)
    ranked_C = sorted(rrf_c.keys(), key=lambda f: -rrf_c[fqn])
    ranked_C = sorted(dense_fqns, key=lambda f: -rrf_c[f])

    # Variant D: Dense + BM25 RRF (k=60)
    bm_scores = bm25.scores(tok(q))
    bm_order = sorted(range(len(bm_scores)), key=lambda i: -bm_scores[i])
    bm_ranked_fqns = [bm25_fqns[i] for i in bm_order]
    
    rrf_d = collections.defaultdict(float)
    for r_dense, fqn in enumerate(dense_fqns, start=1):
        rrf_d[fqn] += 1.0 / (60.0 + r_dense)
    for r_bm, fqn in enumerate(bm_ranked_fqns, start=1):
        rrf_d[fqn] += 1.0 / (60.0 + r_bm)
    all_seen = list(dict.fromkeys(dense_fqns + bm_ranked_fqns))
    ranked_D = sorted(all_seen, key=lambda f: -rrf_d[f])

    for code, r_list in [("A", ranked_A), ("B", ranked_B), ("C", ranked_C), ("D", ranked_D)]:
        p5, r5, f1_5, h1, h5, mrr = metrics_for_ranked(r_list, rel)
        variants[code].append({
            "q": q, "p5": p5, "r5": r5, "f1_5": f1_5, "h1": h1, "h5": h5, "mrr": mrr,
            "top5": r_list[:5]
        })

    # Check if B lost relevant chunk that A had in top 5
    hit5_A = any(f in rel for f in ranked_A[:5])
    hit5_B = any(f in rel for f in ranked_B[:5])
    if hit5_A and not hit5_B:
        queries_lost_by_B.append({
            "q": q,
            "relevant": list(rel),
            "top5_A": ranked_A[:5],
            "top5_B": ranked_B[:5]
        })

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

summary_table = {}
for code in ["A", "B", "C", "D"]:
    rows = variants[code]
    summary_table[code] = {
        "P@5": round(mean([r["p5"] for r in rows]), 3),
        "R@5": round(mean([r["r5"] for r in rows]), 3),
        "F1@5": round(mean([r["f1_5"] for r in rows]), 3),
        "Hit@1": round(mean([r["h1"] for r in rows]), 3),
        "Hit@5": round(mean([r["h5"] for r in rows]), 3),
        "MRR": round(mean([r["mrr"] for r in rows]), 3)
    }

output_data = {
    "summary": summary_table,
    "queries_lost_by_B": queries_lost_by_B,
    "variants": variants
}

with open('eval/results/reranker_variants.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)

print("=" * 80)
print(f"{'VARIANT':<32} | {'P@5':<6} | {'R@5':<6} | {'F1@5':<6} | {'Hit@1':<6} | {'Hit@5':<6} | {'MRR':<6}")
print("=" * 80)
labels = {
    "A": "(A) dense only",
    "B": "(B) dense + cross-encoder",
    "C": "(C) reciprocal rank fusion (dense + CE)",
    "D": "(D) dense + BM25 RRF"
}
for code in ["A", "B", "C", "D"]:
    s = summary_table[code]
    print(f"{labels[code]:<32} | {s['P@5']:<6.3f} | {s['R@5']:<6.3f} | {s['F1@5']:<6.3f} | {s['Hit@1']:<6.3f} | {s['Hit@5']:<6.3f} | {s['MRR']:<6.3f}")

print("=" * 80)
print(f"Queries where (B) loses relevant chunk that (A) had in top 5: {len(queries_lost_by_B)}")
for item in queries_lost_by_B:
    print(f"\n* Query: {item['q']}")
    print(f"  Relevant : {item['relevant']}")
    print(f"  Top-5 (A): {item['top5_A']}")
    print(f"  Top-5 (B): {item['top5_B']}")
