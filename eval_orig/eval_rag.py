"""
Retrieval evaluation for the Code RAG (precision / recall / F1 / accuracy).

Run from the PROJECT ROOT (so storage/qdrant resolves) with your venv active:

  # real pipeline: bge-small -> Qdrant top-20 -> cross-encoder -> top-5
  python eval_rag.py --gold gold_cropchain.json --repo-id repo_3e9c5a1753fa --backend dense

  # lexical baselines (no models needed), chunks read from a JSON dump
  python eval_rag.py --gold gold_cropchain.json --backend bm25 --chunks-json chunks_dump.json

Metrics (per query, then macro-averaged over answerable queries):
  P@k   = relevant in top-k / k
  R@k   = relevant in top-k / total relevant
  F1@k  = harmonic mean of the two
  Hit@k = 1 if any relevant chunk is in top-k   (this is the "accuracy" of retrieval)
  MRR   = 1 / rank of first relevant chunk
Unanswerable queries: abstention accuracy = fraction where the system's best score is
below --abstain-threshold (needs a score; dense backend uses the reranker score).
NOTE: P@k is capped at |relevant|/k for queries with few relevant chunks -- compare it with
the "P@k ceiling" line, not with 1.0.
"""
import argparse, json, math, os, re, sys, collections


def tok(text, with_meta=None):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)          # camelCase -> camel Case
    return re.findall(r"[a-z0-9]+", text.lower())


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


def fqn_of(payload):
    return payload.get("fqn") or f"{payload.get('file_path', '').replace(chr(92), '/')}::{payload.get('symbol')}"


def build_backend(args):
    if args.backend in ("bm25", "bm25-meta"):
        raw = json.load(open(args.chunks_json))
        payloads = [x["payload"] for x in (raw["repo_3e9c5a1753fa"] if isinstance(raw, dict) else raw)]
        use_meta = args.backend == "bm25-meta"
        docs = [tok(p["content"] + (f" {p['file_path']} {p['symbol']}" if use_meta else "")) for p in payloads]
        bm = BM25(docs)

        def search(q, k):
            sc = bm.scores(tok(q))
            order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
            return [(fqn_of(payloads[i]), sc[i]) for i in order]
        return search

    # ---- real pipeline (imports the project's own services) ----
    sys.path.insert(0, os.getcwd())
    from backend.services.embedding import query_embedding
    from backend.services.storage import search_isolated_query
    from backend.services.reranker import rerank_results

    def search(q, k):
        cands = search_isolated_query(args.repo_id, query_embedding(q), limit=args.candidates)
        if args.stage == "retrieve":
            return [(fqn_of(p.payload), p.score) for p in cands][:k]
        rr = rerank_results(q, cands)
        return [(fqn_of(r["point"].payload), r["score"]) for r in rr][:k]
    return search


def metrics_at(ranked, relevant, k):
    top = [f for f, _ in ranked[:k]]
    hit = [f for f in top if f in relevant]
    p = len(hit) / k
    r = len(set(hit)) / len(relevant)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, 1.0 if hit else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--backend", choices=["dense", "bm25", "bm25-meta"], default="dense")
    ap.add_argument("--repo-id")
    ap.add_argument("--chunks-json")
    ap.add_argument("--stage", choices=["retrieve", "rerank"], default="rerank",
                    help="dense only: evaluate Qdrant top-k alone or after the cross-encoder")
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--abstain-threshold", type=float, default=None,
                    help="score below which the system should say 'no evidence' (unanswerable queries)")
    ap.add_argument("--out")
    args = ap.parse_args()

    gold = json.load(open(args.gold))
    search = build_backend(args)
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(max(ks), 10)
    per_query, agg = [], {k: collections.defaultdict(list) for k in ks}
    mrr, ceil = [], {k: [] for k in ks}
    for item in gold["answerable"]:
        rel = set(item["relevant"])
        ranked = search(item["q"], kmax)
        rr = next((1 / (i + 1) for i, (f, _) in enumerate(ranked) if f in rel), 0.0)
        mrr.append(rr)
        row = {"q": item["q"], "n_relevant": len(rel), "mrr": rr, "top5": [f for f, _ in ranked[:5]]}
        for k in ks:
            p, r, f1, h = metrics_at(ranked, rel, k)
            for name, v in zip(("P", "R", "F1", "Hit"), (p, r, f1, h)):
                agg[k][name].append(v)
            ceil[k].append(min(len(rel), k) / k)
            row[f"P@{k}"], row[f"R@{k}"], row[f"F1@{k}"], row[f"Hit@{k}"] = p, r, f1, h
        per_query.append(row)

    print(f"\nbackend={args.backend}  answerable queries={len(per_query)}")
    print(f"{'k':>3} {'P@k':>7} {'P@k ceiling':>12} {'R@k':>7} {'F1@k':>7} {'Hit@k(acc)':>11}")
    mean = lambda xs: sum(xs) / len(xs)
    summary = {"backend": args.backend, "n_answerable": len(per_query), "MRR": round(mean(mrr), 3), "by_k": {}}
    for k in ks:
        s = {n: round(mean(v), 3) for n, v in agg[k].items()}
        s["P_ceiling"] = round(mean(ceil[k]), 3)
        summary["by_k"][k] = s
        print(f"{k:>3} {s['P']:>7.3f} {s['P_ceiling']:>12.3f} {s['R']:>7.3f} {s['F1']:>7.3f} {s['Hit']:>11.3f}")
    print(f"MRR = {summary['MRR']:.3f}")

    if gold.get("unanswerable"):
        scores = [search(u["q"], 1)[0][1] for u in gold["unanswerable"]]
        summary["unanswerable_top_scores"] = [round(float(s), 3) for s in scores]
        print("\nUnanswerable queries: top-1 score (system never returns 'no evidence' unless a threshold is applied)")
        for u, s in zip(gold["unanswerable"], scores):
            print(f"  {s:8.3f}  {u['q']}")
        if args.abstain_threshold is not None:
            acc = mean([1.0 if s < args.abstain_threshold else 0.0 for s in scores])
            fa = mean([1.0 if search(i['q'], 1)[0][1] < args.abstain_threshold else 0.0 for i in gold['answerable']])
            print(f"abstain@{args.abstain_threshold}: correct abstentions {acc:.2f}, wrongful abstentions on answerable {fa:.2f}")
    worst = sorted(per_query, key=lambda r: (r["Hit@5"], r["mrr"]))[:5]
    print("\nWorst queries:")
    for r in worst:
        print(f"  Hit@5={r['Hit@5']:.0f} MRR={r['mrr']:.2f}  {r['q']}\n      got: {r['top5'][:3]}")
    if args.out:
        json.dump({"summary": summary, "per_query": per_query}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
