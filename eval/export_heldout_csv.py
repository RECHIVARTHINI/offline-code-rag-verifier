import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query

repo_id = "repo_3e9c5a1753fa"
heldout_results = json.load(open("eval/results/heldout.json", encoding="utf-8"))

rows = []
for group_name in ["answerable", "bait"]:
    for entry in heldout_results.get(group_name, []):
        q = entry.get("query", "")
        ans = entry.get("answer", "")
        dec = entry.get("decision", "")
        inv = entry.get("invented_tokens", [])
        inv_str = "; ".join(inv) if isinstance(inv, list) else str(inv)

        # Retrieve top 5 FQNs
        cands = search_isolated_query(repo_id, query_embedding(q), limit=5)
        fqns = [c.payload.get("fqn", "") for c in cands]
        fqns_str = "; ".join(fqns)

        rows.append({
            "group": group_name,
            "query": q,
            "answer_text": ans,
            "retrieved_fqns": fqns_str,
            "decision": dec,
            "invented_tokens": inv_str,
            "human_label": "",
            "human_notes": ""
        })

out_csv = "eval/heldout_labeling.csv"
fieldnames = [
    "group",
    "query",
    "answer_text",
    "retrieved_fqns",
    "decision",
    "invented_tokens",
    "human_label",
    "human_notes"
]

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(len(rows))
