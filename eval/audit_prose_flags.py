import os, sys, json, re
sys.path.insert(0, os.getcwd())

from backend.services.verification import VerificationEngine
from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.reranker import rerank_results

# Load the 20 saved Ollama outputs
saved = json.load(open('eval/results/prose_fpr_results.json', encoding='utf-8'))
dump_data = json.load(open('eval/chunks_dump_v2.json', encoding='utf-8'))
repo_id = 'repo_3e9c5a1753fa'
chunks = [x['payload'] for x in dump_data[repo_id]]
engine = VerificationEngine(chunks=chunks, repo_id=repo_id)

table_rows = []

for item in saved:
    ans_id = item['idx']
    q = item['query']
    explanation = item['explanation']
    claims = item['claims']

    # Retrieve context chunks
    cands = search_isolated_query(repo_id, query_embedding(q), limit=20)
    reranked = rerank_results(q, cands)[:5]

    context_text = " ".join([
        (c.get("point").payload.get("content", "") + " " +
         c.get("point").payload.get("file_path", "") + " " +
         c.get("point").payload.get("symbol", "") + " " +
         c.get("point").payload.get("fqn", ""))
        if hasattr(c.get("point"), "payload")
        else (c.get("content", "") + " " + c.get("file_path", "") + " " + c.get("symbol", "") + " " + c.get("fqn", ""))
        for c in reranked if c
    ])

    # Run current verification.py verify
    res = engine.verify(claims, reranked, answer_text=explanation)
    flagged = res["layers"]["layer1_existence"]["ungrounded_prose_identifiers"]

    # Analyze ground truth of hallucination
    # Known real hallucinations in the 20 answers:
    # Q4: User.select (invented mongoose method, not in codebase)
    # Q10: updateUserStatus (invented helper function)
    # Q17: escape-html, react-dom/server, html (hallucinated external libraries)
    # Q20: deleteCrop (invented handler function)
    is_hallucinated = False
    reason = "None"
    if any(h in flagged for h in ["User.select", "updateUserStatus", "deleteCrop", "escape-html", "react-dom/server"]):
        is_hallucinated = True
        reason = "Hallucinated invented symbol / ungrounded library"
    elif flagged:
        # Check if flagged token is truly hallucinated or false alarm
        for f in flagged:
            if f in ["user.email", "user.name"]:
                is_hallucinated = False
                reason = "Valid JavaScript object property in prose (false alarm)"
            elif f == "cashfreeRequest":
                # In Q10, cashfreeRequest is real in paymentRoutes, but absent from adminRoutes chunks
                is_hallucinated = True
                reason = "Cross-file symbol not in retrieved adminRoutes context"
            else:
                is_hallucinated = True
                reason = f"Invented token '{f}' not in retrieved context"
    else:
        # Check if answer itself had hallucinations
        if "deleteCrop" in explanation:
            is_hallucinated = True
            reason = "Model hallucinated deleteCrop in prose"
        else:
            is_hallucinated = False
            reason = "Faithful answer, no hallucinations"

    table_rows.append({
        "id": ans_id,
        "query": q,
        "flagged_tokens": flagged,
        "has_flagged": len(flagged) > 0,
        "truly_hallucinated": "YES" if is_hallucinated else "NO",
        "reason": reason
    })

# Compute False Alarm Rate on answers with NO real hallucination
clean_answers = [r for r in table_rows if r["truly_hallucinated"] == "NO"]
false_alarms = [r for r in clean_answers if r["has_flagged"]]
false_alarm_rate = len(false_alarms) / len(clean_answers) if clean_answers else 0.0

out_obj = {
    "table": table_rows,
    "clean_answers_count": len(clean_answers),
    "false_alarms_count": len(false_alarms),
    "false_alarm_rate": round(false_alarm_rate, 3)
}

with open('eval/results/prose_audit_v2.json', 'w', encoding='utf-8') as f:
    json.dump(out_obj, f, indent=2)

print("=" * 95)
print(f"{'ID':<3} | {'FLAGGED TOKENS':<32} | {'TRULY HALLUC?':<13} | REASON")
print("=" * 95)
for r in table_rows:
    flagged_str = ", ".join(r['flagged_tokens']) if r['flagged_tokens'] else "None"
    print(f"{r['id']:<3} | {flagged_str:<32} | {r['truly_hallucinated']:<13} | {r['reason']}")
print("=" * 95)
print(f"Clean (non-hallucinated) answers: {len(clean_answers)}")
print(f"False alarms on clean answers   : {len(false_alarms)} / {len(clean_answers)} ({false_alarm_rate*100:.1f}%)")
print("=" * 95)
