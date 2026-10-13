"""
eval/eval_rag.py: Evaluates RAG retrieval and reranker abstain threshold
across answerable and unanswerable queries in CropChain.
"""
import sys
import os
import json
import argparse
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.storage import get_qdrant_client, search_isolated_query
from backend.services.embedding import query_embedding
from backend.services.reranker import rerank_results

def evaluate_threshold(dataset_path: str, abstain_threshold: float, collection_name: str = "repo_3e9c5a1753fa") -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    unanswerable_total = 0
    unanswerable_abstained = 0
    answerable_total = 0
    answerable_passed = 0

    print("=" * 80)
    print(f" EVALUATING RAG ABSTAIN THRESHOLD: {abstain_threshold:.2f} (Collection: {collection_name})")
    print("=" * 80)

    for item in cases:
        qid = item["id"]
        query = item["query"]
        is_answerable = item["is_answerable"]

        qvec = query_embedding(query)
        candidates = search_isolated_query(collection_name, qvec, limit=20)
        reranked = rerank_results(query, candidates)[:5]
        top_score = reranked[0]["score"] if reranked else -999.0

        abstained = top_score < abstain_threshold

        if is_answerable:
            answerable_total += 1
            if not abstained:
                answerable_passed += 1
                status = "PASS (Retained)"
            else:
                status = "FAIL (False Abstain)"
        else:
            unanswerable_total += 1
            if abstained:
                unanswerable_abstained += 1
                status = "PASS (Correctly Abstained)"
            else:
                status = "FAIL (False Answer Attempted)"

        results.append({
            "id": qid,
            "query": query,
            "is_answerable": is_answerable,
            "top_score": top_score,
            "abstained": abstained,
            "status": status
        })

        print(f"[{qid}] Ans={is_answerable!s:<5} TopScore={top_score:7.3f} -> {status}")

    unans_recall = (unanswerable_abstained / unanswerable_total * 100.0) if unanswerable_total else 0.0
    ans_retention = (answerable_passed / answerable_total * 100.0) if answerable_total else 0.0

    print("-" * 80)
    print(f"Unanswerable Query Abstain Recall : {unanswerable_abstained}/{unanswerable_total} ({unans_recall:.1f}%)")
    print(f"Answerable Query Pass Rate        : {answerable_passed}/{answerable_total} ({ans_retention:.1f}%)")
    print("=" * 80)

    return {
        "abstain_threshold": abstain_threshold,
        "unanswerable_recall": unans_recall,
        "answerable_retention": ans_retention,
        "details": results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG abstain threshold on gold queries")
    parser.add_argument("--abstain-threshold", type=float, default=-6.0, help="Rerank score threshold below which query is rejected")
    parser.add_argument("--dataset", type=str, default=os.path.join(os.path.dirname(__file__), "gold_cropchain.json"), help="Path to gold queries dataset")
    parser.add_argument("--repo-id", type=str, default="repo_3e9c5a1753fa", help="Indexed repository collection name")
    args = parser.parse_args()

    evaluate_threshold(args.dataset, args.abstain_threshold, args.repo_id)
