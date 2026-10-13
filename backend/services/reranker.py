from typing import List, Dict, Any

_cross_encoder = None

def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def rerank_results(query: str, candidates: List[Any]) -> List[Dict[str, Any]]:
    pairs = []
    valid_candidates = []

    for point in candidates:
        payload = getattr(point, "payload", {}) or {}
        content = payload.get("content") if isinstance(payload, dict) else None
        if content:
            valid_candidates.append(point)
            pairs.append((query, content))

    if not pairs:
        return []

    cross_encoder = get_reranker()
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
    return [
        {
            "point": point,
            "score": float(score),
        }
        for point, score in ranked
    ]
