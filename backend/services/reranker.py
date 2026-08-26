from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_results(query, candidates):
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

    scores = cross_encoder.predict(pairs)
    return [
        {
            "point": point,
            "score": float(score),
        }
        for point, score in zip(valid_candidates, scores)
    ]

