import os
import sys
import json
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import get_or_restore_engine
from backend.services.embedding import query_embedding
from backend.services.storage import search_isolated_query
from backend.services.ollama_service import generate_code_answer

repo_id = 'repo_3e9c5a1753fa'
engine = get_or_restore_engine(repo_id)
gold = json.load(open('eval/gold_v2.json', encoding='utf-8'))
queries = [item['q'] for item in gold['answerable']]

async def run_e2e():
    results = []
    print(f'Starting E2E evaluation for {len(queries)} queries...')
    for i, q in enumerate(queries, 1):
        # 1. Dense top-5
        cands = search_isolated_query(repo_id, query_embedding(q), limit=5)
        reranked = [{'point': c, 'score': c.score} for c in cands]
        
        # 2. Ollama generation
        llm_out = await generate_code_answer(q, reranked)
        answer = llm_out.get('explanation', '')
        claims = llm_out.get('claims', {})
        ollama_failed = 'error' in llm_out or not answer
        
        # 3. Verification
        rep = engine.verify(
            claims=claims,
            retrieved_chunks=reranked,
            ollama_failed=ollama_failed,
            answer_text=answer
        )
        
        entry = {
            'idx': i,
            'query': q,
            'answer': answer,
            'claims': claims,
            'decision': rep['decision'],
            'trust_score': rep['trust_score'],
            'per_layer': rep.get('layers', {}),
            'missing_symbols': rep.get('layers', {}).get('layer1_existence', {}).get('missing_symbols', []),
            'violations': (
                rep.get('layers', {}).get('layer2_structural', {}).get('violations', []) +
                rep.get('layers', {}).get('layer3_callgraph', {}).get('violations', [])
            ),
            'unverifiable_external': rep.get('layers', {}).get('layer3_callgraph', {}).get('unverifiable_external', []),
            'ungrounded_prose_identifiers': rep.get('layers', {}).get('layer1_existence', {}).get('ungrounded_prose_identifiers', []),
            'invented_tokens': rep.get('layers', {}).get('layer1_existence', {}).get('invented_tokens', [])
        }
        results.append(entry)
        dec = entry['decision']
        sc = entry['trust_score']
        print(f'[{i}/{len(queries)}] {dec} ({sc}) | {q}')

    with open('eval/results/e2e_35.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Completed all 35 queries! Saved to eval/results/e2e_35.json')

if __name__ == '__main__':
    asyncio.run(run_e2e())
