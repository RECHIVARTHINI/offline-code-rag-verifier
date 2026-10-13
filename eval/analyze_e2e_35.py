import json

data = json.load(open('eval/results/e2e_35.json', encoding='utf-8'))
counts = {'VERIFIED': 0, 'WARNING': 0, 'REJECTED': 0}
for d in data:
    counts[d['decision']] += 1

print("Counts:")
print(f"VERIFIED: {counts['VERIFIED']} | WARNING: {counts['WARNING']} | REJECTED: {counts['REJECTED']}")
print()
print("Non-VERIFIED Queries and their INVENTED tokens:")
print(f"{'Idx':<4} | {'Decision':<8} | {'Query':<55} | {'Invented Tokens'}")
print("-" * 110)
for d in data:
    if d['decision'] != 'VERIFIED':
        inv = d.get('per_layer', {}).get('layer1_existence', {}).get('invented_tokens', [])
        q = d['query']
        print(f"{d['idx']:<4} | {d['decision']:<8} | {q:<55} | {inv}")

print()
print("Answers 4, 17, and 20 check:")
for idx in [4, 17, 20]:
    match = next(d for d in data if d['idx'] == idx)
    inv = match.get('per_layer', {}).get('layer1_existence', {}).get('invented_tokens', [])
    print(f"Answer {idx:2d}: Decision={match['decision']:<8} | Trust Score={match['trust_score']:.3f} | Invented Tokens={inv}")
