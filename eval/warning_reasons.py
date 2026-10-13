import json

data = json.load(open('eval/results/e2e_35.json', encoding='utf-8'))

counts = {'VERIFIED': 0, 'WARNING': 0, 'REJECTED': 0}
for d in data:
    counts[d['decision']] += 1

print('Overall Decisions:')
print(f"VERIFIED: {counts['VERIFIED']} | WARNING: {counts['WARNING']} | REJECTED: {counts['REJECTED']}\n")

# Break down WARNING reasons
# Note: a query can have multiple contributing reasons (e.g., exists_not_retrieved AND L4 unsupported)
warning_counts = {
    'exists_not_retrieved': 0,
    'unverifiable_external': 0,
    'L2 (structural)': 0,
    'L4 (evidence assertion)': 0,
    'exactly 1 invented token': 0,
    'other': 0
}

rows = []

for d in data:
    if d['decision'] == 'WARNING':
        idx = d['idx']
        q = d['query']
        l1 = d['per_layer'].get('layer1_existence', {})
        l2 = d['per_layer'].get('layer2_structural', {})
        l3 = d['per_layer'].get('layer3_callgraph', {})
        l4 = d['per_layer'].get('layer4_evidence', {})
        
        enr = l1.get('exists_not_retrieved', [])
        ue = l3.get('unverifiable_external', [])
        l2_fail = (l2.get('status') != 'not_checked' and l2.get('passed') is False)
        l4_fail = (l4.get('status') != 'not_checked' and l4.get('passed') is False)
        inv = l1.get('invented_tokens', [])
        inv_1 = (len(inv) == 1)
        
        reasons_list = []
        if enr:
            warning_counts['exists_not_retrieved'] += 1
            reasons_list.append(f"exists_not_retrieved ({len(enr)}: {enr})")
        if ue:
            warning_counts['unverifiable_external'] += 1
            reasons_list.append(f"unverifiable_external ({len(ue)})")
        if l2_fail:
            warning_counts['L2 (structural)'] += 1
            reasons_list.append(f"L2: {l2.get('violations')}")
        if l4_fail:
            warning_counts['L4 (evidence assertion)'] += 1
            reasons_list.append("L4: unsupported assertions")
        if inv_1:
            warning_counts['exactly 1 invented token'] += 1
            reasons_list.append(f"1 invented: {inv}")
        if not reasons_list:
            warning_counts['other'] += 1
            reasons_list.append("other")
            
        rows.append((idx, q, "; ".join(reasons_list)))

print("Table of WARNING Reasons (Contributing Counts across 17 WARNING queries):")
print("-" * 60)
for k, v in warning_counts.items():
    print(f"  {k:<28}: {v}")
print("-" * 60)

print("\nPer-Query WARNING Breakdown:")
print(f"{'Idx':<4} | {'Query':<52} | {'Reasons'}")
print("-" * 110)
for idx, q, r in rows:
    print(f"[{idx:2d}] | {q:<52} | {r}")
