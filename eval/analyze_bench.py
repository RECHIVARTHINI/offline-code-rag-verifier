import json, sys

data = json.load(open('eval/results/bench_new.json'))
# Also get cases from verif_bench
sys.path.insert(0, '.')
import eval_orig.verif_bench as vb

rows = data['rows']
for i, (typ, label, fqn, cl) in enumerate(vb.cases):
    r = rows[i]
    if r['label'] == 0 and r['decision'] != 'VERIFIED':
        print(f"[{i}] {typ} | FQN: {fqn} | decision: {r['decision']} | layers: {r['layers']}")
