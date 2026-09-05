"""Does a quadratic-in-sequence-length feature explain prefill?

Keying prefill on token count assumes cost is linear in tokens. Two steps with
the same token count differ by 74% when the tokens are distributed differently,
which says the missing term is attention: quadratic *within* a sequence, so it
depends on sum(len_i^2) and not on sum(len_i).

Fit both and compare. Features come from the step table, which records
num_scheduled_tokens per request, so nothing new has to be measured.
"""
import glob, json
import numpy as np

rows = []
for path in sorted(glob.glob('compass_ops/pm_n*.jsonl') + glob.glob('compass_ops/sq_*.jsonl')):
    for rec in (json.loads(l) for l in open(path)):
        if not rec.get('num_prefill_tokens'):
            continue
        lens = [int(x) for x in rec['num_scheduled_tokens']]
        rows.append({
            'tag': path.split('/')[-1].replace('.jsonl', ''),
            'tokens': sum(lens),
            'sumsq': sum(x * x for x in lens),
            'seqs': len(lens),
            'step': rec['seconds'],
        })
# One prefill step per configuration: the last is the warmed one.
seen, uniq = set(), []
for r in reversed(rows):
    if r['tag'] in seen:
        continue
    seen.add(r['tag'])
    uniq.append(r)
uniq.sort(key=lambda r: r['tokens'])
print('%-12s %7s %6s %12s %10s' % ('config', 'tokens', 'seqs', 'sum(len^2)', 'step ms'))
for r in uniq:
    print('%-12s %7d %6d %12d %9.2f' % (r['tag'], r['tokens'], r['seqs'], r['sumsq'], r['step'] * 1e3))

y = np.array([r['step'] for r in uniq])
def fit(cols, names):
    X = np.column_stack(cols + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    err = 100 * (pred - y) / y
    return beta, pred, err, names + ['const']

tok = np.array([r['tokens'] for r in uniq], float)
sq = np.array([r['sumsq'] for r in uniq], float)

print()
for cols, names, label in (
        ([tok], ['tokens'], 'tokens only (what keying on token count assumes)'),
        ([tok, sq], ['tokens', 'sum(len^2)'], 'tokens + quadratic attention term')):
    beta, pred, err, nm = fit(cols, names)
    a = np.abs(err)
    print('%s' % label)
    print('   coefficients: ' + ', '.join('%s=%.4g' % (n, b) for n, b in zip(nm, beta)))
    print('   error: median %.1f%%, worst %.1f%%' % (np.median(a), a.max()))
print()
beta, pred, err, _ = fit([tok, sq], ['tokens', 'sum(len^2)'])
print('%-12s %7s %10s %10s %8s' % ('config', 'tokens', 'measured', 'fitted', 'err'))
for r, p, e in zip(uniq, pred, err):
    print('%-12s %7d %9.2fms %9.2fms %+7.1f%%' % (r['tag'], r['tokens'], r['step'] * 1e3, p * 1e3, e))
