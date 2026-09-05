"""What expert parallelism puts in the graph, and whether it can be priced."""
import collections, glob, json, sys

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'compass_ops/ep*.json')):
    try:
        g = json.load(open(path))
    except Exception:
        continue
    ops = g.get('ops') or []
    if not ops:
        continue
    grouped = [o for o in ops if o.get('group')]
    moe = [o for o in ops if any(k in o['name'].lower()
                                 for k in ('moe', 'expert', 'topk', 'a2a', 'all_to_all',
                                           'dispatch', 'combine'))]
    print('== %s: %d ops, %d distinct' % (path, len(ops), len({o['name'] for o in ops})))
    print('   grouped  :', collections.Counter(o['name'] for o in grouped).most_common(6))
    print('   moe-ish  :', collections.Counter(o['name'] for o in moe).most_common(6))
    for o in moe[:3]:
        print('     %-46s %s' % (o['name'][:46], o['input_shapes'][:3]))
