# SPDX-License-Identifier: MIT
"""What ATOM passes each leaf of DeepSeek-V4.1's graphs, from its own capture.

    python v41_capture_truth.py OUT MODEL TP <trace ops shapes> -- <ATOM args>

The capture `trace ops` runs (`forward.capture`), on card 0 of the MI455 A0
node as it read, with each metadata argument and integer KV argument a traced
operator takes named as `v41_ops_rehearsal.identity` names it, among the
tensors `v41_ops_rehearsal.named` gives the capture's runner at that call. OUT
gets, by leaf signature (`ops.signature`) and slot path, each name ATOM passed.
"""

import argparse
import json
import pathlib
import sys
import tempfile
from collections import defaultdict


def _flat(path, value):
    """Each tensor in a traced argument, by the slot path its graph gives it."""
    import torch

    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _flat(f"{path}.{i}", v)
    elif type(value) is dict:
        for i, v in enumerate(value.values()):
            yield from _flat(f"{path}.{i}.1", v)


def main(out, model, tp, argv) -> None:
    here = pathlib.Path(__file__).parent
    sys.path[:0] = [str(here), str(here.parent)]  # its flat imports
    from test_capture_mi455_cells import A0
    from v41_ops_rehearsal import INTEGERS, identity, named

    from atom.compass import ir
    from atom.compass.bench import ops
    from atom.compass.capture import forward, graph
    from atom.compass.capture.forward import decode, prefill

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, action="append", default=[])
    parser.add_argument("--prefill", type=int, nargs=2, action="append", default=[])
    cut = argv.index("--")
    args = parser.parse_args(argv[:cut])
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    called, truth = {}, defaultdict(lambda: defaultdict(set))
    record, affine, parts = forward._Recorder._op, forward._affine, forward._parts

    def recorded(self, name, kind, slots, out_shapes):
        op = record(self, name, kind, slots, out_shapes)
        if self.on and self.runner is not None:
            taken = [
                (p, t)
                for slot, value in slots.items()
                for p, t in _flat(slot, value)
                if self.roles.get(forward._storage(t), ir.Role.METADATA)
                == ir.Role.METADATA
                or str(t.dtype) in INTEGERS
                and self.roles.get(forward._storage(t)) == ir.Role.KV
            ]
            if taken:
                pool = named(self.runner)
                called[id(op)] = op, {p: identity(t, pool) for p, t in taken}
        return op

    def bound_affine(shape, first, second):
        out = affine(shape, first, second)
        for (_, a), (_, b) in zip(first, out):
            if id(a) in called:
                called[id(b)] = b, called[id(a)][1]
        return out

    def split(items):
        for _, op in items:
            if id(op) in called:
                sig = ops.signature(json.loads(json.dumps(graph._json(op))))
                for path, name in called[id(op)][1].items():
                    truth[sig][path].add(name)
        return parts(items)

    forward._Recorder._op, forward._affine, forward._parts = (
        recorded,
        bound_affine,
        split,
    )
    for _ in forward.capture(
        model, tp, shapes, tempfile.mkdtemp(), argv[cut + 1 :], A0["arch"], read=A0
    ):
        pass
    got = {
        sig: {p: sorted(n) for p, n in paths.items()} for sig, paths in truth.items()
    }
    pathlib.Path(out).write_text(json.dumps(got, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])
