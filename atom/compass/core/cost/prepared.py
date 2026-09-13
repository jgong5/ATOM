"""Immutable operator identities owned by a prepared graph template.

Normal dictionaries remain live inputs to the price library. A prepared
operator instead owns a snapshot, so its exact signature and layout can be
computed once without relying on the identity of a mutable dictionary.
"""

from dataclasses import dataclass, field
import json
import marshal


@dataclass(frozen=True)
class PreparedOperator:
    name: str
    signature: str
    cost_key: str
    layout: str
    collective: bool
    snapshot: bytes = field(repr=False)

    def as_dict(self) -> dict:
        """A fresh mutable copy; modifying it cannot change this identity."""
        return marshal.loads(self.snapshot)

    def get(self, key, default=None):
        if key == "name":
            return self.name
        if key in ("context", "int_values"):
            return default
        return self.as_dict().get(key, default)

    def __getitem__(self, key):
        if key == "name":
            return self.name
        return self.as_dict()[key]


def prepare_static_operator(op):
    """Snapshot ordinary context-free input; leave unsupported inputs live.

    Marshal preserves list/tuple and scalar types, including signed zero, and
    rejects subclasses of the JSON scalar/container types. JSON validation
    additionally excludes code objects and byte buffers, which marshal can
    encode but whose Python representations need not identify their contents.
    Both checks happen once, at the explicit template preparation boundary.
    """
    if (type(op) is not dict or op.get("context") or op.get("int_values")
            or op.get("name") in (
                "aiter::unified_attention_with_output_base",
                "aiter::linear_attention_with_output_base")):
        return None
    try:
        snapshot = marshal.dumps(op)
        json.dumps(op)
    except (TypeError, ValueError, RecursionError):
        return None
    from atom.compass.core.cost.identity import cost_key
    from atom.compass.core.cost.library import _layout_fingerprint
    from atom.compass.runtime.microbench import _is_collective_op, signature_of

    try:
        signature = signature_of(op)
        layout = _layout_fingerprint(op)
        collective = _is_collective_op(op)
    except (KeyError, TypeError, ValueError):
        return None
    return PreparedOperator(op["name"], signature, cost_key(signature),
                            layout, collective, snapshot)


def materialize_graph(graph):
    """Return ordinary operator dictionaries for evidence serialization."""
    if graph is None:
        return None
    return dict(graph, ops=[op.as_dict() if isinstance(op, PreparedOperator)
                            else op for op in graph.get("ops") or ()])
