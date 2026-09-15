"""Ordered immutable static segments, interleaved with live operator slots."""

from dataclasses import dataclass, field

from atom.compass.core.cost.prepared import PreparedOperator


@dataclass(frozen=True)
class StaticSegment:
    operators: tuple


@dataclass(frozen=True)
class PreparedPlan:
    steps: tuple
    identity: bytes = field(repr=False)


def prepare_plan(operators):
    from atom.compass.core.cost.priced import HOST_SYNC

    steps, pending = [], []
    for index, op in enumerate(operators):
        if isinstance(op, PreparedOperator):
            if op.name not in HOST_SYNC:
                pending.append(op)
            continue
        if pending:
            steps.append(StaticSegment(tuple(pending)))
            pending = []
        # Mutable slots stay live, including a host marker whose name could
        # change. Their contents are evaluated from the bound cohort.
        steps.append(index)
    if pending:
        steps.append(StaticSegment(tuple(pending)))
    # The contained snapshots are already immutable, type-preserving bytes.
    # Marshal here needs no JSON validation: the schema is owned above.
    import marshal

    identity = marshal.dumps(tuple(op.snapshot if isinstance(op, PreparedOperator)
                                   else None for op in operators))
    return PreparedPlan(tuple(steps), identity)


class PreparedGraph(dict):
    """A bound graph with an immutable operator order and a reusable plan.

    Replacing its operator sequence makes it an ordinary graph again. Dynamic
    operator dictionaries remain live; the plan never snapshots their values.
    """

    __slots__ = ("_planned_operators", "_plan")

    def __init__(self, graph, plan):
        super().__init__(graph)
        self["ops"] = tuple(self.get("ops") or ())
        self._planned_operators = self["ops"]
        self._plan = plan

    def current_plan(self):
        return self._plan if self.get("ops") is self._planned_operators else None
