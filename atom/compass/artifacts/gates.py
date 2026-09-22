# SPDX-License-Identifier: MIT
"""Every gate's state, written into the artifact and checked there.

*"A dead gate is worse than no gate."* A `PRICE_KERNELS` gate read
`WORLD_SIZE`, which the engine never sets -- it spawns its own ranks -- so the
gate was false in every worker and per-kernel breakdowns were taken at TP=2
all along. **The proof was in the artifact**: the price list written under the
supposedly-off gate carried breakdowns for 164 of its 237 entries. Worse, a
device fault had been gated "off under parallelism" on the strength of that
gate; five runs then faulted and each fault was read as evidence about
whatever had changed most recently.

So a gate is recorded as three things, and the third is the one the incident
turns on. A gate states **what it read** to decide its state, and a recorded
gate that read `WORLD_SIZE` disagrees with a gate in force that reads
something else **even when both say off** -- because they are not the same
gate, and the agreement is a coincidence of one being dead. `resolved_from`
cannot be blank: a gate that cannot say what it read is the dead gate before
anyone has noticed.

The check compares the state **recorded in the entry** against the state in
force. Re-reading the flag is what the incident did: the flag answered `off`
in every worker, consistently and wrongly, and nothing held the answer beside
the artifact it had shaped.

The flag that downgrades an invalidation mismatch to a warning is not offered
for a gate disagreement. A warning is what a dead gate already produces, and
the point of recording the state is to get an answer that is not one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .rules import ArtifactRefusal, Rule


@dataclass(frozen=True, slots=True)
class Gate:
    """One gate: its name, the state it resolved to, and what it read."""

    name: str
    state: str
    resolved_from: str

    def __post_init__(self) -> None:
        for field in ("name", "state", "resolved_from"):
            if (
                not isinstance(getattr(self, field), str)
                or not getattr(self, field).strip()
            ):
                raise ArtifactRefusal(
                    Rule.GATE_STATE,
                    f"a gate states {getattr(self, field)!r} for its {field}",
                    "name the gate, the state it resolved to, and what it read "
                    "to decide; a gate that cannot say what it read is the "
                    "dead gate before anyone has noticed",
                )

    @property
    def text(self) -> str:
        return f"`{self.name}` {self.state} (from {self.resolved_from})"

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "resolved_from": self.resolved_from,
        }

    @classmethod
    def from_json(cls, document: object) -> "Gate":
        if not isinstance(document, Mapping):
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                f"{document!r} is not a recorded gate",
                "a recorded gate states its name, its state and what it read",
            )
        return cls(
            name=document.get("name", ""),
            state=document.get("state", ""),
            resolved_from=document.get("resolved_from", ""),
        )


@dataclass(frozen=True, slots=True)
class GateState:
    """The state of every gate that shaped an artifact, as the artifact holds it."""

    gates: tuple[Gate, ...]

    def __post_init__(self) -> None:
        named = [gate.name for gate in self.gates]
        if len(set(named)) != len(named):
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                f"two gates share a name in {sorted(named)}",
                "one row per gate; two rows for one name cannot both be checked",
            )

    @classmethod
    def of(cls, *gates: Gate) -> "GateState":
        """The gates that shaped this artifact. `GateState.of()` says none did."""
        return cls(tuple(gates))

    @property
    def by_name(self) -> Mapping[str, Gate]:
        return {gate.name: gate for gate in self.gates}

    @property
    def text(self) -> str:
        return ", ".join(gate.text for gate in self.gates) or "no gates"

    def check(self, in_force: "GateState") -> None:
        """Refuse unless every gate agrees with the one in force, and on what it read."""
        if not isinstance(in_force, GateState):
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                f"{in_force!r} is not a gate state",
                "state the gates in force, even when there are none; "
                "`GateState.of()` says that and an omission does not",
            )
        recorded, running = self.by_name, in_force.by_name
        disagreements = [
            f"`{name}` shaped this entry and is not in force now"
            for name in sorted(set(recorded) - set(running))
        ] + [
            f"`{name}` is in force now and did not shape this entry"
            for name in sorted(set(running) - set(recorded))
        ]
        for name in sorted(set(recorded) & set(running)):
            was, now = recorded[name], running[name]
            if was.state != now.state:
                disagreements.append(
                    f"`{name}` was {was.state} when this entry was made and is "
                    f"{now.state} now"
                )
            elif was.resolved_from != now.resolved_from:
                disagreements.append(
                    f"`{name}` read {was.resolved_from} when this entry was made "
                    f"and reads {now.resolved_from} now, both saying {now.state}"
                )
        if disagreements:
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                "; ".join(disagreements),
                "the artifact carries the state of every gate that shaped it, "
                "and this load is under other gates; a gate that reads a "
                "different variable is a different gate even when it agrees, "
                "which is how per-kernel breakdowns were taken at TP=2 under a "
                "gate that was off in every worker",
            )

    def as_json(self) -> list:
        return [gate.as_json() for gate in sorted(self.gates, key=lambda g: g.name)]

    @classmethod
    def from_json(cls, document: object) -> "GateState":
        if not isinstance(document, Sequence) or isinstance(document, (str, bytes)):
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                f"{document!r} is not a recorded gate state",
                "an entry lists the gates that shaped it, and an empty list is "
                "the statement that none did",
            )
        return cls(tuple(Gate.from_json(gate) for gate in document))
