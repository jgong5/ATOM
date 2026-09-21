# SPDX-License-Identifier: MIT
"""The deployments a synthetic run drives, and the floors between their parts.

A participant is the unit the clock coordinates, and it is not an OS process.
Three things and only three create one: a prefill/decode role boundary, a
pipeline stage, and an independent replica behind the router. Everything else
ATOM runs wide is already barriered and holds no clock of its own -- a
tensor-parallel group is one participant however many workers it has, because
its engine parks on one reply from rank 0 and the workers never read a clock; a
data-parallel group is one however many engines it has, because it reduces
across them every step. So a four-way tensor-parallel, two-way data-parallel
deployment is one participant, not eight, and widening either one changes the
GPU count in this module and nothing else.

The floors are declared per ordered pair because the matrix refuses an
incomplete one, and that refusal is the point: a peer missing from a row drops
out of the minimum taken over it, which raises the bound and hands out more time
than the peer allows. Three paths carry messages here and each has its own
floor. Two replicas behind the router exchange nothing directly, so any floor is
vacuously safe for them; they are declared at the router-relay floor rather than
at something larger, because a floor that is declared wide is only found to be
wrong when a message crosses it, and this is the declaration a real fleet would
make.
"""

import enum
from dataclasses import dataclass

from atom.compass.clock import (
    TRAFFIC_TO_ENGINE_FLOOR_SECONDS,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
)

#: The one traffic source. Every deployment has exactly one.
TRAFFIC_SOURCE = LpId("traffic-source")

#: Modelled admission delay between the traffic source and an engine, and back
#: out again on the response path.
ADMISSION_FLOOR_SECONDS = TRAFFIC_TO_ENGINE_FLOOR_SECONDS["serving"]

#: Router relay plus the simulated transfer of the cached keys and values. It is
#: the floor between the two roles of one deployment and between two independent
#: replicas alike: both are a relay hop at millisecond scale.
ROLE_BOUNDARY_FLOOR_SECONDS = 1.0e-3

#: A modelled send and receive of the intermediate tensors between two pipeline
#: stages of one replica. The tight one, and the reason stages belong together.
PIPELINE_STAGE_FLOOR_SECONDS = 1.0e-6


class Role(enum.Enum):
    """What a replica does with a request."""

    #: Prefill and decode in one deployment, which is every run without a role
    #: boundary.
    ENGINE = "engine"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class Replica:
    """One independent deployment: its role, its width, and its stage count.

    `tp` is the tensor-parallel width and contributes GPUs without contributing
    a participant. `pp` is the pipeline depth and contributes one participant
    per stage, which is the one group the collapse does not cover.
    """

    role: Role
    index: int
    tp: int
    pp: int

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    def stage_ids(self) -> tuple[LpId, ...]:
        """One identity per stage, zero-padded so the name order is the index order."""
        return tuple(
            LpId(f"{self.role.value}-{self.index:02d}.stage-{stage}")
            for stage in range(self.pp)
        )


@dataclass(frozen=True)
class Deployment:
    """A whole arrangement: one traffic source and some replicas."""

    name: str
    replicas: tuple[Replica, ...]

    @property
    def participants(self) -> tuple[LpId, ...]:
        ids = [TRAFFIC_SOURCE]
        for replica in self.replicas:
            ids.extend(replica.stage_ids())
        return tuple(ids)

    @property
    def gpus(self) -> int:
        return sum(replica.gpus for replica in self.replicas)

    def of_role(self, role: Role) -> tuple[Replica, ...]:
        return tuple(replica for replica in self.replicas if replica.role is role)


def _roles(count: int, tp: int, pp: int) -> tuple[Replica, ...]:
    """`count` prefill replicas and `count` decode replicas, all the same width."""
    return tuple(
        Replica(role, index, tp, pp)
        for role in (Role.PREFILL, Role.DECODE)
        for index in range(count)
    )


#: The arrangements ATOM is deployed in, smallest first. The participant count
#: is a property of the collapse and is asserted; the GPU count is derived from
#: the widths and is reported.
DEPLOYMENTS = (
    Deployment("tp4-one-server", (Replica(Role.ENGINE, 0, 4, 1),)),
    Deployment("tp4-prefill-tp4-decode", _roles(1, 4, 1)),
    Deployment("tp8-role-disaggregated", _roles(1, 8, 1)),
    Deployment(
        "tp8-pp4",
        (Replica(Role.PREFILL, 0, 8, 1), Replica(Role.DECODE, 0, 8, 4)),
    ),
    Deployment("eight-replicas-each-tp8", _roles(8, 8, 1)),
    Deployment("eight-replicas-each-tp8-pp4", _roles(8, 8, 4)),
)


def clock_parts(deployment: Deployment) -> tuple[LpRegistry, LookaheadMatrix]:
    """The registry and the complete lookahead matrix for one deployment."""
    registry = LpRegistry()
    for lp_id in deployment.participants:
        registry.register(lp_id)
    replica_of = {}
    for slot, replica in enumerate(deployment.replicas):
        for lp_id in replica.stage_ids():
            replica_of[lp_id] = slot
    matrix = LookaheadMatrix(registry)
    for source in registry.ids():
        for target in registry.ids():
            if source == target:
                continue
            link_class, floor = _link(replica_of.get(source), replica_of.get(target))
            matrix.declare(source, target, link_class, floor)
    return registry, matrix


def _link(source_replica, target_replica) -> tuple[LinkClass, float]:
    """Which path a message between these two takes, and its floor."""
    if source_replica is None or target_replica is None:
        return LinkClass.TRAFFIC_TO_ENGINE, ADMISSION_FLOOR_SECONDS
    if source_replica == target_replica:
        return LinkClass.PIPELINE_STAGE_TO_STAGE, PIPELINE_STAGE_FLOOR_SECONDS
    return LinkClass.PREFILL_TO_DECODE, ROLE_BOUNDARY_FLOOR_SECONDS
