# SPDX-License-Identifier: MIT
"""What a refusal says, and the one list that makes an unknown key legible.

A refusal here is a result, not an error path, so it is built to be read by
whoever hit it. Three parts, always: which rule declined, what in the document
tripped it, and what would satisfy it. A message that names only the first two
tells a user their spec is wrong and leaves them to guess the fix, and the fix
is the only part they need.

The separation rule is the one that decides every inclusion question:

    the spec describes the machine; ATOM's config describes the deployment;
    anything ATOM could configure belongs to ATOM, and Compass reads it there.

It is enforced by the schema being **closed** rather than by the list below. A
document key that is not a declared field is refused wherever it sits, so a
spec cannot carry a tensor-parallel width, a block size or a memory-utilisation
fraction even though no rule anywhere names them -- and it equally cannot carry
the next such knob, which nobody has thought to list yet. A denylist would have
to be complete to work; a closed schema is complete by construction.

`DEPLOYMENT_OWNED` exists only to improve the message. When the unknown key is
one of the knobs a spec author is most likely to reach for, the refusal says
where ATOM configures it instead of saying merely that it is unrecognised. The
values name engine-argument fields rather than command-line flags because the
field names are what a test can check against the engine's own surface, and a
list that drifts away from that surface would be folklore inside a month.
"""

import enum


class Rule(enum.Enum):
    """The rules a spec is read against, as a refusal names them."""

    SEPARATION = "the spec describes the machine, not the deployment"
    NO_DEFAULTS = "a runtime constant has no default"
    DERATE = "a spec-peak number carries a derate"
    PINNED_STACK = "the constants are pinned to a software stack"
    TOKENIZER_IDENTITY = "a tokenizer is measured or it is refused"
    SHAPE = "the document has the shape the schema declares"


#: Knobs the engine already owns, mapped to where it owns them. Not the
#: mechanism -- the closed schema is -- only the part of the message that turns
#: "unrecognised" into "ATOM configures this, and Compass reads it from there".
DEPLOYMENT_OWNED = {
    "block_size": "EngineArgs.block_size",
    "cudagraph_capture_sizes": "EngineArgs.cudagraph_capture_sizes",
    "cudagraph_mode": "EngineArgs.cudagraph_mode",
    "gpu_memory_utilization": "EngineArgs.gpu_memory_utilization",
    "kv_cache_dtype": "EngineArgs.kv_cache_dtype",
    "level": "EngineArgs.level",
    "max_model_len": "EngineArgs.max_model_len",
    "max_num_seqs": "EngineArgs.max_num_seqs",
    "tensor_parallel_size": "EngineArgs.tensor_parallel_size",
}


class SpecRefusal(Exception):
    """A declined answer that names its rule and what would satisfy it."""

    def __init__(self, rule: Rule, what: str, remedy: str) -> None:
        self.rule = rule
        self.what = what
        self.remedy = remedy
        super().__init__(f"{rule.value}: {what}. {remedy}")


class StackMismatch(UserWarning):
    """The stack a spec was measured against is not the stack now loaded."""


class FingerprintMismatch(UserWarning):
    """The tokenizer now loaded is not the one whose rates were measured."""


def refuse_unknown_key(path: str) -> None:
    """Decline a key the schema does not declare, saying why it is not one."""
    owner = DEPLOYMENT_OWNED.get(path.rsplit(".", 1)[-1])
    if owner is not None:
        raise SpecRefusal(
            Rule.SEPARATION,
            f"`{path}` states how ATOM was launched, not what the machine is",
            f"remove it; {owner} carries it and Compass reads it from there",
        )
    raise SpecRefusal(
        Rule.SEPARATION,
        f"`{path}` is not a field of this schema",
        "the schema is closed: a property of the machine is added to the field "
        "table deliberately, and anything ATOM could configure stays in ATOM's "
        "config, where Compass reads it",
    )
