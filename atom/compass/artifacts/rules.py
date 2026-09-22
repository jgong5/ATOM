# SPDX-License-Identifier: MIT
"""What a refusal from the artifact store says, and the rules it declines under.

D41 opens with eight incidents that share one shape: the number was fine and
the question of *which artifact answered* was not. Each rule below is one of
those incidents written as something the store can decline, so the failure
arrives as a named refusal at the moment it is made rather than as a plausible
number a week later.

A refusal carries three parts, as everywhere else in Compass: which rule
declined, what tripped it, and what would satisfy it. The third part is the
only one the caller needs, and it is the one an exception message usually
leaves out.
"""

import enum


class Rule(enum.Enum):
    """The rules an artifact is stored and resolved under, as a refusal names them."""

    KEY_IS_A_TUPLE = "a key is a tuple, never a path"
    ONE_NAMING_FUNCTION = "write and read sides go through one naming function"
    EVERY_RANK_WRITES = "a per-rank artifact carries every rank of its topology"
    PROVENANCE = "an entry names every executed source root"
    IMMUTABLE = "a handed-off entry is immutable"
    RESOLUTION = "resolution names its answer"


class ArtifactRefusal(Exception):
    """A declined answer that names its rule and what would satisfy it."""

    def __init__(self, rule: Rule, what: str, remedy: str) -> None:
        self.rule = rule
        self.what = what
        self.remedy = remedy
        super().__init__(f"{rule.value}: {what}. {remedy}")
