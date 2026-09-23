# SPDX-License-Identifier: MIT
"""Resolution names its answer: which artifacts answered a step, and which missed.

This is not diagnostics polish. *"Price refusal is not graph absence"* was a
day's confusion on its own, and an `incomplete: N/2570` line
proves less than it looks: it states a count, not which of two stages
declined. The same count is produced by a missing `op_graph` and by a
`price_list` that refused -- one means the structure was never traced, the
other that it was traced and a price is unavailable, and they are answered by
different work.

So a step keeps a ledger. Each entry that answered is recorded by **key and
digest**, so "which artifact answered this number" has an answer that does not
depend on anyone's memory of which store was mounted. Each key that missed is
recorded with **the refusal that declined it**, so the kind and the reason are
both in the record.

`report()` never states a count on its own, and `require_complete()` is the
refusal a step ends on, naming every key that
missed rather than the first. The misses are collected rather than raised
where they happen for that reason alone: a step that stops at the first
refusal reports one name where it could have reported all of them, and the
next run then discovers the second.
"""

from dataclasses import dataclass, field

from .keys import Key
from .rules import ArtifactRefusal, Rule


@dataclass(frozen=True, slots=True)
class Answer:
    """One artifact that answered, named by what it is rather than where it was."""

    key: Key
    digest: str

    @property
    def text(self) -> str:
        return f"answered  {self.key}  {self.digest}"


@dataclass(frozen=True, slots=True)
class Miss:
    """One key that did not answer, and the named refusal that declined it."""

    key: Key
    rule: Rule
    because: str

    @property
    def text(self) -> str:
        return f"missed    {self.key}  [{self.rule.name}] {self.because}"


@dataclass
class Resolution:
    """The ledger of one predicted step: what answered it, and what did not."""

    step: str
    answers: list[Answer] = field(default_factory=list)
    misses: list[Miss] = field(default_factory=list)

    def answered(self, key: Key, digest: str) -> None:
        """Record that this entry answered, by key and by the digest it carried."""
        self.answers.append(Answer(key, digest))

    def missed(self, key: Key, declined: ArtifactRefusal) -> None:
        """Record that this key did not answer, and what said so."""
        self.misses.append(Miss(key, declined.rule, declined.what))

    @property
    def complete(self) -> bool:
        return not self.misses

    def report(self) -> str:
        """The count **and** its decomposition, which is the only honest form.

        `incomplete: 2/3` is what this replaces: two steps with one missing
        `op_graph` and one refused `price_list` produce the same count and
        need different work.
        """
        asked = len(self.answers) + len(self.misses)
        head = f"{self.step}: {len(self.answers)} of {asked} answered"
        return "\n".join([head] + [line.text for line in self.answers + self.misses])

    def require_complete(self) -> None:
        """Refuse unless every key answered, naming each that did not."""
        if self.complete:
            return
        raise ArtifactRefusal(
            Rule.RESOLUTION,
            f"{self.step} was answered by {len(self.answers)} of "
            f"{len(self.answers) + len(self.misses)} artifacts, and these missed: "
            + "; ".join(f"{miss.key}: {miss.because}" for miss in self.misses),
            "a count says how many and not which, so a missing graph and a "
            "refused price read alike; each key above names the stage that "
            "has to answer before this step has a number",
        )
