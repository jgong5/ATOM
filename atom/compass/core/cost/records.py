"""Evidence-bearing result types shared by generic and family price readers."""


class AttestedAttentionRecord(dict):
    """A validated attention measurement with an independently witnessed count."""

    __slots__ = ("attested_launch_count",)

    def __init__(self, record, count, **metadata):
        super().__init__(record, **metadata)
        self.attested_launch_count = count
        self["attention_exact_override"] = True
        self["launch_count"] = count


class OperatorEventRecord(dict):
    """Whole-operator timing with unknown kernel count and no extra launch fee."""
