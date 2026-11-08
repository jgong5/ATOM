# SPDX-License-Identifier: MIT
"""A probed number is published with the p10 and p90 its probe measured,
stated in `provenance.origins` beside its subject. Readers use the value. A
count or a text is read, not sampled, and so is a field marked `exact`."""

from .fields import SCHEMA, Kind
from .machine import MachineSpec
from .rules import Rule, SpecRefusal

#: The kinds a probe samples.
BANDED = (Kind.QUANTITY, Kind.BYTES)


def refuse_unbanded(spec: MachineSpec) -> None:
    """Refuse to publish `spec` while a probed number states no band, naming
    each."""
    owed = [
        f.path
        for f in SCHEMA
        if f.kind in BANDED and not f.exact and f.path in spec.values
    ]
    unbanded = []
    for label in owed:
        found = spec.origin(label)
        if found["origin"] == "probed" and "p10" not in found:
            unbanded.append(f"`{label}`")
    if unbanded:
        raise SpecRefusal(
            Rule.BAND,
            f"{', '.join(unbanded)} probed with no p10 and p90",
            "write the band of the probe's records beside each in "
            "`provenance.origins`; a probed number alone reads as more certain "
            "than its probe measured it",
        )
