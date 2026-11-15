# SPDX-License-Identifier: MIT
"""A probed number's band is its two ends, the low one first, and only a
probed field states one."""

import pytest
from test_spec_origin import saved, three_origins

from atom.compass.spec import MachineSpec, SpecRefusal, merge

#: A probed number of `three_origins`, to state a band on.
BANDWIDTH = "device.memory.bandwidth_bytes_per_s"


@pytest.mark.parametrize(
    "band",
    [
        {"p10": 2.0, "p90": 1.0},
        {"p10": 1.0},
        {"p10": True, "p90": 2.0},
        {"p10": "1", "p90": 2.0},
        {"p10": {"a": 1.0}, "p90": {"b": 2.0}},
    ],
)
def test_a_band_is_two_ends_the_low_one_first(band):
    document = saved(merge(three_origins()).document)
    document["provenance"]["origins"][BANDWIDTH] |= band
    with pytest.raises(SpecRefusal, match="a band, a `p10` at most its `p90`"):
        MachineSpec.from_mapping(document)


def test_only_a_probed_field_states_a_band():
    document = saved(merge(three_origins()).document)
    assumed = document["provenance"]["origins"]["interconnect.inter_node.derate"]
    assumed |= {"p10": 1e-3, "p90": 2e-3}
    with pytest.raises(SpecRefusal, match="only kind with a subject"):
        MachineSpec.from_mapping(document)
