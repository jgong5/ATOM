# SPDX-License-Identifier: MIT
"""compass fit host --stated-recv (#3720): W_recv stated for each forward kind where no record stamps a later
pipeline stage's receive, as MiniMax-M3.1's P PP4 records do not; a run's later stage refuses a law with none."""

import math

import pytest

from atom.compass.fit.host import stated_recv


def test_w_recv_is_stated_for_every_kind_the_entry_fits():
    laws = {"prefill": {"E": [1e-3, 0, 0]}, "middle": {"E": [9e-3, 0, 0]}}
    assert stated_recv("0", laws) == {"prefill": [0.0, 0.0, 0.0], "middle": [0.0, 0.0, 0.0]}
    assert stated_recv("0.0002", laws)["middle"] == [0.0002, 0.0, 0.0]
    assert stated_recv(None, laws) == {}


def test_a_bad_value_or_a_fitted_w_recv_is_refused():
    laws = {"prefill": {"E": [1e-3, 0, 0]}}
    for word in ("-1", "x", "nan", str(math.inf)):
        with pytest.raises(ValueError, match="not a number of seconds"):
            stated_recv(word, laws)
    with pytest.raises(ValueError, match="fit W_recv from the records already"):
        stated_recv("0", {"prefill": {"W_recv": [1e-4, 0, 0]}})
