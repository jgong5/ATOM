# SPDX-License-Identifier: MIT
"""`measure ops` reads a measured point's shape back from the text it was
written as, each count by its name, whatever else the text states."""

import pytest

from atom.compass.bench import ops
from atom.compass.capture.forward import decode, prefill


@pytest.mark.parametrize(
    "shape",
    [
        prefill(7, 1),
        prefill(7, 1, gathered=20),
        prefill(7, 1, width=4),
        prefill(462, 1, gathered=8389, width=2),
        decode(4),
        decode(4, gathered=32),
        decode(4)._replace(width=8),
        decode(4, gathered=32)._replace(width=8),
        decode(4, rows=3),
    ],
    ids=str,
)
def test_a_shape_reads_back_from_its_text(shape):
    assert ops._shape(str(shape)) == shape
