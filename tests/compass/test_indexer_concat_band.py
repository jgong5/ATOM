# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A prefill indexer band stays under 2 GiB as aiter's `fp8_mqa_logits`
allocates it: the step's keys rounded up to 256 columns."""

import importlib
import pathlib
import sys
import types

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import logits_rows

# Without the package's own `__init__`: the `atom.model_ops.indexer` package
# imports Triton kernels that need a GPU driver to load, and `rows.py` and the
# siblings it imports relatively do not. The package stands in by its path.
_PACKAGE = "atom.model_ops.indexer"
_DIR = pathlib.Path(__file__).resolve().parents[2] / "atom/model_ops/indexer"
_held = sys.modules.get(_PACKAGE)
if _held is None:
    sys.modules[_PACKAGE] = types.ModuleType(_PACKAGE)
    sys.modules[_PACKAGE].__path__ = [str(_DIR)]
try:
    rows_module = importlib.import_module(f"{_PACKAGE}.rows")
finally:
    if _held is None:
        del sys.modules[_PACKAGE]


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    monkeypatch.delenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", raising=False)


def _bands(rows, keys):
    concat = rows_module.ConcatKeys(None, None, keys, None)
    return rows_module.concat_rows(torch.zeros(rows, dtype=torch.int32), concat).bands


def test_a_band_on_unaligned_keys_stays_under_2_gib_as_allocated():
    """15,360 rows on 43,136 keys: 12,416 rows fit the exact width, but aiter
    allocates them 43,264 wide, past 2^31 bytes."""
    bands = _bands(15360, 43136)
    assert bands == ((0, 12288), (12288, 15360))
    assert all((end - first) * 43264 * 4 < 2**31 for first, end in bands)


@pytest.mark.parametrize("keys", [256, 32768, 43264, 1 << 20])
def test_keys_on_256_band_at_their_own_width(keys):
    band = logits_rows(15360, keys)
    assert _bands(15360, keys)[0] == (0, band)
