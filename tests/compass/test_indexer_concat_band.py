# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A prefill indexer band stays under 2 GiB as aiter's `fp8_mqa_logits`
allocates it: the step's keys rounded up to 256 columns."""

import importlib.util
import pathlib

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import logits_rows

# By path: the `atom.model_ops.indexer` package imports Triton kernels that
# need a GPU driver to load, and `rows.py` alone does not.
_PATH = pathlib.Path(__file__).resolve().parents[2] / "atom/model_ops/indexer/rows.py"
_spec = importlib.util.spec_from_file_location("_indexer_rows_under_test", _PATH)
rows_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rows_module)


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
