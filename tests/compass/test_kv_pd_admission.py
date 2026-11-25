# SPDX-License-Identifier: MIT
"""Decode admission (`pd_admission`) over the simulated connector, as over
Mooncake's consumer: the 1P1D recipe's decode runs it, so a simulated decode
starts with it and its real `Scheduler` defers a receive past its budget.

The engines are `test_kv_remote_prefill`'s: real schedulers stepped in
`EngineCore`'s order on scripted clocks."""

from types import SimpleNamespace

import pytest
from test_kv_remote_prefill import (
    PROMPT,
    R,
    SEND,
    admit,
    at,
    geometry,  # noqa: F401  a fixture
    pd,  # noqa: F401  a fixture
    prefilled,
)

from atom.kv_transfer.disaggregation.pd_admission import PdAdmissionController
from atom.model_engine.sequence import SequenceStatus


def _config(engine, **options):
    """What `PdAdmissionController.from_config` reads of a decode's config."""
    return SimpleNamespace(
        kv_transfer_config={"pd_admission": {"enabled": True, **options}},
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        sequence_parallel_size=1,
        parallel_config=SimpleNamespace(),
        pool_entry_bytes={"kv": 1000},
        num_kvcache_blocks=engine.block_manager.kv.num_blocks,
    )


def test_a_simulated_decode_admits_a_receive_and_prefill_does_not(pd):
    assert pd.d.kv_connector.supports_pd_admission is True
    assert pd.p.kv_connector.supports_pd_admission is False
    made = PdAdmissionController.from_config(
        _config(pd.d), pd.d.kv_connector, pd.d.block_manager, first_decode_tokens=1
    )
    assert made is not None and made.block_bytes == 1000
    with pytest.raises(ValueError, match="requires a Mooncake consumer"):
        PdAdmissionController.from_config(
            _config(pd.p), pd.p.kv_connector, pd.p.block_manager, first_decode_tokens=1
        )


def test_the_receive_range_is_the_prompt_less_a_lined_up_prefix(pd, seq_factory):
    (x,) = prefilled(pd, 1, R)
    params = dict(x.kv_transfer_params_output)
    seq = seq_factory(PROMPT, kv_transfer_params=params)
    blocks = len(params["remote_block_ids"])
    d = pd.d.kv_connector
    assert d.pd_receive_block_offset(seq, blocks, 0) == 0
    assert d.pd_receive_block_offset(seq, blocks, 1) == 1
    assert d.pd_receive_block_offset(seq, blocks, blocks) == 0, "whole prompt hit"
    assert pd.p.kv_connector.pd_receive_block_offset(seq, blocks, 1) is None
    other = seq_factory(PROMPT, kv_transfer_params=params | {"block_size": 64})
    assert d.pd_receive_block_offset(other, blocks, 1) == 0, "pages do not line up"
    local = seq_factory(PROMPT)
    assert d.pd_receive_block_offset(local, blocks, 1) is None


def test_a_receive_past_the_byte_budget_waits_for_the_one_pending(pd, seq_factory):
    """Two prefilled requests reach decode together under a budget of one
    request's receive: the first is admitted and parked for its write, the
    second deferred, holding no blocks, until the first's credit is gone."""
    xs = prefilled(pd, 2, R)
    at(pd.rt_d, SEND)
    ys = [
        seq_factory(PROMPT, kv_transfer_params=dict(x.kv_transfer_params_output))
        for x in xs
    ]
    blocks = len(xs[0].kv_transfer_params_output["remote_block_ids"])
    pd.d.pd_admission = PdAdmissionController(
        blocks * 1000, 1000, pd.d.kv_connector, pd.d.block_manager
    )
    for y in ys:
        pd.d.add(y)
    admit(pd.d)
    assert ys[0].status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert ys[1].status is SequenceStatus.WAITING and not ys[1].block_table
    assert pd.d.pd_admission.pending_bytes == blocks * 1000
    assert pd.d.pd_admission.deferred["bytes"] == 1
