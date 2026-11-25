# SPDX-License-Identifier: MIT
"""Compass's simulated connector puts an engine in the P/D role, and moves the
NVFP4 pages and own-dtype draft rows, of the Mooncake deployment it stands in
for: a simulated MiniMax-M3.1 PP4 prefill with its DSpark draft beside an NVFP4
KV (the 1P1D recipe's) starts where the real one does."""

from types import SimpleNamespace

import pytest

from atom.config import Config
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.pd_producer import disaggregated_kv_role

M3 = "MiniMaxM3SparseForConditionalGeneration"


def _cfg(kv):
    spec = SimpleNamespace(method="dspark", use_dspark=lambda: True)
    return SimpleNamespace(
        pipeline_parallel_size=4,
        hf_config=SimpleNamespace(architectures=[M3]),
        speculative_config=spec,
        dspark=SimpleNamespace(confidence_schedule=False),
        kv_transfer_config=kv,
        enable_tbo=False,
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("name", ["compass", "mooncake"])
def test_the_simulated_connector_takes_the_role_mooncake_takes(name, role):
    kv = {"kv_connector": name, "kv_role": role}
    assert disaggregated_kv_role(SimpleNamespace(kv_transfer_config=kv)) == role


def test_the_simulated_connector_under_multi_takes_its_role():
    # deploy.sh's prefill on the Mooncake Store: the offload folds in as `multi`.
    kv = {
        "kv_connector": "multi",
        "connectors": [
            {"kv_connector": "compass", "kv_role": "kv_producer"},
            {"kv_connector": "mooncake_store", "kv_role": "kv_both"},
        ],
    }
    assert disaggregated_kv_role(SimpleNamespace(kv_transfer_config=kv)) == (
        "kv_producer"
    )


@pytest.mark.parametrize("name", ["compass", "mooncake"])
def test_a_pp4_dspark_producer_starts_simulated_as_real(name):
    Config._check_pp_support(_cfg({"kv_connector": name, "kv_role": "kv_producer"}))


def test_a_pp4_dspark_engine_with_no_pd_role_is_still_refused():
    with pytest.raises(ValueError, match="pp_speculative_methods declares"):
        Config._check_pp_support(_cfg(None))


def test_the_simulated_connector_moves_what_mooncake_moves():
    for flags in (
        KVConnectorFactory._moves_nvfp4_pages,
        KVConnectorFactory._moves_own_dtype_draft,
    ):
        assert flags["compass"] and flags["mooncake"]
    assert "compass" in KVConnectorFactory.nvfp4_page_connectors()
    assert "compass" in KVConnectorFactory.own_dtype_draft_connectors()
