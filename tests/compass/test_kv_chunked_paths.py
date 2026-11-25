# SPDX-License-Identifier: MIT
"""The paths a chunked P->D transfer takes in a simulated deployment: the
router sends each request to both frontends at once (atomesh's
`dispatch_atom_chunked_internal`), so traffic stamps it alike on a request
path into each, prefill relays nothing, and prefill tells decode its write is
done on ``kv_write_done``. The prefill's `/kv_transfer_info` tells the router
it may chunk, as a Mooncake producer's does, naming no host."""

import asyncio
from importlib import metadata
from types import SimpleNamespace

import pytest

from atom.compass.clock import prefill_decode_table
from atom.compass.run import TRAFFIC
from atom.entrypoints.openai import api_server
from atom.utils.clock import LPRuntime

HOPS = {
    "admission_s": 2.0**-11,
    "ipc_s": 2.0**-10,
    "stream_s": 2.0**-10,
    "router_s": 2.0**-12,
}


def _names(table, lp, into=False):
    rows = table.channels_into(lp) if into else table.channels_from(lp)
    return sorted(c.name for c in rows)


def test_a_chunked_table_sends_each_request_to_both_frontends_and_relays_nothing():
    table = prefill_decode_table(
        **HOPS, kv_write_req_s=2.0**-9, dp=(1, 4), chunked=True
    )
    assert _names(table, TRAFFIC) == [
        "traffic->frontend-D:http",
        "traffic->frontend-P:http",
    ]
    assert _names(table, TRAFFIC, into=True) == ["frontend-D->traffic:stream"]
    names = {c.name for lp in table.registry.ids() for c in table.channels_from(lp)}
    assert not [n for n in names if n.endswith(":relay")]
    done = sorted(n for n in names if ":kv_write_done" in n)
    assert done == [f"engine-P->engine-D:kv_write_done#dp0.dp{d}" for d in range(4)]
    a, b = (table.lookahead(c) for c in _names(table, TRAFFIC))
    assert a == b == HOPS["admission_s"] + HOPS["router_s"]
    assert table.lookahead(done[0]) == 2.0**-9


def test_each_stage_of_a_pipelined_chunked_prefill_tells_decode_its_writes():
    """Each stage writes its own layers: decode hears from every one."""
    table = prefill_decode_table(
        **HOPS, kv_write_req_s=2.0**-9, dp=(1, 4), prefill_pp=4, chunked=True
    )
    names = {c.name for lp in table.registry.ids() for c in table.channels_from(lp)}
    done = sorted(n for n in names if ":kv_write_done" in n)
    assert done == [
        f"engine-P-pp{s}->engine-D:kv_write_done#dp0.dp{d}"
        for s in range(4)
        for d in range(4)
    ]


def test_a_table_that_is_not_chunked_is_as_before():
    old = prefill_decode_table(**HOPS, kv_write_req_s=2.0**-9, dp=(1, 4))
    names = {c.name for lp in old.registry.ids() for c in old.channels_from(lp)}
    assert "frontend-P->frontend-D:relay" in names
    assert not [n for n in names if "kv_write_done" in n]
    assert _names(old, TRAFFIC) == ["traffic->frontend-P:http"]


def test_a_chunked_table_refuses_a_router_lp():
    with pytest.raises(ValueError, match="router LP's concurrent sends"):
        prefill_decode_table(
            **HOPS, kv_write_req_s=2.0**-9, router_in_s=2.0**-13, chunked=True
        )


class _Script:
    def send(self, request):
        pass


def test_traffic_stamps_each_request_alike_on_both_frontends():
    try:
        metadata.version("compass-harness")
        from compass_harness.traffic_lp import TrafficLP
    except metadata.PackageNotFoundError as e:
        pytest.skip(f"{e.name} is not installed")
    except RuntimeError as e:  # compass_harness checks aiperf at import
        pytest.skip(str(e))

    table = prefill_decode_table(**HOPS, kv_write_req_s=2.0**-9, chunked=True)
    rt = LPRuntime(TRAFFIC, table, _Script())
    rt.start_run()
    lp = SimpleNamespace(
        rt=rt, http="traffic->frontend-P:http", also=("traffic->frontend-D:http",)
    )
    stamps = [TrafficLP._stamp(lp) for _ in range(3)]
    assert stamps == [(HOPS["admission_s"] + HOPS["router_s"], k) for k in range(3)]
    sent = {}
    for ch, seq, arrival in rt.send_log:
        sent.setdefault(ch, []).append((arrival, seq))
    assert sent["traffic->frontend-P:http"] == sent["traffic->frontend-D:http"]


@pytest.mark.parametrize("connector", ["compass", "mooncake"])
def test_a_chunked_producer_advertises_it_the_simulated_one_at_no_host(
    monkeypatch, connector
):
    sub = {
        "kv_connector": connector,
        "kv_role": "kv_producer",
        "enable_chunked_transfer": True,
        "handshake_port": 6401,
    }
    config = SimpleNamespace(
        kv_transfer_config=sub,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        kv_cache_block_size=128,
        decode_context_parallel_size=1,
    )
    monkeypatch.setattr(api_server, "engine", SimpleNamespace(config=config))
    monkeypatch.setattr("atom.utils.network.get_ip", lambda: "127.0.0.1")
    info = asyncio.run(api_server.kv_transfer_info())
    host = "compass-simulated.invalid" if connector == "compass" else "127.0.0.1"
    assert info["chunked_transfer"] == {
        "remote_host": host,
        "remote_handshake_port": 6401,
        "remote_pp_size": 4,
        "block_size": 128,
        "dcp_size": 1,
    }
