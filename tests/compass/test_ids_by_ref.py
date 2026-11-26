# SPDX-License-Identifier: MIT
"""A prompt's ids handed from a simulated prefill frontend to decode by
reference (`atom.compass.run.handoff`, the run file's ``ids_by_ref``).

Prefill's response, rendered as its server renders it, carries a reference
in place of the ids and leaves them in a file; ATOM's chat endpoint on
decode's side resolves the same ids from it and removes the file, so a
reference is read once, and a streaming request holds them as the untracked
tuple it holds ids inline as. Every count of a body's bytes sees the ids
inline: the relay service prefill stamps on its response, and decode's
intake charges and turn order behind `HttpChannel` in its run. Ids that are
not non-negative ints within 64 bits go inline, and with no directory
nothing changes. `compass run --ids-by-ref` writes the key, which only a
prefill-decode run takes; its authority makes the run's directory for the
run and removes it after, and each frontend installs it.
"""

import collections
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import starlette.requests
import test_fe_gc_hold as gc_hold
from conftest import atom_config_double
from starlette.responses import JSONResponse
from test_drive_runfile import build
from test_fe_decode_once import RELAY, _in_the_run
from test_fe_gc_hold import IDS, KV, server  # noqa: F401  the fixture
from test_handoff_codec import (
    LONG,
    _client,
    _hand_off,
    _response,
    pristine,  # noqa: F401  the fixture
)
from test_pd_relay_term import stamped
from test_pd_slice import PD
from test_stream_events import TABLE
from test_vertical_slice import _run_file

from atom.compass import carriers, clock_transport
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.run import handoff
from atom.entrypoints.openai import api_server
from atom.utils import clock, compass_loop
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import HttpChannel

JSON = {"content-type": "application/json"}


@pytest.fixture
def by_ref(pristine, monkeypatch, tmp_path):  # noqa: F811
    """Both sides installed, ids by reference through `tmp_path`."""
    monkeypatch.setattr(handoff, "_DIR", None)
    monkeypatch.setattr(handoff, "_READ_BACK", collections.OrderedDict())
    handoff.install("P", str(tmp_path))
    handoff.install("D", str(tmp_path))
    return tmp_path


def _text(ids) -> str:
    return json.dumps(ids, separators=(",", ":"))


def _prefill(ids) -> tuple[bytes, list]:
    """Prefill's response for `ids`, and what it carries in their place."""
    body = JSONResponse(_response(ids)).body
    return body, json.loads(body)[handoff.KEY]


def test_decode_takes_the_ids_prefill_handed_by_reference_and_removes_their_file(
    by_ref,
):
    _, ref = _prefill(LONG)
    assert ref[0] == handoff.REF and len(os.listdir(by_ref)) == 1
    reply = _client().post(
        "/v1/chat/completions", content=_hand_off(_text(ref)), headers=JSON
    )
    assert reply.json() == {"ids": LONG}
    # Their check skipped: the ids read back were taken as they are.
    assert os.listdir(by_ref) == [] and not handoff._READ_BACK


def test_a_reference_is_read_once_and_a_malformed_or_second_one_is_refused(by_ref):
    _, ref = _prefill(LONG)
    client, body = _client(), _hand_off(_text(ref))
    assert client.post("/v1/chat/completions", content=body, headers=JSON).json() == {
        "ids": LONG
    }
    again = client.post("/v1/chat/completions", content=body, headers=JSON)
    assert again.status_code == 400
    with pytest.raises(ValueError, match="read already"):
        handoff._take(json.loads(body), body)
    # Its fields name the file to read and remove: only ints are taken.
    bad = _hand_off(_text([handoff.REF, "../x", 0, 5]))
    with pytest.raises(ValueError, match="malformed"):
        handoff._take(json.loads(bad), bad)
    with pytest.raises(ValueError, match="two hand-off references"):
        handoff.inline_extra(_hand_off(_text(ref), f'"prompt_token_ids":{_text(ref)}'))


def test_a_prefill_response_and_its_relay_service_count_the_ids_inline(
    by_ref, monkeypatch
):
    body, _ = _prefill(LONG)
    monkeypatch.setattr(handoff, "_DIR", None)
    inline, _ = _prefill(LONG)
    monkeypatch.setattr(handoff, "_DIR", str(by_ref))
    assert len(body) < len(inline) // 100
    assert len(body) + handoff.inline_extra(body) == len(inline)
    relay = (1.5e-3, 1e-6, 2e-3)
    service = [
        carriers.relay_carried(stamped(b, relay, 4000)[1])[1] for b in (body, inline)
    ]
    assert service[0] == service[1] > 0


def test_decode_s_intake_and_turn_order_count_a_reference_s_ids_inline(
    by_ref, monkeypatch
):
    """Two hand-offs relayed at one instant, 60,000 ids by reference and
    1,000 inline: counted inline, the first body is the longer, so it takes
    the second turn, and each intake is charged for its bytes inline. Each
    body is decoded once: the app takes the decoding its stamp was read from."""
    intake = (1e-4, 1e-8, 1e-4)
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setattr(compass_run, "http_intake", lambda: intake)
    charged, charge = [], compass_loop.charge
    monkeypatch.setattr(
        compass_loop, "charge", lambda d: charged.append(d) or charge(d)
    )
    lookahead = TABLE.lookahead(RELAY)

    def relayed(texts):
        return [
            carriers.with_relay_stamp(_hand_off(t), lambda k=k: (lookahead, k), None)
            for k, t in enumerate(texts)
        ]

    short = _text(LONG[:1000])
    inline = relayed([_text(LONG), short])
    bodies = relayed([_text(_prefill(LONG)[1]), short])
    decoded = []  # the bytes of each decode

    def sized(decode):
        return lambda b: decoded.append(len(b)) or decode(b)

    decoder = SimpleNamespace(decode=sized(carriers.JSON_DECODER.decode))
    monkeypatch.setattr(carriers, "JSON_DECODER", decoder)
    monkeypatch.setattr(handoff, "JSON_DECODER", decoder)
    monkeypatch.setattr(
        starlette.requests, "json", SimpleNamespace(loads=sized(json.loads))
    )
    replies = _in_the_run(_client().app, bodies)
    assert replies == [(200, {"ids": LONG}), (200, {"ids": LONG[:1000]})]
    fixed, per_byte, _ = intake
    assert charged == [pytest.approx(fixed + per_byte * len(b)) for b in inline[::-1]]
    # Beside a reference's own few bytes, which `inline_extra` decodes.
    assert sorted(n for n in decoded if n > 100) == sorted(map(len, bodies))


@pytest.mark.parametrize("ids", [[5, -1], [True, 2], [1.0, 2], [2**64, 1], []])
def test_ids_other_than_non_negative_64_bit_ints_go_inline_as_before(
    by_ref, monkeypatch, ids
):
    content = {**_response([1]), handoff.KEY: ids}
    body = JSONResponse(content).body
    assert os.listdir(by_ref) == [] and handoff.inline_extra(body) == 0
    monkeypatch.setattr(handoff, "_DIR", None)
    assert JSONResponse(content).body == body


def test_with_no_directory_ids_go_inline_and_decode_refuses_a_reference(
    pristine, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(handoff, "_DIR", None)
    handoff.install("P")
    handoff.install("D")
    body, ids = _prefill(LONG)
    assert ids == LONG and handoff.inline_extra(body) == 0
    ref = _text([handoff.REF, os.getpid(), 0, 100])
    reply = _client().post("/v1/chat/completions", content=_hand_off(ref), headers=JSON)
    assert reply.status_code == 400
    assert "non-negative integers" in reply.json()["refused"]


def test_a_streaming_hand_off_by_reference_holds_its_ids_as_an_untracked_tuple(
    by_ref, server, monkeypatch  # noqa: F811
):
    """ATOM's chat route behind frontend-D's `HttpChannel`, as
    `test_fe_gc_hold` drives it with the ids inline: the request holds the
    ids read back as a tuple the collector stops tracking, and the engine is
    handed the same prompt, a list, and the same metadata."""
    body = json.loads(gc_hold.BODY)
    body["kv_transfer_params"][handoff.KEY] = _prefill(IDS)[1]
    monkeypatch.setattr(gc_hold, "BODY", _text(body).encode())
    rt = LPRuntime(LpId("frontend-D"), TABLE, None)
    monkeypatch.setattr(clock, "_installed", rt)
    sent, held = gc_hold._stream_held(
        server, HttpChannel(api_server.app, lambda scope: None)
    )
    assert sent[0][:2] == ("http.response.start", 200)
    assert held == {"kind": tuple, "same": True, "tracked": 0}
    assert not handoff._READ_BACK and os.listdir(by_ref) == []
    ((prompt, kv),) = server.engine
    assert type(prompt) is list and prompt == IDS and kv == KV


@pytest.mark.parametrize("flags, on", [((), False), (("--ids-by-ref",), True)])
def test_ids_go_by_reference_only_where_the_run_command_says(tmp_path, flags, on):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    run = build(tmp_path, "--step-stub", str(stub), *flags)
    assert run[compass_run.IDS_BY_REF] is on
    off = dict(run, **{compass_run.IDS_BY_REF: not on})
    assert compass_run.configuration(run) == compass_run.configuration(off)


@pytest.mark.parametrize("keys", [{"ids_by_ref": "yes"}, {"ids_by_ref": True}])
def test_a_run_refuses_ids_by_reference_outside_a_prefill_decode_run(
    tmp_path, monkeypatch, keys
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, **keys)))
    with pytest.raises(ValueError, match="ids_by_ref"):
        compass_run.spec()


def test_the_run_s_authority_makes_its_ids_directory_and_removes_it_after(
    tmp_path, monkeypatch
):
    """Made empty before the authority serves, a killed run's file gone, and
    removed once it has written the summary, with the file of a hand-off the
    router answered before its relay."""
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, **PD, ids_by_ref=True)))
    path = Path(compass_run.ids_dir(compass_run.spec()))
    path.mkdir(exist_ok=True)
    (path / "1_0").write_bytes(b"")
    seen = []

    def summary(run, authority, ended):
        (path / "2_0").write_bytes(b"")
        seen.append(ended)

    monkeypatch.setattr(
        compass_run, "serve", lambda a, e: seen.append(os.listdir(path))
    )
    monkeypatch.setattr(compass_run, "_until_end", lambda run, a: False)
    monkeypatch.setattr(compass_run, "_write_summary", summary)
    try:
        compass_run.authority("tcp://127.0.0.1:1")
        assert seen == [[], False] and not path.exists()
    finally:
        compass_run.shutil.rmtree(path, ignore_errors=True)


def test_a_decode_frontend_s_bootstrap_installs_the_run_s_ids_directory(
    tmp_path, monkeypatch, pristine  # noqa: F811
):
    run_file = _run_file(tmp_path, **PD, ids_by_ref=True)
    run = json.loads(run_file.read_text())
    authority = clock_transport.serve(
        ClockAuthority(compass_run.channel_table(run)), "tcp://127.0.0.1:0"
    )
    monkeypatch.setenv(compass_run.ENV, str(run_file))
    monkeypatch.setenv(compass_run.CLOCK_ENV, authority.endpoint)
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_authority", None)
    monkeypatch.setattr(handoff, "_DIR", None)
    config = atom_config_double(
        kv_transfer_config={"kv_connector": "compass", "kv_role": "kv_consumer"},
        runner_qualname=compass_run.ATOM_RUNNER,
    )
    try:
        with compass_run._owning_ids_dir(run):
            with compass_run.frontend(config):
                pass
            assert handoff._DIR == compass_run.ids_dir(run)
    finally:
        authority.close()
    # A frontend away from the host its authority made the directory on.
    with pytest.raises(ValueError, match="ids directory .* is missing"):
        handoff.install("D", compass_run.ids_dir(run))
