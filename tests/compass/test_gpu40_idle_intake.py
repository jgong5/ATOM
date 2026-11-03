# SPDX-License-Identifier: MIT
"""mi355-gpu-40's spec states the idle intake its lone requests read, and a
simulated run under the law file captured there charges a request that finds no
other intake on the loop that, not the burst's fixed term."""

import json

from test_spec_verbs import SPEC_ROOT, _mi355x_spec

LAW = SPEC_ROOT / "scripts/compass/laws"
GPU40 = LAW / "deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-decode.json"


def test_gpu40_states_the_idle_intake_its_lone_requests_read(monkeypatch):
    script, spec = _mi355x_spec(monkeypatch, "mi355x-gpu40")
    model, _, root, value, band, record = script.IDLE
    origin = spec.origin("host.http_intake.idle_s")
    assert origin["origin"] == "probed"
    assert (origin["model"], origin["source_root"]) == (model, f"commit:{root}")
    assert origin["record"] == f"sha256:{record}"
    assert (origin["p10"], origin["p90"]) == band
    assert spec.value("host.http_intake.idle_s") == value


def test_a_simulated_run_on_gpu40_charges_a_lone_request_the_idle_intake(
    monkeypatch, tmp_path
):
    from atom.compass import run as compass_run

    script, _ = _mi355x_spec(monkeypatch, "mi355x-gpu40")
    _, _, terms, _, _ = script.INTAKE
    run = tmp_path / "run.json"
    host = json.loads(GPU40.read_text())["host"]
    run.write_text(json.dumps({"bound_s": 600.0, "host": host}))
    monkeypatch.setenv(compass_run.ENV, str(run))
    fixed, _, idle = compass_run.http_intake()
    assert fixed == terms["fixed_s"]
    assert idle == script.IDLE[3]
    assert idle < fixed
