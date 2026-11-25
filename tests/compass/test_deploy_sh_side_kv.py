# SPDX-License-Identifier: MIT
"""`deploy.sh`'s per-side kv settings: PREFILL_KV_EXTRA and DECODE_KV_EXTRA
reach their side's kv config, whole in a record, and in a simulated run only
those the simulated connector honours (`SIM_KV_KEYS`: `pd_admission`), the
rest named on stderr. The M3.1 1P1D recipe's decode runs `pd_admission` and
its prefill chunked transfer. The authority and instances are
`test_deploy_sh`'s planted ones."""

import json
import signal
import subprocess

import pytest
from test_deploy_sh import _until, argv, planted  # noqa: F401  a fixture

ADMISSION = {
    "pd_admission": {
        "enabled": True,
        "max_pending_kv_fraction": 0.15,
        "num_reserved_decode_tokens": 512,
    }
}
CHUNKED = {"enable_chunked_transfer": True, "chunked_admission_timeout_s": 1500}


def _kv(d, name):
    return json.loads(json.loads((d / f"{name}.argv.json").read_text())[-1])


@pytest.mark.parametrize("cmd", ["run", "record"])
def test_each_side_s_kv_settings_reach_its_kv_config(
    tmp_path, planted, cmd
):  # noqa: F811
    env, flags = planted
    env |= {
        "PREFILL_KV_EXTRA": json.dumps(CHUNKED),
        "DECODE_KV_EXTRA": json.dumps({**ADMISSION, "ib_device": "rdma4"}),
    }
    words = argv(tmp_path, flags if cmd == "run" else ["--gpu", "0,1"])
    words[3] = cmd
    d = tmp_path / "d"
    script = subprocess.Popen(
        words, env=env, cwd=tmp_path, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        _until(d / "server.json", script)
    finally:
        script.send_signal(signal.SIGTERM)
        _, err = script.communicate(timeout=60)

    p, dk = _kv(d, "prefill"), _kv(d, "decode")
    assert (p["kv_role"], dk["kv_role"]) == ("kv_producer", "kv_consumer")
    assert dk["pd_admission"] == ADMISSION["pd_admission"]
    assert "pd_admission" not in p
    assert {k: p[k] for k in CHUNKED} == CHUNKED
    if cmd == "record":
        assert p["kv_connector"] == dk["kv_connector"] == "mooncake"
        assert dk["ib_device"] == "rdma4"
    else:
        assert p["kv_connector"] == dk["kv_connector"] == "compass"
        assert "ib_device" not in dk
        assert "DECODE_KV_EXTRA leaves ['ib_device'] out" in err
        assert "PREFILL_KV_EXTRA leaves" not in err
        # Chunked: the run file says so, and both ends name one notice address.
        run = json.loads((d / "run_file.json").read_text())
        assert run["kv_chunked"] is True
        assert p["compass_kv_write_done"] == dk["compass_kv_write_done"]


def test_a_side_s_kv_settings_that_are_no_json_object_exit_2(
    tmp_path, planted
):  # noqa: F811
    env, flags = planted
    env["DECODE_KV_EXTRA"] = "[1]"
    done = subprocess.run(
        argv(tmp_path, flags), env=env, cwd=tmp_path, capture_output=True,
        text=True, check=False, timeout=60,
    )  # fmt: skip
    assert done.returncode == 2, done.stderr
    assert "DECODE_KV_EXTRA is no JSON object: [1]" in done.stderr
