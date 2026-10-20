# SPDX-License-Identifier: MIT
"""`deploy.sh OUT record` starts each instance as one `compass record` on
Mooncake, the decode instance through the command prefix its `--on` names,
which stops it there too, and states the KV link its layout spans, as a
simulated deployment's run file does.

The planted pieces are `test_deploy_sh`'s, its `compass` instance standing in
for `compass record`, and a prefix that stands in for `ssh <node>`: it passes
its words through a shell again, and a signal to it does not reach them.
"""

import json
import signal
import subprocess
import time

import pytest
from test_deploy_sh import ROUTER_ARGS, _until, argv, planted  # noqa: F401

ON = """\
#!/bin/sh
node=$1
shift
echo "$node $*" >>"$PLANTED_ON"
trap 'exit 143' TERM
sh -c "$*" &
wait $!
"""


@pytest.fixture
def on(tmp_path, planted):  # noqa: F811
    """The stand-in prefix, which logs each command it runs to on.log."""
    prefix = tmp_path / "bin" / "on"
    prefix.write_text(ON)
    prefix.chmod(0o755)
    planted[0]["PLANTED_ON"] = str(tmp_path / "on.log")
    return f"{prefix} node-b"


def _on(tmp_path) -> list[str]:
    log = tmp_path / "on.log"
    return log.read_text().splitlines() if log.exists() else []


def _record(tmp_path, on, router=ROUTER_ARGS):
    """A real deployment across two nodes, both on GPUs 0 and 1."""
    words = argv(tmp_path, ["--gpu", "0,1"], router=router)
    words[3] = "record"
    at = words.index("--decode") + 1
    words[at:at] = ["--on", on, "--host", "127.0.0.1"]
    return words


def test_a_two_node_record_starts_decode_through_its_prefix_and_stops_it_there(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    d = tmp_path / "d"
    words = _record(tmp_path, on, router=[*ROUTER_ARGS, "--dp-aware"])
    script = subprocess.Popen(words, env=env, cwd=tmp_path, stdout=subprocess.PIPE)
    _until(d / "server.json", script)

    model = ["--model", str(tmp_path)]
    ports = set()
    for side, role, own, atom in (
        ("prefill", "kv_producer", [], [*model, "-tp", "2", "--enable-dp-attention"]),
        ("decode", "kv_consumer", ["--host", "127.0.0.1"], model),
    ):
        seen = json.loads((d / f"{side}.argv.json").read_text())
        out = ["--out", str(d / side), "--"]
        assert seen[:-2] == ["record", "--gpu", "0,1", *own, *out, *atom]
        assert seen[-2] == "--kv-transfer-config"
        kv = json.loads(seen.pop())
        assert kv.pop("kv_role") == role and kv.pop("kv_connector") == "mooncake"
        ports.add(kv.pop("handshake_port"))
        assert not kv
    assert len(ports) == 2
    assert _on(tmp_path) == [f"node-b bash {d}/decode.sh"]
    router = json.loads((tmp_path / "router.json").read_text())
    assert router["argv"][-1] == "--dp-aware"
    assert set(json.loads((d / "server.json").read_text())) == {"url", "pid"}
    assert not (d / "run_file.json").exists()
    assert not (d / "authority.argv.json").exists()

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    pid = int((d / "decode.pid").read_text())
    assert _on(tmp_path)[1:] == [f"node-b kill -TERM {pid}"]
    assert all((d / side / "run.json").exists() for side in ("prefill", "decode"))
    written = json.loads((d / "run.json").read_text())
    assert written["kv_link"] == "inter_node" and "run_file" not in written


@pytest.mark.parametrize("node_b, link", [(True, "inter_node"), (False, "intra_node")])
def test_a_simulated_deployment_prices_kv_on_the_link_its_layout_spans(
    tmp_path, planted, on, node_b, link  # noqa: F811
):
    env, flags = planted
    d = tmp_path / "d"
    words = argv(tmp_path, flags)
    if node_b:
        at = words.index("--decode") + 1
        words[at:at] = ["--on", on]
    script = subprocess.Popen(words, env=env, cwd=tmp_path, stdout=subprocess.PIPE)
    _until(d / "server.json", script)

    assert json.loads((d / "run_file.json").read_text())["kv_link"] == link
    assert _on(tmp_path) == ([f"node-b bash {d}/decode.sh"] if node_b else [])

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    assert json.loads((d / "run.json").read_text())["kv_link"] == link


def test_a_deployment_that_fails_stops_the_decode_through_its_prefix(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    (tmp_path / "bin" / "atomesh").write_text("#!/bin/sh\nexit 3\n")
    d = tmp_path / "d"

    done = subprocess.run(
        _record(tmp_path, on),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 1
    assert "deploy: router exited before it answered" in done.stderr
    pid = int((d / "decode.pid").read_text())
    assert _on(tmp_path)[1:] == [f"node-b kill -TERM {pid}"]
    for _ in range(100):
        if (d / "decode" / "run.json").exists():
            break
        time.sleep(0.1)
    assert (d / "decode" / "run.json").exists()
