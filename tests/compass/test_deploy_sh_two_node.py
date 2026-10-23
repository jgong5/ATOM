# SPDX-License-Identifier: MIT
"""`deploy.sh OUT record` starts each instance as one `compass record` on
Mooncake, the decode instance through the command prefix its `--on` names,
which stops it there too, and states the KV link its layout spans, as a
simulated deployment's run file does.

The planted pieces are `test_deploy_sh`'s, its `compass` instance standing in
for `compass record`, and a prefix that stands in for `ssh <node>`: it passes
its words through a shell again, and a signal to it does not reach them. It
can run its command late, whether or not it is still there
(`PLANTED_ON_DELAY`), or hang instead of running the start or the stop
(`PLANTED_ON_HANG`).
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from test_deploy_sh import ROUTER_ARGS, _until, argv, planted  # noqa: F401

ON = """\
#!/bin/sh
node=$1
shift
echo "$node $*" >>"$PLANTED_ON"
trap 'exit 143' TERM
case "$1:${PLANTED_ON_HANG-}" in
bash:start | kill:kill) sleep 100 & wait $! ;;
esac
sh -c "sleep ${PLANTED_ON_DELAY:-0}; $*" &
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


def _gone(pid: int) -> bool:
    """Whether process `pid` exits within 15 s; a zombie has."""
    for _ in range(150):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.1)
    return False


def _run(words, env, cwd):
    return subprocess.run(
        words,
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


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
        # Only the prefill binds ports from its handshake port; the decode's
        # Mooncake listeners take ports the kernel hands out.
        if side == "prefill":
            kv.pop("handshake_port")
        assert not kv
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


def test_on_as_the_last_word_exits_2_naming_it_before_anything_starts(
    tmp_path, planted  # noqa: F811
):
    env, _ = planted
    words = _record(tmp_path, "unused")
    del words[words.index("--on") + 1 :]

    done = subprocess.run(
        words,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert done.returncode == 2, done.stderr
    assert "deploy: --on takes a command prefix" in done.stderr
    assert not (tmp_path / "d").exists()


def test_with_on_an_out_a_shell_would_split_exits_2_before_anything_starts(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    words = _record(tmp_path, on)
    words[2] = str(tmp_path / "a b" / "d")

    done = _run(words, env, tmp_path)

    assert done.returncode == 2, done.stderr
    assert "--on's prefix may parse" in done.stderr
    assert not (tmp_path / "a b").exists()


def test_a_stop_before_the_decode_wrote_its_pid_still_reaches_it_there(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    env["PLANTED_ON_DELAY"] = "3"
    d = tmp_path / "d"
    script = subprocess.Popen(
        _record(tmp_path, on), env=env, cwd=tmp_path, stdout=subprocess.PIPE
    )
    _until(d / "decode.sh", script)
    assert not (d / "decode.pid").exists()

    script.send_signal(signal.SIGTERM)
    script.communicate(timeout=60)

    for _ in range(100):  # the prefix runs the command late
        if (d / "decode.pid").exists():
            break
        time.sleep(0.1)
    pid = int((d / "decode.pid").read_text())
    assert _on(tmp_path) == [f"node-b bash {d}/decode.sh", f"node-b kill -TERM {pid}"]
    assert _gone(pid)


def test_a_prefix_that_never_starts_the_decode_fails_in_prefix_s(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    env |= {"PLANTED_ON_HANG": "start", "PREFIX_S": "2", "WAIT_S": "30"}
    start = time.monotonic()

    done = _run(_record(tmp_path, on), env, tmp_path)

    assert done.returncode == 1
    assert "deploy: the decode instance wrote no process id in 2 s" in done.stderr
    assert time.monotonic() - start < 20


def test_a_decode_that_exits_at_once_fails_the_deployment_before_the_prefill_answers(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    env["PLANTED_FAIL"] = "decode"
    # The prefill instance's command starts 30 s late.
    bindir = tmp_path / "bin"
    (bindir / "python3").rename(bindir / "python3-planted")
    (bindir / "python3").write_text(
        '#!/bin/sh\ncase "$*" in *"/prefill --"*) sleep 30 ;; esac\n'
        'exec "$(dirname "$0")/python3-planted" "$@"\n'
    )
    (bindir / "python3").chmod(0o755)
    start = time.monotonic()

    done = _run(_record(tmp_path, on), env, tmp_path)

    assert done.returncode == 1
    assert "deploy: the decode instance exited before it answered" in done.stderr
    assert time.monotonic() - start < 20


def test_a_stop_whose_prefix_hangs_gives_up_after_prefix_s(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    env |= {"PLANTED_ON_HANG": "kill", "PREFIX_S": "2", "WAIT_S": "5"}
    d = tmp_path / "d"
    script = subprocess.Popen(
        _record(tmp_path, on),
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _until(d / "server.json", script)

    script.send_signal(signal.SIGTERM)
    try:
        _, err = script.communicate(timeout=60)
    finally:
        # The decode instance the stop never reached.
        os.kill(int((d / "decode.pid").read_text()), signal.SIGKILL)

    assert script.returncode == 1
    assert "killed after 5 s" in err


def test_a_stop_as_server_json_appears_stops_the_deployment_whole(
    tmp_path, planted  # noqa: F811
):
    env, flags = planted
    # `mv` holds 2 s once it has put server.json in place.
    mv = tmp_path / "bin" / "mv"
    mv.write_text(
        '#!/bin/sh\n/bin/mv "$@" || exit\ncase "$2" in */server.json) sleep 2 ;; esac\n'
    )
    mv.chmod(0o755)
    d = tmp_path / "d"
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE
    )
    _until(d / "server.json", script)

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    assert (d / "run.json").exists()


def test_the_router_reaches_each_instance_whatever_proxy_the_environment_names(
    tmp_path, planted, on  # noqa: F811
):
    env, _ = planted
    for name in ("NO_PROXY", "no_proxy"):
        env.pop(name, None)
    env |= dict.fromkeys(("HTTP_PROXY", "http_proxy"), "http://127.0.0.1:1")
    words = _record(tmp_path, on)
    # The decode on a host of its own, as on another node.
    words[words.index("--host") + 1] = "127.0.0.2"
    script = subprocess.Popen(words, env=env, cwd=tmp_path, stdout=subprocess.PIPE)
    _until(tmp_path / "d" / "server.json", script)

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out


def test_every_port_the_deployment_holds_free_is_one_no_bind_to_port_0_takes(
    tmp_path, planted, on  # noqa: F811
):
    """ATOM's ranks bind ports the kernel hands out while they load, before
    Mooncake binds its handshake ports."""
    env, _ = planted
    d = tmp_path / "d"
    script = subprocess.Popen(
        _record(tmp_path, on), env=env, cwd=tmp_path, stdout=subprocess.PIPE
    )
    _until(d / "server.json", script)
    script.send_signal(signal.SIGTERM)
    script.communicate(timeout=60)

    router = json.loads((tmp_path / "router.json").read_text())["argv"]
    ends = [
        int(router[router.index(f) + 1]) + 1 for f in ("--port", "--prometheus-port")
    ]
    for side, ranks in (("prefill", 2),):
        kv = json.loads(json.loads((d / f"{side}.argv.json").read_text())[-1])
        ends.append(kv["handshake_port"] + ranks)
    ephemeral = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0]
    assert max(ends) <= int(ephemeral), ends


@pytest.fixture
def port_range(tmp_path, planted):  # noqa: F811
    """A file that stands in for `ip_local_port_range` in the ports script."""
    bindir, path = tmp_path / "bin", tmp_path / "ip_local_port_range"
    (bindir / "python3").rename(bindir / "python3-planted")
    code = "import sys; from atom.compass.drive import serve; "
    code += "serve.PORT_RANGE = serve.Path(sys.argv.pop(1)); exec(sys.stdin.read())"
    (bindir / "python3").write_text(
        '#!/bin/sh\nreal="$(dirname "$0")/python3-planted"\n'
        '[ "$1" = - ] || exec "$real" "$@"\nshift\n'
        f'exec "$real" -c \'{code}\' {path} "$@"\n'
    )
    (bindir / "python3").chmod(0o755)
    return path


def test_a_block_with_no_room_below_the_ephemeral_range_exits_1_naming_both(
    tmp_path, planted, on, port_range  # noqa: F811
):
    env, _ = planted
    port_range.write_text("17004\t60999\n")
    words = _record(tmp_path, on)
    words[words.index("-tp") + 1] = "4"

    done = _run(words, env, tmp_path)

    assert done.returncode == 1, done.stderr
    assert "deploy: no 4 free ports in a row from 17001 below 17004" in done.stderr
    assert "Traceback" not in done.stderr
    assert not (tmp_path / "d" / "prefill.sh").exists()


def test_every_port_the_deployment_draws_is_above_mooncake_s_rpc_ports(
    tmp_path, planted, on, port_range  # noqa: F811
):
    """Mooncake draws each rank's RPC port from 15000 to 17000, and each
    instance's server its ports below 15000."""
    env, _ = planted
    port_range.write_text("17101\t60999\n")
    d = tmp_path / "d"
    script = subprocess.Popen(
        _record(tmp_path, on), env=env, cwd=tmp_path, stdout=subprocess.PIPE
    )
    _until(d / "server.json", script)
    script.send_signal(signal.SIGTERM)
    script.communicate(timeout=60)

    router = json.loads((tmp_path / "router.json").read_text())["argv"]
    ports = [int(router[router.index(f) + 1]) for f in ("--port", "--prometheus-port")]
    kv = json.loads(json.loads((d / "prefill.argv.json").read_text())[-1])
    ports += [kv["handshake_port"], kv["handshake_port"] + 1]
    assert all(17000 < p < 17101 for p in ports), ports
