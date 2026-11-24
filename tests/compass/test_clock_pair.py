# SPDX-License-Identifier: MIT
"""`clock_pair` pairs another node's CLOCK_MONOTONIC with this one's over the
command prefix that reaches it, and reads the pairs back as an offset at an
instant of that node's clock.

The other node is this one, reached through a prefix that, as `ssh <node>`
does, runs its words in a shell of its own: its clock is this one's, so each
pair's offset is none, within the error it states.
"""

import itertools
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from atom.compass.drive import clock_pair

#: A prefix as `ssh <node>`: its first word the node, the rest run in a shell.
ON = '#!/bin/sh\nshift\necho "$*" >>"$0.log"\nexec sh -c "$*"\n'


@pytest.fixture
def on(tmp_path):
    """The prefix, its log of the commands it ran beside it."""
    prefix = tmp_path / "on"
    prefix.write_text(ON)
    prefix.chmod(0o755)
    return prefix


def _client(pairs, t):
    """How far that node's clock runs ahead of this one's at `t` on this one's,
    linear in it between `pairs`, each (this one's clock, offset), and the
    nearest's outside them: c2verify's c2common.Clock, which reads it so."""
    if t <= pairs[0][0]:
        return pairs[0][1]
    for (t0, a), (t1, b) in itertools.pairwise(pairs):
        if t <= t1:
            return a + (t - t0) / (t1 - t0) * (b - a)
    return pairs[-1][1]


def test_an_offset_linear_in_this_node_s_clock_is_linear_in_that_one_s():
    """The offset at an instant of that node's clock is the one at the
    instant of this one's it is: the same line between the same two pairs."""
    pairs = [(100.0, 1.056), (160.0, 1.058), (220.0, 1.0575), (280.0, 1.0601)]
    rows = [(round(t * 1e9), round(o * 1e9), 20_000) for t, o in pairs]
    offsets = clock_pair.Offsets(rows)

    for k in range(400):
        here = 90.0 + 0.5 * k
        offset = _client(pairs, here)
        got, error = offsets.at(here + offset)
        assert got == pytest.approx(offset, abs=1e-12)
        assert error == pytest.approx(2e-5)


def test_each_instant_takes_the_larger_error_of_the_pairs_either_side():
    offsets = clock_pair.Offsets([(10**9, 0, 5_000), (3 * 10**9, 0, 30_000)])

    assert offsets.at(0.5) == pytest.approx((0.0, 5e-6))
    assert offsets.at(2.0) == pytest.approx((0.0, 3e-5))
    assert offsets.at(4.0) == pytest.approx((0.0, 3e-5))


def test_of_two_pairs_out_of_order_on_that_node_s_clock_the_smaller_error_stays():
    """Two pairs a microsecond apart whose offsets differ by more, as only
    their errors can make them: the one of smaller error is kept."""
    rows = [(10**9, 3_000, 40_000), (10**9 + 1_000, 0, 20_000), (2 * 10**9, 0, 9)]

    offsets = clock_pair.Offsets(rows)

    assert offsets.rows == [(10**9 + 1_000, 0, 20_000), (2 * 10**9, 0, 9)]


def test_a_file_of_pairs_reads_as_each_node_s_offsets_and_refuses_a_line_of_none(
    tmp_path,
):
    path = tmp_path / clock_pair.FILE
    rows = [
        {"node": "10.0.0.2", "mono_ns": 2 * 10**9, "offset_ns": 7, "error_ns": 1},
        {"node": "10.0.0.3", "mono_ns": 10**9, "offset_ns": 5, "error_ns": 1},
        {"node": "10.0.0.2", "mono_ns": 10**9, "offset_ns": 3, "error_ns": 1},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    read = clock_pair.read(path)

    assert list(read) == ["10.0.0.2", "10.0.0.3"]
    assert read["10.0.0.2"].rows == [(10**9, 3, 1), (2 * 10**9, 7, 1)]
    with open(path, "a") as more:
        more.write('{"node": "10.0.0.2", "mono_ns": 1}\n')
    with pytest.raises(ValueError, match=f"{path} line 4 is no clock pair"):
        clock_pair.read(path)


def test_a_pair_over_a_prefix_runs_this_file_there_and_states_its_error(on):
    """The prefix reaches this node, so the offset is none, within the error
    the fastest tenth of the exchanges state."""
    before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    got = clock_pair.pair([str(on), "node-b"], rounds=100)
    after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    assert list(got) == ["mono_ns", "offset_ns", "error_ns"]
    assert before < got["mono_ns"] < after
    assert 0 < got["error_ns"] < after - before
    assert abs(got["offset_ns"]) <= got["error_ns"]
    ran = on.with_name("on.log").read_text().splitlines()
    assert ran == [f"python3 -I -u {os.path.abspath(clock_pair.__file__)} serve"]


def _main(tmp_path, *words, env=None):
    """The command's run on `words` after its file, under `env` or this
    process's environment, and the pairs it wrote."""
    out = tmp_path / clock_pair.FILE
    command = [sys.executable, "-m", "atom.compass.drive.clock_pair", str(out)]
    done = subprocess.run(
        [*command, *words],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    rows = out.read_text().splitlines() if out.exists() else []
    return done, [json.loads(r) for r in rows]


def test_each_other_node_is_paired_once_by_the_first_prefix_naming_it(tmp_path, on):
    """Two instances on 10.0.0.2, one on node-c by name, one this node's
    loopback and one its unspecified address: a pair each of the two others."""
    words = ["--"]
    for url, node in (
        ("http://10.0.0.2:8000", "b"),
        ("http://10.0.0.2:8001", "b2"),
        ("http://127.0.0.1:8002", "here"),
        ("http://0.0.0.0:8003", "here"),
        ("http://node-c:8004", "c"),
    ):
        words += [url, "2", str(on), f"node-{node}"]

    done, rows = _main(tmp_path, *words)

    assert done.returncode == 0, done.stderr
    assert [r["node"] for r in rows] == ["10.0.0.2", "node-c"]
    assert all(abs(r["offset_ns"]) <= r["error_ns"] for r in rows)
    assert len(on.with_name("on.log").read_text().splitlines()) == 2


def test_a_node_s_override_is_named_by_its_address_in_letters_digits_and_underscores():
    assert clock_pair.override("10.19.0.127") == "CLOCK_PAIR_PREFIX_10_19_0_127"
    assert clock_pair.override("fd00::7") == "CLOCK_PAIR_PREFIX_fd00__7"
    assert clock_pair.override("pit2-p03-g27") == "CLOCK_PAIR_PREFIX_pit2_p03_g27"


def test_a_node_whose_override_is_set_is_paired_through_it_in_place_of_its_prefix(
    tmp_path, on
):
    """10.0.0.2's instance came up through `false`, which reaches nothing,
    and its override names the prefix, which pairs it; 10.0.0.3's override is
    blank, so its instance's own prefix pairs it."""
    other = tmp_path / "other"
    other.write_text(ON)
    other.chmod(0o755)
    env = os.environ | {
        clock_pair.override("10.0.0.2"): f"{on}  node-b ",
        clock_pair.override("10.0.0.3"): "  ",
    }
    words = ["--", "http://10.0.0.2:8000", "1", "false"]
    words += ["http://10.0.0.3:8000", "2", str(other), "node-c"]

    done, rows = _main(tmp_path, *words, env=env)

    assert done.returncode == 0, done.stderr
    assert [r["node"] for r in rows] == ["10.0.0.2", "10.0.0.3"]
    serve = f"python3 -I -u {os.path.abspath(clock_pair.__file__)} serve"
    for prefix in (on, other):
        assert prefix.with_name(f"{prefix.name}.log").read_text() == serve + "\n"


def test_a_node_its_prefix_cannot_reach_writes_no_pair_and_fails_naming_it(
    tmp_path, on
):
    words = ["--", "http://10.0.0.3:8000", "1", "false"]
    words += ["http://10.0.0.2:8000", "2", str(on), "node-b"]

    done, rows = _main(tmp_path, "--timeout", "10", *words)

    assert done.returncode == 1
    assert done.stderr == "clock_pair: 10.0.0.3: `false` closed with no answer\n"
    assert [r["node"] for r in rows] == ["10.0.0.2"]


def test_every_s_seconds_from_its_start_it_pairs_each_node_until_a_sigterm(
    tmp_path, on
):
    out = tmp_path / clock_pair.FILE
    command = [sys.executable, "-m", "atom.compass.drive.clock_pair", str(out)]
    command += ["--every", "3", "--", "http://10.0.0.2:8000", "2", str(on), "b"]
    started = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    watch = subprocess.Popen(command)
    try:
        for _ in range(300):
            if out.exists() and len(out.read_text().splitlines()) >= 2:
                break
            assert watch.poll() is None
            time.sleep(0.1)
    finally:
        watch.send_signal(signal.SIGTERM)
        assert watch.wait(timeout=30) == 0

    rows = [json.loads(r) for r in out.read_text().splitlines()]
    assert len(rows) >= 2 and {r["node"] for r in rows} == {"10.0.0.2"}
    # The first at once, the next 3 s after it.
    assert rows[0]["mono_ns"] - started < 3e9
    assert rows[1]["mono_ns"] - rows[0]["mono_ns"] >= 3e9
