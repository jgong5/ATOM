# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs atomesh's unit and Compass cargo tests offline, or refuses
when a diff touching atom/mesh would leave them skipped.

Each case runs the real script over the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` plus an ``atom/mesh/Cargo.toml``, with
a stand-in ``cargo`` first on PATH that logs its environment and arguments and
exits as told. Nothing needs cargo or a network. The last case's stand-in logs
when each of its runs starts and ends instead.
"""

from test_gate_cpu_verdict_through_pipe import BLIND, _run, _tree

CARGO = """#!/bin/sh
echo "$CARGO_NET_OFFLINE $*" >> "$0.log"
case "$1" in metadata) exit {metadata};; test) exit {test};; esac
"""
# Upstream's unit tests, and every Compass target by its file name alone, less
# the tests that fail on a loaded host (#515, #514).
TARGETS = (
    "--lib --test compass_a_test --test compass_b"
    " -- --exact --skip core::worker::tests::test_load_counter_performance"
    " --skip policies::tree::tests::test_tree_structure_integrity_after_stress"
)


def _gate(tmp_path, changed, metadata=0, test=0):
    root = _tree(tmp_path / "ATOM", stamped=True)
    (root / ".compass-changed").write_text(changed + "\n")
    (root / "atom" / "mesh" / "tests").mkdir(parents=True)
    (root / "atom" / "mesh" / "Cargo.toml").write_text("")
    for name in ("compass_b.rs", "compass_a_test.rs", "routing_tests.rs"):
        (root / "atom" / "mesh" / "tests" / name).write_text("")
    (tmp_path / "Cargo.lock").write_text("# image lock\n")
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir()
    cargo.write_text(CARGO.format(metadata=metadata, test=test))
    cargo.chmod(0o755)
    out = _run(
        root,
        f"COMPASS_CARGO_LOCK={tmp_path}/Cargo.lock PATH={cargo.parent}:$PATH {{gate}}",
    )
    log = cargo.with_suffix(".log")
    log = log.read_text().splitlines() if log.exists() else []
    return out, out.stdout.splitlines()[-1], log, root


def test_a_mesh_diff_whose_cargo_tests_would_skip_is_refused(tmp_path):
    out, verdict, log, _ = _gate(tmp_path, "atom/mesh/src/lib.rs", metadata=101)
    assert out.returncode == 96, out.stdout + out.stderr
    assert "touches atom/mesh and its cargo tests would skip" in verdict, verdict
    assert "pytest: rc=" not in out.stdout, out.stdout
    assert log == ["true metadata --format-version 1"], log


def test_an_unchanged_mesh_that_cannot_resolve_is_reported_not_refused(tmp_path):
    out, _, log, _ = _gate(tmp_path, BLIND, metadata=101)
    # 98 is the GPU-gate answer the stamped BLIND diff ends on, after pytest.
    assert out.returncode == 98, out.stdout + out.stderr
    assert "mesh:   cargo tests skipped, atom/mesh unchanged" in out.stdout
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert len(log) == 1, log


def test_the_compass_cargo_tests_run_offline_under_the_image_lock(tmp_path):
    out, _, log, root = _gate(tmp_path, "atom/mesh/src/lib.rs")
    assert out.returncode == 0, out.stdout + out.stderr
    assert log == ["true metadata --format-version 1", f"true test {TARGETS}"], log
    assert (root / "atom" / "mesh" / "Cargo.lock").read_text() == "# image lock\n"


def test_a_failing_cargo_test_fails_the_gate_after_pytest(tmp_path):
    out, verdict, _, _ = _gate(tmp_path, "atom/mesh/src/lib.rs", test=101)
    assert out.returncode == 101, out.stdout + out.stderr
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert "cargo test failed in atom/mesh" in verdict, verdict


TIMED = """#!/bin/sh
echo "$(date +%s%N) start $$ $(pwd -P) $*" >> "$0.log"
[ "$1" = build ] && sleep 1
echo "$(date +%s%N) end $$" >> "$0.log"
"""
BUILD = """import subprocess
def test_{name}():
    subprocess.run(["cargo", "build", "{name}"], cwd={cwd!r}, check=True)
"""


def test_two_shards_never_run_cargo_in_atom_mesh_at_once(tmp_path):
    # test_a.py and test_b.py start in two shards at once and build in
    # atom/mesh; test_c.py, after test_a.py in the first shard, builds elsewhere.
    root = _tree(tmp_path / "ATOM", stamped=True)
    mesh = root / "atom" / "mesh"
    mesh.mkdir(parents=True)
    (mesh / "Cargo.toml").write_text("")
    (root / "tests" / "compass").mkdir()
    for name, cwd in (("a", mesh), ("b", mesh), ("c", tmp_path)):
        test = BUILD.format(name=name, cwd=str(cwd))
        (root / "tests" / "compass" / f"test_{name}.py").write_text(test)
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir()
    cargo.write_text(TIMED)
    cargo.chmod(0o755)
    out = _run(root, f"COMPASS_GATE_SHARDS=2 PATH={cargo.parent}:$PATH {{gate}}")
    assert "pytest: rc=0" in out.stdout, out.stdout + out.stderr
    starts, ends = {}, {}
    for line in cargo.with_suffix(".log").read_text().splitlines():
        at, event, pid, *rest = line.split()
        if event == "start":
            starts[pid] = (int(at), rest[0], " ".join(rest[1:]))
        else:
            ends[pid] = int(at)
    builds = {
        args: (start, ends[pid], cwd)
        for pid, (start, cwd, args) in starts.items()
        if args.startswith("build")
    }
    assert sorted(builds) == ["build a", "build b", "build c"], builds
    assert builds["build c"][2] == str(tmp_path), builds
    first, second = sorted([builds["build a"], builds["build b"]])
    assert first[1] <= second[0], f"two shards built in atom/mesh at once: {builds}"
