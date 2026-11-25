# SPDX-License-Identifier: MIT
"""An `op_graph` states the files its trace ran and read, and answers while
each reads as it did.

A graph whose capture had imported a module of a root and read a data file
of it answers a run's lookup and `measure ops` while both read as they did:
an edit to a file of the root it never imported nor read keeps it; an edit to
the module or to the data file misses, and the run traces the shape again and
publishes its graph beside the old one. A reader that traces nothing names
each file that reads otherwise. A real capture states the files it ran and
read under ATOM's and aiter's roots, Compass's own left out.

No device: but for the real capture, the one root is a package this test
writes, and a run's lazy traces take a stand-in tracer, in a thread of this
process, which imports its module and reads its data file.
"""

import functools
import importlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_capture_ir import CONFIG, MAIN, TREE
from test_op_graph_traced_under import config  # noqa: F401  a fixture

from atom.compass import capture
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Kind,
    Reading,
    Topology,
)
from atom.compass.bench import ops
from atom.compass.capture import forward
from atom.compass.capture.forward import decode, structure
from atom.compass.capture.graph import Traced
from atom.compass.ir import Graph, Op

ARCH = "gfx942"
RANK = next(iter(Topology().ranks()))
#: Each file of the root, as the test writes it and as an edit leaves it.
FILES = {
    "ran.py": ("LEAF = 1\n", "LEAF = 2\n"),
    "idle.py": ("LEAF = 1\n", "LEAF = 2\n"),
    "table.json": ('{"rows": 1}', '{"rows": 2}'),
}


def forget():
    """A new process's readings: each file and root is read again."""
    for cached in (capture._roots, capture._place, capture.file_reading):
        cached.cache_clear()
    capture._here.cache_clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The package `watched`, `capture.ROOTS`' one root: `ran.py`, which the
    stand-in tracer imports, `idle.py`, which nothing imports, and
    `table.json`, which it reads."""
    package = tmp_path / "src" / "watched"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name, (text, _) in FILES.items():
        (package / name).write_text(text)
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.setattr(capture, "ROOTS", {"watched": "watched"})
    forget()
    yield package
    for name in ("watched.ran", "watched"):
        sys.modules.pop(name, None)
    forget()


def tracer(config, leaf, package):  # noqa: F811
    """A stand-in for `forward.Tracer` that runs as a capture of `config` on
    `ARCH`: it imports `watched.ran`, reads `table.json` and writes a file
    beside it, then gives each decode shape as one graph of the one leaf named
    `leaf`, stating its files as the tracer does."""
    at = structure(config, "decode", True, ARCH)
    model = compass_run._file_digest(config.model, "config.json")
    reads = forward.watch_reads()

    def trace(shapes, unsplit=()):
        importlib.import_module("watched.ran")
        (package / "table.json").read_text()
        (package / "written.txt").write_text("written, not read")
        graph = Graph(Traced(at, str(shapes[0])), Op(leaf, "opaque_leaf", (), ()))
        return [(graph, model, {"arch": ARCH, "files": capture.files_of(reads)})]

    return SimpleNamespace(trace=trace)


def published(store, config, package, leaf="aiter.held"):  # noqa: F811
    """The key `trace ops` publishes the stand-in's decode B=2 graph at, and
    the files it states."""
    [(graph, model, declared)] = tracer(config, leaf, package).trace([decode(2)])
    key = capture.publish(store, graph, model, declared, 1, [], "trace ops")
    return key, declared["files"]


def edit(package, name):
    """`name` as an edit leaves it, read so by a new process."""
    (package / name).write_text(FILES[name][1])
    forget()


def lookup(monkeypatch, store, config, package):  # noqa: F811
    """A run's lookup over `store`, given ATOM's arguments, each lazy trace the
    stand-in's of one leaf `aiter.here`; and its counts."""
    stand_in = tracer(config, "aiter.here", package)
    monkeypatch.setattr(compass_run, "_tracer", lambda *args: (None, stand_in))
    monkeypatch.setattr(compass_run, "_trace_process", lambda: ThreadPoolExecutor(1))
    lazy = {"lazy_traces": 0, "lazy_trace_wall_seconds": 0}
    args = ["--model", config.model]
    return compass_run._op_graphs(str(store.root), config, ARCH, args, lazy), lazy


def test_a_graph_states_each_file_of_its_roots_it_ran_or_read_and_no_other(
    tmp_path, config, root  # noqa: F811
):
    key, files = published(ArtifactStore(tmp_path / "store"), config, root)

    assert files == {
        "watched": {
            name: capture.file_reading("watched", name)
            for name in ("__init__.py", "ran.py", "table.json")
        }
    }
    reading = capture.files_reading(files)
    assert key.values["source_root"] == f"{reading.kind}-{reading.value}"


@pytest.mark.parametrize(
    "edited, answers", [("idle.py", True), ("ran.py", False), ("table.json", False)]
)
def test_a_run_reads_a_graph_while_each_file_it_ran_or_read_reads_as_it_did(
    monkeypatch, tmp_path, config, root, edited, answers  # noqa: F811
):
    store = ArtifactStore(tmp_path / "store")
    held, _ = published(store, config, root)
    edit(root, edited)
    run, lazy = lookup(monkeypatch, store, config, root)

    assert run("decode", 2, 2)["op"] == ("aiter.held" if answers else "aiter.here")
    assert lazy["lazy_traces"] == (not answers)
    here = capture.lookup(store, held.values["structure"], "decode B=2")
    assert (here == held) == answers
    if not answers:  # traced again and published beside it
        assert store.read(held).key == held and store.read(here).key == here
        differ = [k for k in here.values if here.values[k] != held.values[k]]
        assert differ == ["source_root"]


@pytest.mark.parametrize("edited", ["ran.py", "table.json"])
def test_a_reader_that_traces_nothing_names_each_file_that_reads_otherwise(
    tmp_path, config, root, edited  # noqa: F811
):
    store = ArtifactStore(tmp_path / "store")
    published(store, config, root)
    edit(root, edited)
    named = f"differs in files watched/{edited}"

    with pytest.raises(ArtifactRefusal) as bench:
        ops.graph(store, config, decode(2), ARCH)
    assert named in str(bench.value)
    lookup = compass_run._op_graphs(str(store.root), config, ARCH)
    assert named in lookup("decode", 2, 2)


def test_a_graph_keyed_by_its_files_answers_under_its_torch_alone(
    monkeypatch, tmp_path, config, root  # noqa: F811
):
    store = ArtifactStore(tmp_path / "store")
    under = capture.traced_under()
    with monkeypatch.context() as m:
        other = {**under, "torch": Reading.stated("2.0.0+other")}
        m.setattr(capture, "traced_under", lambda: other)
        published(store, config, root, "aiter.other")
    at = structure(config, "decode", True, ARCH)

    assert capture.lookup(store, at, "decode B=2") is None
    key, _ = published(store, config, root)
    assert capture.lookup(store, at, "decode B=2") == key
    assert ops.graph(store, config, decode(2), ARCH)["region"]["op"] == "aiter.held"


def test_a_capture_states_the_files_it_ran_and_read_and_answers_while_they_do(
    tmp_path, monkeypatch
):
    """The 27B traced at two widths in a process of its own, as
    `test_capture_ir` traces it: its second graph, traced after the first was
    published, states ATOM's model runner and a data file of aiter's it read,
    and no Compass file nor ATOM's server, and answers here until one of its
    files reads otherwise."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(tmp_path / "model")]
    argv += ["--width", "2", "--width", "4"]
    done = subprocess.run(
        [sys.executable, "-c", MAIN, *argv, "--store", str(tmp_path / "store")],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert len(tokens) == 2, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store = ArtifactStore(tmp_path / "store")
    entry = store.read(store.key_from_token(Kind.OP_GRAPH, tokens[1]))
    files = json.loads(entry.read_member("files", RANK, "json"))
    data = [path for path in files["aiter"] if not path.endswith(".py")]
    at, shape = entry.key.values["structure"], entry.key.values["shape"]

    reading = capture.files_reading(files)
    assert entry.key.values["source_root"] == f"{reading.kind}-{reading.value}"
    assert "model_engine/model_runner.py" in files["atom"] and data
    assert not [p for p in files["atom"] if p.startswith(("compass/", "entrypoints/"))]
    assert capture.lookup(store, at, shape) == entry.key
    real = capture.file_reading
    for edited, answers in (
        (("atom", "entrypoints/openai_server.py"), True),
        (("atom", "model_engine/model_runner.py"), False),
        (("aiter", data[0]), False),
    ):

        def reading_here(root, path, edited=edited):
            return "0" * 64 if (root, path) == edited else real(root, path)

        monkeypatch.setattr(capture, "file_reading", functools.cache(reading_here))
        capture._here.cache_clear()
        assert (capture.lookup(store, at, shape) == entry.key) == answers, edited
