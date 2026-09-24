# SPDX-License-Identifier: MIT
"""The names a worker dispatches, and the reply each caller is waiting for.

`AsyncIOProc.busy_loop` resolves every RPC with `getattr(runner, name, None)`
and forwards the result only `if out is not None`. Both halves fail quietly:

* a name the runner does not have is skipped, the worker stays healthy, and a
  `wait_out=True` caller blocks on an unbounded `outputs_queue.get()` for the
  rest of the process's life;
* a method that answers None does the same thing, one step later.

Raising is the loud path, and the tests below pin the chain that makes it loud:
the worker dies, the manager's process monitor turns that into a `SystemExit`
on the output queue, and `call_func` re-raises it in the caller. What crosses
that boundary is the type and nothing else -- the `SystemExit` is constructed
with no arguments -- so the rule this file enforces is not "return something
sensible" but **return what the call site unpacks**, checked against ATOM's
source, never against a list anyone typed here.

Nothing here imports `atom.model_engine.model_runner`: that import runs aiter's
architecture probe, which shells out to `rocminfo` and raises where there is no
GPU. The composed class is therefore read from source, the way the sibling
module's tests read it.
"""

import ast
import collections
import pathlib
import re
from importlib.util import resolve_name
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from atom.compass.runner.overrides import (
    RPC_SURFACE,
    NonAllocatingRunner,
    RunnerRefusal,
    unanswered_rpc_names,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
ENGINE = REPO / "atom" / "model_engine"
ATOM_RUNNER = ENGINE / "model_runner.py"
ASYNC_PROC = (ENGINE / "async_proc.py").read_text()
PACKAGE = REPO / "atom" / "compass" / "runner"
PACKAGE_DOC = ast.get_docstring(ast.parse((PACKAGE / "__init__.py").read_text()))
UNWAITED = {name for name, waits in RPC_SURFACE.items() if not waits}
COMPOSED = (REPO / "atom/compass/runner/model_runner.py").read_text()
BROADCAST = ("call_func", "call_func_with_aggregation")
# Every class in the tree that answers a dispatched name and is not in
# `model_runner.py`. The leftover names have to land on one of these; a name
# that lands on none of them is a park with no owner.
ROLLOUT = (
    "atom/rollout/model_runner_ext.py",
    "atom/rollout/weight_updater.py",
    "atom/rollout/memory_manager.py",
)
# The two trees the profiler reply crosses: `model_engine`, which builds the
# dict and forwards it, and the Compass package that replaces the producer.
# The `trace_dir` claim below is about these and nothing else -- reading the
# whole `atom/` tree would let any package's passing mention of the string
# fail a test about this one reply.
REPLY_SURFACE = (ENGINE, REPO / "atom" / "compass" / "runner")


class Site(NamedTuple):
    """One broadcast of one name, and what the caller does with the reply."""

    file: str
    line: int
    waits: bool
    aggregated: bool
    arity: int  # values unpacked from the reply; 0 when it is discarded


def _classes(path):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _methods(node):
    return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}


def _busy_loop():
    """`AsyncIOProc.busy_loop`, parsed."""
    return next(
        n
        for n in ast.walk(ast.parse(ASYNC_PROC))
        if isinstance(n, ast.FunctionDef) and n.name == "busy_loop"
    )


def _refusal_comment():
    """The comment block the composed module's refusal is written inside.

    Comments are not AST nodes, so this is source text: every indented `#`
    line of `model_runner.py`, joined into one string. The module's only
    indented comment is that block; the two unindented ones are the SPDX
    header, which is why the column is enough to select it.
    """
    lines = [
        line.strip().lstrip("#").strip()
        for line in COMPOSED.splitlines()
        if line.strip().startswith("#") and not line.startswith("#")
    ]
    return " ".join(line for line in lines if line)


def _raised_name(node):
    """The name of the exception a call constructs, if it constructs one."""
    return getattr(getattr(node, "func", None), "id", None)


def _arity(parent):
    """How many values the caller takes off the reply, from its parent node.

    `0` means the reply is discarded -- the broadcast is a bare statement and
    nothing can read what came back. `1` means it is used whole: bound to a
    name, handed straight back to this function's own caller, or passed on as
    an argument. Anything above `1` is a tuple unpack, which is the only shape
    that fixes a length rather than just a type.

    `ast.Return` is the case worth naming, because reading only `ast.Assign`
    scores `return self.runner_mgr.call_func(...)` as a discard when the value
    is in fact the function's result -- `engine_core.py:749`, `dummy_execution`.
    """
    if isinstance(parent, ast.Expr):
        return 0
    if isinstance(parent, ast.Assign):
        target = parent.targets[0]
        return len(target.elts) if isinstance(target, ast.Tuple) else 1
    return 1


def _mentions(roots, needle, base=REPO):
    """Files under `roots` containing `needle`, as paths relative to `base`."""
    return {
        str(path.relative_to(base))
        for root in roots
        for path in root.rglob("*.py")
        if needle in path.read_text()
    }


def _call_sites():
    """Every name ATOM broadcasts to a worker, found by walking its source.

    The set of dispatched names is not written down anywhere in ATOM -- the
    dispatch is `getattr` on whatever arrives over the ring -- so it is
    recovered from the calls that put a name on the ring.

    Two filters can silently shrink that recovery, so both report what they
    dropped instead of dropping it quietly: broadcasts from `atom/compass`
    itself, and broadcasts whose first argument is not a literal and so names
    no one method. `test_both_filters_in_the_derivation_drop_nothing` asserts
    each is empty, which is what keeps the enumeration a derivation rather than
    an accident of the current tree.
    """
    sites: dict[str, list[Site]] = {}
    non_literal: list[str] = []
    from_compass: list[str] = []
    for path in sorted((REPO / "atom").rglob("*.py")):
        tree = ast.parse(path.read_text())
        parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr not in BROADCAST or not node.args:
                continue
            where = f"{path.relative_to(REPO)}:{node.lineno}"
            if "compass" in path.parts:
                from_compass.append(where)
                continue
            if not isinstance(node.args[0], ast.Constant):
                non_literal.append(where)
                continue
            aggregated = node.func.attr == "call_func_with_aggregation"
            waits = aggregated or any(
                k.arg == "wait_out" and getattr(k.value, "value", None) is True
                for k in node.keywords
            )
            sites.setdefault(node.args[0].value, []).append(
                Site(
                    path.name,
                    node.lineno,
                    waits,
                    aggregated,
                    _arity(parent.get(node)),
                )
            )
    return sites, non_literal, from_compass


SITES, NON_LITERAL, FROM_COMPASS = _call_sites()
BASE = _methods(_classes(ATOM_RUNNER)["ModelRunner"])
RAPID = _methods(_classes(ATOM_RUNNER)["RapidServeModelRunner"])
MIXIN = _methods(
    _classes(REPO / "atom/compass/runner/overrides.py")["NonAllocatingRunner"]
)
EXTENSION_CLASSES = {
    name
    for rel in ROLLOUT
    for node in _classes(REPO / rel).values()
    for name in _methods(node)
}


class Runner(NonAllocatingRunner):
    """The overrides over a base that supplies only what they read.

    `config` is present rather than absent: the base sets it first thing in its
    own `__init__`, so every override reads it, and a stub without one tests a
    shape the class is never in.
    """

    def __init__(self):
        self.config = SimpleNamespace(
            disagg_is_decode=False,
            speculative_config=None,
            eos_token_id=-1,
            stop_token_ids=[],
            pipeline_parallel_size=1,
        )
        self.capture_sizes = [0]
        self.capture_sizes_np = "untouched"


# --- the twelve, derived rather than typed ----------------------------------


def test_the_surface_is_every_dispatched_name_a_model_runner_answers():
    """The enumeration: what ATOM broadcasts, intersected with what it defines.

    A deployment's runner is whatever `runner_qualname` names, and the names
    broadcast at it are fixed by the engine. The ones that reach a plain
    `ModelRunner` are exactly this intersection -- no list, no judgement.
    """
    assert set(RPC_SURFACE) == set(SITES) & BASE
    assert len(RPC_SURFACE) == 12


def test_both_filters_in_the_derivation_drop_nothing():
    """The enumeration is a derivation only while both of these are empty.

    `_call_sites` skips broadcasts from `atom/compass` and broadcasts whose
    first argument is not a literal. Either would remove a name from `SITES`
    with nothing going red: the intersection would simply be smaller, and the
    table above would be edited to match it and still pass. Asserting the
    dropped sets are empty turns both filters from a silent exclusion into a
    tripwire -- an f-string call site, or a compass-side broadcast, fails here
    and has to be accounted for rather than vanishing.
    """
    assert NON_LITERAL == []
    assert FROM_COMPASS == []


def test_the_dispatched_names_outside_the_surface_belong_to_other_runners():
    """Without this the intersection above could shrink and look like a pass.

    The membership checks alone are not enough. A newly dispatched name that
    no class in the tree answers lands in `extension`, misses `BASE`, and would
    pass every assertion here while being a guaranteed park for whoever sends
    it. So the leftovers are counted, and `extension` is intersected against
    the classes that actually define those methods rather than accepted as a
    residue of two set subtractions.
    """
    rest = set(SITES) - set(RPC_SURFACE)
    assert rest, "no names left over means the intersection proved nothing"
    disagg = rest & RAPID
    extension = rest - RAPID
    assert len(disagg) == 7 and len(extension) == 7
    assert "prefill_forward" in disagg
    assert "update_weights" in extension
    # The leftovers are a runner ATOM ships (`RapidServeModelRunner`) or one it
    # mixes in elsewhere; none of them is on the class Compass extends, and
    # every one of them is defined by some class in the tree.
    assert not extension & BASE
    assert extension <= EXTENSION_CLASSES


@pytest.mark.parametrize("name", sorted(RPC_SURFACE))
def test_each_name_waits_exactly_as_its_own_call_sites_say(name):
    assert {s.waits for s in SITES[name]} == {RPC_SURFACE[name]}


# --- absence is the quietest failure on this surface -------------------------


def test_a_name_the_runner_lacks_is_skipped_and_not_raised():
    """`getattr(..., None)` plus `continue`: the worker never notices."""
    busy = _busy_loop()
    getattrs = [
        n
        for n in ast.walk(busy)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "getattr"
        and len(n.args) == 3
        and isinstance(n.args[2], ast.Constant)
        and n.args[2].value is None
    ]
    assert getattrs
    assert any(isinstance(n, ast.Continue) for n in ast.walk(busy))


def test_a_reply_is_forwarded_only_when_it_is_not_none():
    """A present method answering None parks its caller like an absent one.

    Asserted as the structure rather than as a string, because the claim is
    that *every* way a reply leaves the loop is behind that guard. `busy_loop`
    puts on two queues -- the primary output queue and the KV queue -- and if
    either one were ever moved outside the `if`, a None would start reaching a
    caller on that path and this file's whole account of the surface would be
    wrong for it.
    """
    busy = _busy_loop()
    guards = [
        n
        for n in ast.walk(busy)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and getattr(n.test.left, "id", None) == "out"
        and isinstance(n.test.ops[0], ast.IsNot)
        and n.test.comparators[0].value is None
    ]
    assert len(guards) == 1
    puts = [
        n.lineno
        for n in ast.walk(busy)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "put_nowait"
    ]
    guarded = [
        n.lineno
        for n in ast.walk(guards[0])
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "put_nowait"
    ]
    assert puts and sorted(puts) == sorted(guarded)


def test_the_caller_that_waits_has_no_timeout_to_rescue_it():
    """So a reply that is never queued is a wait with no end, not an error."""
    call_func = next(
        n
        for n in ast.walk(ast.parse(ASYNC_PROC))
        if isinstance(n, ast.FunctionDef) and n.name == "call_func"
    )
    gets = [
        n
        for n in ast.walk(call_func)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "get"
    ]
    assert gets and not any(g.args or g.keywords for g in gets)


def test_the_aggregating_form_is_the_one_bounded_wait():
    """Its breach is a stall and a None, not a hang -- and only one name uses it."""
    aggregated = {n for n, s in SITES.items() if any(x.aggregated for x in s)}
    assert aggregated == {"async_proc_aggregation"}
    assert '_KV_FUNC_NAMES = frozenset(["async_proc_aggregation"])' in ASYNC_PROC
    assert "output_queue.get(timeout=timeout)" in ASYNC_PROC


def test_a_refusal_reaches_the_caller_instead_of_stranding_it():
    """Why raising is the loud option although `busy_loop` catches nothing.

    `out = func(*args)` sits under no `try`, so an exception unwinds out of
    `busy_loop`, out of `AsyncIOProc.__init__` -- which is the process target --
    and the worker exits. That alone would strand the caller. What does not
    strand it is the manager: a monitor thread, started from `__init__` before
    any RPC can be sent, waits on the process sentinels and calls `exit()`,
    which puts a `SystemExit` on the output queue **before** it finalizes
    anything, and `call_func` re-raises it. Every wait on that path is bounded.

    The ordering is the load-bearing part, so it is asserted rather than read:
    a finalizer that ran first would tear the manager down with the caller
    still parked.

    What the caller does *not* get is asserted too: the `SystemExit` is built
    with no arguments, so the refusal's type crosses the boundary and its name
    and its reason do not. Anything a successor wants readable in the engine's
    own log it has to log on the worker side before raising.
    """
    tree = ast.parse(ASYNC_PROC)
    manager = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AsyncIOProcManager"
    )
    bodies = {n.name: n for n in manager.body if isinstance(n, ast.FunctionDef)}

    # The monitor exists before the first RPC can be broadcast.
    init = bodies["__init__"]
    assert "monitor_procs" in {
        n.func.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_self.exit()" in ASYNC_PROC

    # The worker's own dispatch catches nothing, which is what kills it.
    busy = _busy_loop()
    assert not [n for n in ast.walk(busy) if isinstance(n, ast.Try)]

    # SystemExit is queued before the manager finalizes its parent.
    lines = {}
    for n in ast.walk(bodies["exit"]):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            queued = n.func.attr == "put_nowait" and n.args
            if queued and _raised_name(n.args[0]) == "SystemExit":
                lines["queued"] = n.lineno
                # Empty args: the type is the whole of the message.
                assert not n.args[0].args and not n.args[0].keywords
            if n.func.attr == "parent_finalizer":
                lines["finalized"] = n.lineno
    assert lines["queued"] < lines["finalized"]
    assert "if isinstance(ret, SystemExit):\n                raise ret" in ASYNC_PROC


# --- the surface is answered, and that is checked where it is composed -------


@pytest.mark.parametrize("name", sorted(RPC_SURFACE))
def test_the_composed_runner_defines_every_name(name):
    """Read from source: importing the composed class needs a driver."""
    assert name in MIXIN | BASE


def test_a_runner_with_a_hole_is_reported_by_name():
    assert unanswered_rpc_names(object) == tuple(RPC_SURFACE)
    whole = type("Whole", (), {n: None for n in RPC_SURFACE})
    assert unanswered_rpc_names(whole) == tuple(RPC_SURFACE)  # None is silence too
    answering = type("Answering", (), {n: (lambda self: True) for n in RPC_SURFACE})
    assert unanswered_rpc_names(answering) == ()


def test_the_binding_module_refuses_rather_than_composing_a_hole():
    """Two greps, and this is as far as a CPU tier can go.

    Executing the device means importing the composed class, which imports
    `ModelRunner`, which runs aiter's architecture probe and needs a driver.
    So this asserts the device is wired up and not that it fires correctly;
    that it fires correctly can only be observed with a driver present.
    What is checked here is the part that can drift silently: that the message
    partitions the missing names on `RPC_SURFACE` instead of telling one story
    about all twelve, since two of them are waited on by nobody.
    """
    assert "unanswered_rpc_names(CompassModelRunner)" in COMPOSED
    assert "raise RunnerRefusal(" in COMPOSED
    assert "RPC_SURFACE[name]" in COMPOSED
    assert "not RPC_SURFACE[name]" in COMPOSED


# --- capture_cudagraph: the three values, taken from the unpack ---------------


def test_capture_cudagraph_answers_the_arity_its_call_sites_unpack():
    arities = {s.arity for s in SITES["capture_cudagraph"]}
    assert arities == {3}
    assert len(SITES["capture_cudagraph"]) == 2
    cap_cost, bs, pool_bytes = Runner().capture_cudagraph()
    # Exactly what the two call sites then do with the three, none of which
    # may raise on the value this runner sends.
    assert f"{cap_cost:.2f}" == "0.00"
    assert f"cudagraph capture{bs}" == "cudagraph capture[]"
    assert pool_bytes / (1 << 30) == 0.0


def test_capture_cudagraph_says_that_nothing_was_captured():
    assert Runner().capture_cudagraph() == (0.0, [], 0)


def test_capture_cudagraph_leaves_the_eager_capture_sizes_alone():
    """The attention metadata builder reads them on every step."""
    runner = Runner()
    runner.capture_cudagraph()
    assert runner.capture_sizes == [0]
    assert runner.capture_sizes_np == "untouched"


def test_the_base_capture_reaches_a_device_before_it_reaches_the_model():
    """Why this one is replaced and the rest of the surface is not."""
    body = ATOM_RUNNER.read_text().split("    def capture_cudagraph(self):")[1]
    body = body.split("\n    def ")[0]
    assert 'self.forward_vars["kv_indptr"].gpu.zero_()' in body
    assert "graph_capture()" in body


# --- the refusals, with the shape a successor has to produce -----------------


def test_get_num_blocks_refuses_and_the_keys_its_caller_reads_are_named():
    site = SITES["get_num_blocks"][0]
    tree = ast.parse((ENGINE / site.file).read_text())
    required = {
        n.slice.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        and getattr(n.value, "id", None) == "block_info"
        and isinstance(n.slice, ast.Constant)
    }
    optional = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "get"
        and getattr(n.func.value, "id", None) == "block_info"
    }
    assert required == {"num_kvcache_blocks", "state_runtime"}
    assert optional == {"pool_entries", "pool_entries_per_req"}
    answered = next(
        n
        for n in ast.walk(_classes(ATOM_RUNNER)["ModelRunner"])
        if isinstance(n, ast.FunctionDef) and n.name == "get_num_blocks"
    )
    returned = next(
        {k.value for k in n.value.keys}
        for n in ast.walk(answered)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
    )
    assert required | optional == returned
    # The four keys are now forwarded from the base rather than described in a
    # refusal, so what is left to check on this side is that the one path that
    # answers nothing still names what would make it answer.
    with pytest.raises(RunnerRefusal, match="install_device_readings"):
        Runner().get_num_blocks()


def test_the_fourth_value_is_a_two_key_wire_dict_and_not_an_opaque_object():
    """The tightest constraint on the four, and the one easiest to read past.

    `engine_core.py` does not store `block_info["state_runtime"]`; it passes it
    to `StateRuntime.from_wire`, which rejects a non-`Mapping` with `TypeError`
    and any key set other than `{"transfer", "checkpoint_spec"}` with
    `ValueError`. Both raise in the **parent**, on the first RPC of the
    engine's life, so a successor that fills the key with anything plausible
    takes the engine down rather than degrading.
    """
    engine = (ENGINE / "engine_core.py").read_text()
    assert 'StateRuntime.from_wire(block_info["state_runtime"])' in engine
    from_wire = next(
        n
        for n in ast.walk(_classes(ENGINE / "state_runtime.py")["StateRuntime"])
        if isinstance(n, ast.FunctionDef) and n.name == "from_wire"
    )
    raised = {
        _raised_name(n.exc) for n in ast.walk(from_wire) if isinstance(n, ast.Raise)
    }
    assert {"TypeError", "ValueError"} <= raised
    expected = next(
        n
        for n in ast.walk(from_wire)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "expected"
    )
    assert {k.value for k in expected.value.elts} == {"transfer", "checkpoint_spec"}
    # And the block count itself is not free either.
    block_manager = (ENGINE / "block_manager.py").read_text()
    assert "assert num_blocks > 0" in block_manager


def test_forward_refuses_and_its_reply_is_one_object_read_for_nine_attributes():
    """What a successor owes: one value, never unpacked, read by attribute.

    Four sites broadcast it, two per file. Three use the reply and the fourth
    discards it -- `pp_engine_core.py:118`, the head's launch, whose tokens
    come back over the transport instead. The nine attribute names are
    recovered from the consumers rather than listed, so a tenth read, or a
    rename, fails here.
    """
    assert collections.Counter(s.file for s in SITES["forward"]) == {
        "engine_core.py": 2,
        "pp_engine_core.py": 2,
    }
    assert {s.arity for s in SITES["forward"]} == {0, 1}
    engine = (ENGINE / "engine_core.py").read_text()
    assert (
        "self.scheduler.postprocess(\n            seqs,\n            fwd_out," in engine
    )

    reads = set()
    for rel, local in (
        ("scheduler.py", "fwd_output"),
        ("pp_engine_core.py", "fwd_out"),
    ):
        tree = ast.parse((ENGINE / rel).read_text())
        reads |= {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) == local
        }
    assert reads == {
        "req_ids",
        "token_ids",
        "draft_token_ids",
        "is_deferred_out",
        "logprobs",
        "get_idx",
        "num_rejected",
        "num_bonus",
        "dspark_ell",
    }
    with pytest.raises(RunnerRefusal, match="produces output"):
        Runner().forward(object())


def test_the_pp_reply_is_read_in_the_head_and_not_at_the_last_stages_call():
    """The correction that a substring check cannot make.

    `pp_engine_core.py:379` is inside `_downstream_busy_loop` -- the **last**
    stage. Nothing there reads the reply: the two nearby `.req_ids` reads are
    on `batch`, and `fwd_out` is handed whole to `send_tokens`. The `.req_ids`
    read is one ZMQ hop later, in the head, at `:144-147`, on what
    `recv_tokens()` returned at `:139`. So the reply must survive a pickle
    round trip, which reading only the call site would never say.
    """
    tree = ast.parse((ENGINE / "pp_engine_core.py").read_text())
    enclosing = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Attribute)
                and getattr(n.value, "id", None) == "fwd_out"
            ):
                enclosing.setdefault(n.attr, set()).add(fn.name)
    assert enclosing == {"req_ids": {"_pp_head_step"}}

    downstream = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_downstream_busy_loop"
    )
    broadcast = [
        n.args[0].value
        for n in ast.walk(downstream)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) in BROADCAST
        and n.args
        and isinstance(n.args[0], ast.Constant)
    ]
    assert "forward" in broadcast

    transport = (REPO / "atom/distributed/pp_transport.py").read_text()
    for fn in ("send_tokens", "recv_tokens"):
        body = transport.split(f"def {fn}(")[1].split("\n    def ")[0]
        assert "pickle." in body


def test_dummy_execution_is_this_runners_forward_and_refuses_with_it():
    """Which is why it is not replaced: it answers as soon as `forward` does."""
    body = next(
        n
        for n in ast.walk(_classes(ATOM_RUNNER)["ModelRunner"])
        if isinstance(n, ast.FunctionDef) and n.name == "dummy_execution"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "forward"
        for n in ast.walk(body)
    )
    assert body.body[-1].value.value is True
    # Its one site returns the reply to its own caller (`engine_core.py:749`),
    # which is a read and not a discard. Scoring it 0 is the modelling error
    # `_arity` exists to avoid, and this is the site that catches it.
    assert [s.arity for s in SITES["dummy_execution"]] == [1]


# --- the reply rule, applied to the names left to ATOM -----------------------


@pytest.mark.parametrize(
    "name", sorted(n for n, w in RPC_SURFACE.items() if w and n not in MIXIN)
)
def test_every_waited_name_left_to_atom_ends_in_a_value(name):
    """A bare `return` on any of these is a caller that never wakes up."""
    body = next(
        n
        for n in ast.walk(_classes(ATOM_RUNNER)["ModelRunner"])
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    assert isinstance(body.body[-1], ast.Return) and body.body[-1].value is not None
    assert not [
        n for n in ast.walk(body) if isinstance(n, ast.Return) and n.value is None
    ]


def test_the_profiler_replies_are_forwarded_whole_and_never_unpacked():
    """The pair whose keys come from the implementation, not from a caller.

    `ModelRunner.stop_profiler` documents a `{trace_dir, elapsed}` dict, and it
    is tempting to read that as the shape the call site requires. It is not.
    `engine_utility.py` logs the reply whole and puts it in a response
    envelope; `llm_engine.py:300`'s `.get("result", {})` is on that envelope,
    not on the reply; and `trace_dir` appears nowhere on either side of the
    reply except the three lines of `model_runner.py` that produce it. What
    the callers impose is non-None and picklable. The keys are a convention a
    successor inherits, and stating them as a requirement would be stating a
    tighter contract than anything checks -- which is a documentation defect
    even when it errs safe.

    The scan reads the engine and the Compass runner package -- the code this
    reply crosses -- and not every package that will ever sit under
    `atom/compass`. The three tests below hold it to both halves of that: it
    still catches a mention inside the runner package, and it stays quiet
    about one outside it.
    """
    assert {s.arity for s in SITES["start_profiler"]} == {1}
    assert {s.arity for s in SITES["stop_profiler"]} == {1}
    assert _mentions(REPLY_SURFACE, "trace_dir") == {
        "atom/model_engine/model_runner.py"
    }
    utility = (ENGINE / "engine_utility.py").read_text()
    assert '("UTILITY_RESPONSE", {"cmd": "stop_profile", "result": result})' in utility


# --- the scan itself, which has to catch something and not everything --------


FAKE_TREE = (
    "atom/model_engine/model_runner.py",
    "atom/compass/runner/overrides.py",
    "atom/compass/spec/reader.py",
)
PRODUCER = FAKE_TREE[0]


def _fake_repo(tmp_path, mentioning):
    """A three-file stand-in repo, with `trace_dir` written into `mentioning`.

    One call builds either direction of the pin. The roots handed back are
    `REPLY_SURFACE` re-rooted under `tmp_path` rather than two paths written
    out again here: a scan widened to `atom/` would be widened here too, and
    the direction it is widened past -- the package beside the runner -- is the
    third file. Roots typed out a second time would pin `_mentions` and leave
    the scope free, which is what these two tests exist to stop.
    """
    for rel in FAKE_TREE:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# trace_dir\n" if rel in mentioning else "pass\n")
    return tuple(tmp_path / root.relative_to(REPO) for root in REPLY_SURFACE)


def test_the_scan_looks_at_both_sides_of_the_reply():
    """A scan narrowed until it reaches nothing would pass forever.

    The producer and the two Compass runner modules are named here, so a root
    that moves or a package that is renamed fails rather than quietly
    shrinking the set the assertion above is computed over.

    Naming files bounds the scan from below only, and every wider scope -- up
    to `atom/` itself, which is the scope this one replaced -- satisfies a
    lower bound. The second assertion is the upper one, and it is exact: the
    roots are the engine and the runner package, each holding a
    `model_runner.py` -- ATOM's, which produces the reply, and the Compass one
    that stands in for it. A sibling added to the constant, `atom/compass/clock`
    say, fails here instead of passing quietly.
    """
    scanned = {
        str(path.relative_to(REPO))
        for root in REPLY_SURFACE
        for path in root.rglob("*.py")
    }
    assert {
        "atom/model_engine/model_runner.py",
        "atom/compass/runner/overrides.py",
        "atom/compass/runner/model_runner.py",
    } <= scanned
    assert set(REPLY_SURFACE) == {ENGINE, PACKAGE}


def test_the_scan_still_catches_the_string_inside_the_runner_package(tmp_path):
    """Where the contract does reach, the assertion is the old assertion."""
    roots = _fake_repo(tmp_path, {PRODUCER, "atom/compass/runner/overrides.py"})
    assert _mentions(roots, "trace_dir", tmp_path) == {
        PRODUCER,
        "atom/compass/runner/overrides.py",
    }


def test_the_scan_ignores_a_compass_package_the_reply_never_reaches(tmp_path):
    """The half a read of the whole tree gets wrong.

    A module under `atom/compass` that the reply never touches may mention
    `trace_dir` -- in a comment, in a docstring, in a field name of its own --
    without failing a test about the profiler reply. Nor may the scan read a
    root of its own when handed none: every file contains the empty string,
    and handed no roots it finds nothing. A root added only when roots are
    handed in, or only for a non-empty needle, passes every test in this file.
    """
    roots = _fake_repo(tmp_path, {PRODUCER, "atom/compass/spec/reader.py"})
    assert _mentions(roots, "trace_dir", tmp_path) == {PRODUCER}
    assert _mentions((), "") == set()


@pytest.mark.parametrize("surface", [(), (ENGINE,), REPLY_SURFACE])
def test_the_reply_assertion_scans_the_roots_the_constant_names(monkeypatch, surface):
    """The tests above hold `REPLY_SURFACE`; this one holds its reader.

    The scan is replaced by a spy that records the roots it is handed, and the
    profiler-reply assertion runs with the constant set to `surface`: empty,
    the engine alone, and the constant as it stands. The spy must see exactly
    that surface each time. A root added beside the constant fails all three
    runs, and the constant's two roots written out at the call site fail the
    first two; a fallback for an empty constant fails the empty one; a root
    that appears only when the constant is set, or one derived from every root
    -- each root's parent, say -- fails the last two; one derived from the
    Compass root alone -- its parent, all of `atom/compass` -- fails the last.
    A scan that bypasses `_mentions` reaches the spy not at all. What passes is
    any expression that is the identity at these three surfaces, the last of
    which is the one the assertion runs at.
    """
    seen = []

    def spy(roots, needle):
        seen.append(roots)
        return {"atom/model_engine/model_runner.py"}

    monkeypatch.setitem(globals(), "REPLY_SURFACE", surface)
    monkeypatch.setitem(globals(), "_mentions", spy)
    test_the_profiler_replies_are_forwarded_whole_and_never_unpacked()
    assert seen == [surface]


def test_the_two_names_no_caller_waits_for_and_what_replying_costs():
    """A reply nobody reads sits on the queue for whoever asks next.

    `process_kvconnector_output` therefore answers nothing at all. `exit` does
    reply, and is the one place that is harmless: `busy_loop` breaks on the
    name, so there is no next caller.

    These two are also the reason the composed class's refusal message
    partitions the surface. A hole at either parks nobody -- the worker skips
    the name and carries on -- so "each one would park its caller" is false for
    exactly these.
    """
    assert {n for n, w in RPC_SURFACE.items() if not w} == {
        "exit",
        "process_kvconnector_output",
    }
    bodies = {
        n.name: n
        for n in ast.walk(_classes(ATOM_RUNNER)["ModelRunner"])
        if isinstance(n, ast.FunctionDef)
    }
    silent = bodies["process_kvconnector_output"]
    assert not [n for n in ast.walk(silent) if isinstance(n, ast.Return) and n.value]
    assert bodies["exit"].body[-1].value.value is True
    assert 'if func_name == "exit":\n                break' in ASYNC_PROC


# --- the break at `exit`, and the comment that describes it ------------------


def test_the_break_on_exit_is_a_sibling_of_the_per_runner_loop():
    """Where the break sits is the whole of why a hole at `exit` is not a hang.

    Read off the structure rather than off the source text, because the
    indentation *is* the claim: the `if` is a statement of the `while` body
    beside `for runner in self.runners`, not a statement inside it. So it
    fires on the name dequeued at the top of the iteration and consults no
    reply -- a runner with no `exit` is skipped by the `getattr`, and the loop
    still breaks.

    The assertion in
    `test_the_two_names_no_caller_waits_for_and_what_replying_costs` pins the
    same two lines as an exact string. That catches the break moving one level
    in, because its own indentation would change, but says nothing about what
    the `if` is a sibling of or about what its test reaches. Both are here.
    """
    busy = _busy_loop()
    loop = next(n for n in busy.body if isinstance(n, ast.While))
    dispatch = next(n for n in loop.body if isinstance(n, ast.For))
    guards = [
        n
        for n in loop.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and getattr(n.test.left, "id", None) == "func_name"
        and getattr(n.test.comparators[0], "value", None) == "exit"
    ]
    assert len(guards) == 1
    guard = guards[0]
    assert [type(n) for n in guard.body] == [ast.Break]
    assert guard not in list(ast.walk(dispatch))
    assert guard.col_offset == dispatch.col_offset
    assert {n.col_offset for n in dispatch.body} == {dispatch.col_offset + 4}
    dequeued = next(
        n
        for n in loop.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "attr", None) == "get_func"
    )
    bound = {n.id for n in ast.walk(dequeued.targets[0]) if isinstance(n, ast.Name)}
    reached = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
    assert reached == {"func_name"} and reached <= bound


def test_the_refusal_comment_says_the_loop_breaks_and_not_that_it_hangs():
    """The prose beside the refusal, held to the structure above.

    This is the half the surface was missing. The comment, the package
    docstring's sentence about the surface and the string assertion above
    were added together, and the comment said the opposite of that assertion:
    that an unanswered `exit` means "the loop never breaks". Nothing failed,
    because nothing read the prose. So the words are read here.

    `never breaks` is the claim that was wrong. It is refused as that
    lower-case phrase, including where a comment line wraps between the two
    words. The three phrases required are the three findings of the structure
    test -- that it breaks, what the `if` is a sibling of, and that it tests a
    name rather than a reply -- so prose and source now fail together.
    """
    comment = _refusal_comment()
    assert "The loop breaks either way" in comment
    assert "the break is a sibling of the per-runner loop" in comment
    assert "tests the dispatched name rather than any reply" in comment
    assert "never breaks" not in comment


def test_what_the_comment_says_a_hole_at_exit_loses_is_what_exit_does():
    """Each loss the comment names, against `ModelRunner.exit`'s own body.

    The comment is what this test holds against `ModelRunner.exit`'s body: it
    says what shutdown fails to release about ATOM's code rather than this
    package's, so it drifts whenever `exit` is edited. It also states which of
    those losses is empty here -- the five KV deletions are `hasattr`-guarded
    and this runner allocates no KV tensor, so they find nothing. The guard is
    asserted too: dropping it would make the comment's own exception false.
    """
    comment = _refusal_comment()
    body = next(
        n
        for n in ast.walk(_classes(ATOM_RUNNER)["ModelRunner"])
        if isinstance(n, ast.FunctionDef) and n.name == "exit"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(body) if isinstance(n, ast.Call)}
    assert {"destroy_dist_env", "torch.cuda.empty_cache"} <= calls
    assert "`ModelRunner.exit` never runs" in comment
    assert "the distributed environment is never destroyed" in comment
    assert "`torch.cuda.empty_cache()` never runs" in comment
    deleted = {
        ast.unparse(t)
        for n in ast.walk(body)
        if isinstance(n, ast.Delete)
        for t in n.targets
    }
    assert "self.model" in deleted
    assert "`self.model` is never dropped" in comment
    literal_loops = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.For) and isinstance(n.iter, (ast.Tuple, ast.List))
    ]
    assert len(literal_loops) == 1
    kv = literal_loops[0]
    assert {e.value for e in kv.iter.elts if isinstance(e, ast.Constant)} == {
        "kv_cache",
        "kv_scale",
        "index_cache",
        "mamba_k_cache",
        "mamba_v_cache",
    }
    assert any(
        isinstance(n, ast.If)
        and getattr(getattr(n.test, "func", None), "id", None) == "hasattr"
        for n in ast.walk(kv)
    )
    assert "five KV-tensor deletions are `hasattr`-guarded" in comment


def test_the_unanswered_helper_describes_its_whole_return_and_not_one_half():
    """The list is returned unpartitioned; its docstring may not be.

    `unanswered_rpc_names` draws from all twelve, and its only caller in the
    package splits them on `RPC_SURFACE` before reporting them. A docstring
    that gives one story for the whole return is the claim
    `test_the_two_names_no_caller_waits_for_and_what_replying_costs` already
    calls false, so the two names it excepts are read back out of the prose
    and compared with the table rather than typed here. The count word is
    held to `len(RPC_SURFACE)` the same way -- the word is looked up from the
    table's length, not typed beside a literal 12 -- and the single-caller
    claim to the tree.
    """
    doc = " ".join(unanswered_rpc_names.__doc__.split())
    unwaited = {n for n, w in RPC_SURFACE.items() if not w}
    assert {n for n in RPC_SURFACE if f"`{n}`" in doc} == unwaited
    assert "a hole in either parks no one" in doc
    assert "parks forever" not in doc
    word = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}.get(len(RPC_SURFACE))
    assert word is not None, f"no count word for {len(RPC_SURFACE)} names"
    assert f"all {word}" in doc
    callers = [
        str(f.relative_to(REPO))
        for f in sorted((REPO / "atom").rglob("*.py"))
        if any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "unanswered_rpc_names"
            for n in ast.walk(ast.parse(f.read_text()))
        )
    ]
    assert callers == ["atom/compass/runner/model_runner.py"]
    tree = ast.parse(COMPOSED)
    bound = next(
        n.targets[0].id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "unanswered_rpc_names"
    )
    splits = sorted(
        ast.unparse(c.ifs[0])
        for c in ast.walk(tree)
        if isinstance(c, ast.comprehension)
        and getattr(c.iter, "id", None) == bound
        and c.ifs
    )
    assert splits == ["RPC_SURFACE[name]", "not RPC_SURFACE[name]"]


def test_the_zero_block_form_in_the_tree_answers_two_of_the_four_keys():
    """A trap for whoever ends the `get_num_blocks` refusal above.

    `RapidServeModelRunner.get_num_blocks` is the obvious thing to copy -- a
    non-allocating runner answering without a device -- and it returns
    `num_kvcache_blocks` and `state_runtime` only. That is not a breach: the
    caller takes the two pool-entry keys with a default. It is short, and the
    shortfall is invisible at zero blocks and not invisible above them, so the
    count is asserted here rather than left to be noticed.

    The default is not neutral either. `block_manager.py:116-121` turns a
    missing `pool_entries` into `num_state_slots = 0`, which is a decision
    input at `:162` and in the permanent-unschedulable predicate at
    `scheduler.py:1364-1376` -- so on a per-request-state model the missing key
    reads as "no slots ever existed" rather than raising.
    """
    short = next(
        n
        for n in ast.walk(_classes(ATOM_RUNNER)["RapidServeModelRunner"])
        if isinstance(n, ast.FunctionDef) and n.name == "get_num_blocks"
    )
    keys = next(
        {k.value for k in n.value.keys}
        for n in ast.walk(short)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
    )
    assert keys == {"num_kvcache_blocks", "state_runtime"}
    engine = (ENGINE / "engine_core.py").read_text()
    assert 'block_info.get("pool_entries", {})' in engine
    assert 'block_info.get("pool_entries_per_req", {})' in engine


# --- the package docstring, read against the table it describes --------------
#
# What is read here is what the docstring states through tokens the tree can
# check: bullet leads, the `file.py:NN` citations under each lead, a module
# count, and the engine names, functions and paths each module's bullet cites.
# The sentences between them are not read. Which half of the table parks its
# caller, and whether an engine import sits at module scope, differ from their
# false forms only in wording, and asserting wording is not a check on what the
# wording claims. Which function holds which engine name is not read either: a
# bullet is held to the union of the names and the functions they are imported
# in, not to the pairs. Nor is an engine module loaded through `__import__`,
# which is a call and not an import statement.

CITATION = r"`([a-z_]+\.py:\d+)`"
# Words from two up: a split needs two, so "the one module here" is no count.
COUNT_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine")


def _bullets(doc):
    """Each top-level bullet of *doc*, keyed by the backticked name it opens with.

    The package docstring carries two bullet lists -- the modules it splits and
    the dispatched names no caller waits for -- and every entry of both opens
    with one backticked identifier. Reading the leads rather than searching the
    whole text is what lets the two lists be checked separately, and it is also
    what keeps an incidental mention from counting: `forward` is named in the
    prose of a module bullet and is not an entry of either list. A bullet's
    text runs through the lines indented under it, so what it says is read as
    that entry's and not as the whole docstring's.
    """
    pattern = r"^- `([A-Za-z_][A-Za-z0-9_]*)`(.*(?:\n  .*)*)"
    return dict(re.findall(pattern, doc, flags=re.MULTILINE))


def _modules():
    """The package's own modules: its `.py` files and its subpackage directories."""
    return {
        p.stem
        for p in PACKAGE.iterdir()
        if p.suffix == ".py" or (p.is_dir() and any(p.rglob("*.py")))
    } - {"__init__"}


def _stated_module_counts(doc):
    """Every count *doc* puts before "module(s)": a digit, or a word from two up."""
    words = re.findall(rf"(?i)\b(\d+|{'|'.join(COUNT_WORDS)})\s+modules?\b", doc)
    return [int(w) if w.isdigit() else COUNT_WORDS.index(w.lower()) + 2 for w in words]


def test_the_package_docstring_lists_every_module_beside_it():
    """The split it describes has to be over the package's own modules.

    The docstring said "Two modules" for as long as there were three:
    `step_output` was added after the sentence was written, and nothing went
    red, because a count in prose has nothing to disagree with. So the bullets
    are compared against the directory instead of against a number, and the
    next module either appears in them or fails here -- including one that
    arrives as a subpackage directory rather than as a `.py` file.
    """
    modules = _modules()
    assert "step_output" in modules, "the walk found no package to compare against"
    assert set(_bullets(PACKAGE_DOC)) - set(RPC_SURFACE) == modules


def test_a_module_count_the_package_docstring_states_is_the_packages():
    """A count in prose, given the directory to disagree with.

    The docstring states no count today, and need not. Where it does state one,
    it is the number of modules the package holds, so "Two modules" put back
    above three bullets fails here rather than reading as true.
    """
    assert _stated_module_counts("Two modules; eight\n modules; one module") == [2, 8]
    for stated in _stated_module_counts(PACKAGE_DOC):
        assert stated == len(_modules()), f"the docstring says {stated} modules"


def test_the_package_docstring_partitions_the_surface_the_way_the_table_does():
    """A hole in this surface is quiet, but it is not one failure.

    `RPC_SURFACE`'s value is whether the caller waits, and the two answers fail
    differently: a hole in a waited name parks its caller for the life of the
    process, and a hole in an unwaited one parks nobody and loses the work the
    name stood for. The docstring claimed the first for all twelve until it was
    corrected, which named the one failure mode that cannot happen at the other
    two and sent a reader debugging a leak or a missing KV load to look for a
    park that does not exist.

    This fails in both directions. Reinstating the unpartitioned sentence
    leaves the docstring naming neither name; flipping an entry of the table,
    or adding a thirteenth that no caller waits for, leaves it naming the wrong
    ones. The two guards either side of the assertion keep it from passing
    vacuously if the table ever stopped recording two answers at all.
    """
    assert UNWAITED, "a table with nothing unwaited would pass the partition trivially"
    assert UNWAITED != set(RPC_SURFACE), "and so would a table with nothing waited"
    assert set(_bullets(PACKAGE_DOC)) & set(RPC_SURFACE) == UNWAITED


def test_every_site_the_package_docstring_cites_is_one_no_caller_waits_for():
    """The partition is a claim about call sites, so it carries them.

    Each unwaited bullet names where ATOM broadcasts that name, and `SITES` is
    recovered from ATOM's own source rather than from this file, so the
    citations are checked against the tree instead of read as decoration. Set
    equality covers the way either half drifts: a broadcast that moves, or a
    second site that appears, is uncited; a site that starts passing
    `wait_out=True`, or a name that leaves the unwaited half, is cited and
    should not be.

    The union alone does not say which name a site belongs to: two bullets
    with their bodies swapped cite the same six sites between them. So each
    bullet's citations are also held to its own name's sites.
    """
    cited = set(re.findall(CITATION, PACKAGE_DOC))
    sites = [s for name in UNWAITED for s in SITES[name]]
    assert [s for s in sites if s.waits] == []
    assert cited == {f"{s.file}:{s.line}" for s in sites}
    bullets = _bullets(PACKAGE_DOC)
    assert {n: set(re.findall(CITATION, bullets.get(n, ""))) for n in UNWAITED} == {
        n: {f"{s.file}:{s.line}" for s in SITES[n]} for n in UNWAITED
    }


def _imported(node, alias):
    """The absolute module an import names, relative spellings resolved.

    The alias is joined on only when the module is not itself in the engine,
    so `from atom import model_engine` reads as `atom.model_engine`.
    """
    if isinstance(node, ast.Import):
        return alias.name
    module = resolve_name("." * node.level + (node.module or ""), "atom.compass.runner")
    engine = module.startswith("atom.model_engine")
    return module if engine else f"{module}.{alias.name}"


def _engine_imports(mod):
    """(name bound, engine module, enclosing function) for each engine import."""
    tree = ast.parse((PACKAGE / f"{mod}.py").read_text())
    scope = {
        c: f.name
        for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        for c in ast.walk(f)
    }
    return [
        (a.asname or a.name, _imported(n, a), scope.get(n))
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
        if _imported(n, a).startswith("atom.model_engine")
    ]


@pytest.mark.parametrize("mod", ["overrides", "step_output", "model_runner"])
def test_a_module_bullet_cites_the_engine_imports_its_module_makes(mod):
    """A module bullet cites every engine import its module makes, and no other."""
    cited = set(re.findall(r"`([^`]+)`", _bullets(PACKAGE_DOC)[mod]))
    imports = _engine_imports(mod)
    missing = [i for i in imports if not {i[0], i[2] or i[0]} <= cited]
    assert missing == [], f"the {mod} bullet omits {missing}"
    named = {c for c in cited if c.startswith("atom.model_engine")}
    assert named <= {i[1] for i in imports}, f"the {mod} bullet names {named}"
