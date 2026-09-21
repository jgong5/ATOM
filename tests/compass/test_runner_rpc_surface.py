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
on the output queue, and `call_func` re-raises it in the caller. So the rule
this file enforces is not "return something sensible" but **return what the
call site unpacks** -- checked against ATOM's source, never against a list
anyone typed here.

Nothing here imports `atom.model_engine.model_runner`: that import runs aiter's
architecture probe, which shells out to `rocminfo` and raises where there is no
GPU. The composed class is therefore read from source, the way the sibling
module's tests read it.
"""

import ast
import pathlib
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
BROADCAST = ("call_func", "call_func_with_aggregation")


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


def _raised_name(node):
    """The name of the exception a call constructs, if it constructs one."""
    return getattr(getattr(node, "func", None), "id", None)


def _call_sites():
    """Every name ATOM broadcasts to a worker, found by walking its source.

    The set of dispatched names is not written down anywhere in ATOM -- the
    dispatch is `getattr` on whatever arrives over the ring -- so it is
    recovered from the calls that put a name on the ring.
    """
    sites: dict[str, list[Site]] = {}
    for path in sorted((REPO / "atom").rglob("*.py")):
        if "compass" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr not in BROADCAST or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant):
                continue
            aggregated = node.func.attr == "call_func_with_aggregation"
            waits = aggregated or any(
                k.arg == "wait_out" and getattr(k.value, "value", None) is True
                for k in node.keywords
            )
            target = parent.get(node)
            target = target.targets[0] if isinstance(target, ast.Assign) else None
            arity = (
                len(target.elts)
                if isinstance(target, ast.Tuple)
                else 1
                if target is not None
                else 0
            )
            sites.setdefault(node.args[0].value, []).append(
                Site(path.name, node.lineno, waits, aggregated, arity)
            )
    return sites


SITES = _call_sites()
BASE = _methods(_classes(ATOM_RUNNER)["ModelRunner"])
RAPID = _methods(_classes(ATOM_RUNNER)["RapidServeModelRunner"])
MIXIN = _methods(
    _classes(REPO / "atom/compass/runner/overrides.py")["NonAllocatingRunner"]
)


class Runner(NonAllocatingRunner):
    """The overrides over a base that supplies only what they read."""

    def __init__(self):
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


def test_the_dispatched_names_outside_the_surface_belong_to_other_runners():
    """Without this the intersection above could shrink and look like a pass."""
    rest = set(SITES) - set(RPC_SURFACE)
    assert rest, "no names left over means the intersection proved nothing"
    disagg = rest & RAPID
    extension = rest - RAPID
    assert "prefill_forward" in disagg
    assert "update_weights" in extension
    # The leftovers are a runner ATOM ships (`RapidServeModelRunner`) or one it
    # mixes in elsewhere; none of them is on the class Compass extends.
    assert not extension & BASE


@pytest.mark.parametrize("name", sorted(RPC_SURFACE))
def test_each_name_waits_exactly_as_its_own_call_sites_say(name):
    assert {s.waits for s in SITES[name]} == {RPC_SURFACE[name]}


# --- absence is the quietest failure on this surface -------------------------


def test_a_name_the_runner_lacks_is_skipped_and_not_raised():
    """`getattr(..., None)` plus `continue`: the worker never notices."""
    busy = next(
        n
        for n in ast.walk(ast.parse(ASYNC_PROC))
        if isinstance(n, ast.FunctionDef) and n.name == "busy_loop"
    )
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
    assert "if out is not None:" in ASYNC_PROC


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
    busy = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "busy_loop"
    )
    assert not [n for n in ast.walk(busy) if isinstance(n, ast.Try)]

    # SystemExit is queued before the manager finalizes its parent.
    lines = {}
    for n in ast.walk(bodies["exit"]):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            queued = n.func.attr == "put_nowait" and n.args
            if queued and _raised_name(n.args[0]) == "SystemExit":
                lines["queued"] = n.lineno
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
    src = (REPO / "atom/compass/runner/model_runner.py").read_text()
    assert "unanswered_rpc_names(CompassModelRunner)" in src
    assert "raise RunnerRefusal(" in src


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
    with pytest.raises(RunnerRefusal, match="memory model"):
        Runner().get_num_blocks()


def test_forward_refuses_and_its_reply_is_an_object_not_a_tuple():
    """What a successor owes: one value, read for its attributes.

    No site unpacks it. Three read it whole and one discards it -- the middle
    pipeline stage, whose tokens go out over the transport rather than back.
    """
    assert {s.arity for s in SITES["forward"]} == {0, 1}
    engine = (ENGINE / "engine_core.py").read_text()
    pp = (ENGINE / "pp_engine_core.py").read_text()
    assert (
        "self.scheduler.postprocess(\n            seqs,\n            fwd_out," in engine
    )
    assert "fwd_out.req_ids" in pp and "send_tokens(fwd_out)" in pp
    with pytest.raises(RunnerRefusal):
        Runner().forward(object())


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


def test_the_two_names_no_caller_waits_for_and_what_replying_costs():
    """A reply nobody reads sits on the queue for whoever asks next.

    `process_kvconnector_output` therefore answers nothing at all. `exit` does
    reply, and is the one place that is harmless: `busy_loop` breaks on the
    name, so there is no next caller.
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


def test_the_zero_block_form_in_the_tree_answers_two_of_the_four_keys():
    """A trap for whoever ends the `get_num_blocks` refusal above.

    `RapidServeModelRunner.get_num_blocks` is the obvious thing to copy -- a
    non-allocating runner answering without a device -- and it returns
    `num_kvcache_blocks` and `state_runtime` only. That is not a breach: the
    caller takes the two pool-entry keys with a default. It is short, and the
    shortfall is invisible at zero blocks and not invisible above them, so the
    count is asserted here rather than left to be noticed.
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
