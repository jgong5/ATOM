# SPDX-License-Identifier: MIT
"""The runner subclass, and the construction that owns no device memory.

These run without a driver, which is the reason the behaviour lives in
`atom.compass.runner.overrides` rather than beside the subclass: importing
`atom.model_engine.model_runner` runs aiter's architecture probe, which shells
out to `rocminfo` and raises where there is no GPU. So the bodies are exercised
directly and the binding to ATOM's class is read from the source instead.

The two facts worth naming, because both were surprises:

* The base class warms the model from inside its own `__init__`, and warmup
  drives a forward. A runner with no weights therefore cannot construct unless
  it skips warmup -- the forward that warmup would drive is its own. That chain
  is asserted over ATOM's source below, so this stops being true loudly rather
  than quietly.
* Allocation is checked by counting dispatched operators, not by reading a CUDA
  allocator, so the check is meaningful on a machine with no CUDA allocator to
  read. A control asserts the counter sees an allocation when there is one;
  without it a broken counter and a clean runner look identical.
"""

import ast
import dataclasses
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from atom.compass.runner import COMPASS_RUNNER_QUALNAME
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    UnbuiltModel,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / "atom" / "compass" / "runner"
ENGINE = REPO / "atom" / "model_engine"
ATOM_RUNNER = ENGINE / "model_runner.py"

# The methods that own memory or run a step, and so are the ones replaced.
OVERRIDDEN = {
    "_build_and_load_model",
    "_maybe_warmup",
    "get_num_blocks",
    "allocate_kv_cache",
    "capture_cudagraph",
    "forward",
}


def _classes(path):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _methods(node):
    return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}


def _self_calls(node):
    """Names of `self.x(...)` calls anywhere inside a function definition."""
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    }


class Recorder(TorchDispatchMode):
    """Every operator torch dispatches while this is entered."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


class Runner(NonAllocatingRunner):
    """The overrides over a base that supplies only what they read."""

    def __init__(self):
        self.config = SimpleNamespace(num_kvcache_blocks=None)


class SomeModel:
    """Stands in for the model class the engine hands the runner."""


# --- what is replaced, and how that differs from the runner already in tree ---


def test_the_replaced_methods_are_the_ones_that_own_memory_or_run_a_step():
    assert _methods(_classes(PACKAGE / "overrides.py")["NonAllocatingRunner"]) == (
        OVERRIDDEN
    )


def test_every_replaced_method_exists_on_the_class_being_replaced():
    """A rename upstream turns an override into a new method, silently."""
    assert OVERRIDDEN <= _methods(_classes(ATOM_RUNNER)["ModelRunner"])


def test_the_difference_from_the_in_tree_non_allocating_runner_is_three_methods():
    """`RapidServeModelRunner` is the working non-allocating runner in the tree.

    It overrides two things this one does not. `__init__`: it has to bind a
    method before the base runs, and this class has nothing to bind, so leaving
    `__init__` alone is what keeps every read lazy. `_kv_budget_extra_reserve`:
    it holds bytes back because a second process shares its GPU, and a runner
    that allocates nothing has no tenant to hold anything back from.

    This one overrides one thing it does not: `capture_cudagraph`. RapidServe
    allocates no weights of its own but imports real ones over CUDA IPC, so it
    has a model to trace and keeps ATOM's capture. This runner has none, and
    ATOM's capture zeroes device buffers and opens a graph pool before it finds
    that out.

    Its `_init_weight_params_on_meta` is not in the difference because it is not
    an override -- it is a helper the base does not have.
    """
    base = _methods(_classes(ATOM_RUNNER)["ModelRunner"])
    template = _methods(_classes(ATOM_RUNNER)["RapidServeModelRunner"]) & base
    assert template - OVERRIDDEN == {"__init__", "_kv_budget_extra_reserve"}
    assert OVERRIDDEN - template == {"capture_cudagraph"}


# --- construction allocates nothing -----------------------------------------


def test_construction_dispatches_no_operator_at_all():
    """The measured result: zero, against a base that reads a checkpoint."""
    runner = Runner()
    with Recorder() as recorded:
        runner._build_and_load_model(SomeModel)
        runner._maybe_warmup()
        runner.allocate_kv_cache(2048)
    assert recorded.ops == []


def test_the_recorder_sees_an_allocation_when_there_is_one():
    """Otherwise a dead recorder reads exactly like a clean runner."""
    with Recorder() as recorded:
        torch.empty(4)
    assert recorded.ops


def test_the_model_that_replaces_the_weights_owns_nothing():
    runner = Runner()
    runner._build_and_load_model(SomeModel)
    assert isinstance(runner.model, UnbuiltModel)
    assert list(runner.model.parameters()) == []
    assert list(runner.model.buffers()) == []


def test_calling_that_model_refuses_and_names_the_class_it_stands_for():
    with pytest.raises(RunnerRefusal, match="SomeModel"):
        UnbuiltModel(SomeModel)(torch.zeros(1))


# --- warmup drives a forward, which is why it is skipped ---------------------


def test_the_base_class_warms_the_model_from_init_and_warmup_runs_a_forward():
    """The chain that decides whether a weightless runner can construct."""
    runner = _classes(ATOM_RUNNER)["ModelRunner"]
    bodies = {n.name: n for n in runner.body if isinstance(n, ast.FunctionDef)}
    assert "_maybe_warmup" in _self_calls(bodies["__init__"])
    assert "warmup_model" in _self_calls(bodies["_maybe_warmup"])
    assert "forward" in _self_calls(bodies["warmup_model"])


def test_skipping_warmup_is_what_lets_construction_finish():
    """`forward` refuses, so a warmup that ran one would raise out of `__init__`."""
    runner = Runner()
    assert runner._maybe_warmup() is None
    with pytest.raises(RunnerRefusal):
        runner.forward(object())


# --- the two answers that are refused, and the one that is arithmetic --------


def test_sizing_the_kv_pool_refuses_rather_than_inventing_a_block_count():
    with pytest.raises(RunnerRefusal, match="memory model"):
        Runner().get_num_blocks()


def test_allocating_the_kv_cache_records_the_count_and_no_bytes():
    runner = Runner()
    assert runner.allocate_kv_cache(2048) is True
    assert runner.config.num_kvcache_blocks == 2048


# --- the seam: reachable without changing ATOM -------------------------------


def test_the_qualname_names_a_class_this_package_defines():
    module, _, name = COMPASS_RUNNER_QUALNAME.rpartition(".")
    path = REPO.joinpath(*module.split(".")).with_suffix(".py")
    assert name in _classes(path)


def test_selecting_a_runner_is_a_config_field_that_already_exists():
    from atom.config import Config

    field = {f.name: f for f in dataclasses.fields(Config)}["runner_qualname"]
    assert field.default == "atom.model_engine.model_runner.ModelRunner"


def test_the_worker_process_instantiates_whatever_that_field_names():
    assert "config.runner_qualname" in (ENGINE / "engine_core.py").read_text()
    assert (
        "resolve_obj_by_qualname(runner_qualname)"
        in (ENGINE / "async_proc.py").read_text()
    )


@pytest.mark.parametrize(
    "path",
    sorted(PACKAGE.rglob("*.py")),
    ids=lambda p: p.name,
)
def test_only_the_binding_module_reaches_the_engine(path):
    """Everything else stays runnable where the engine cannot be imported."""
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    engine = {m for m in imported if m.split(".")[0] == "atom"} - {
        m for m in imported if m.startswith("atom.compass.runner")
    }
    assert engine == (
        {"atom.model_engine.model_runner"} if path.name == "model_runner.py" else set()
    )
