# SPDX-License-Identifier: MIT
"""The step grid in a run: its bundle loaded from the artifact store by key, and
checked against the resolved ATOM config before the first step.

The run file's ``step_grid`` is the store's root and the bundle's key fields:

    {"store": <root directory>, "model": ..., "engine_config": ..., "grid": ...,
     "source_root": ..., "device": ..., "software_stack": ..., "ptl": ...,
     "torch": ...}

A bundle prices a TP group's whole step, so a run reads one member at any
width, ``params.dp0of1.pp0of1.pcp0of1.tp0of1.json``, the one `compass fit
steps` publishes beside its report:

    {"grid": <a StepGrid mapping>, "residual": <a Residual mapping>,
     "bounds": {"max_num_seqs": n, "max_num_batched_tokens": n,
                "max_model_len": n},
     "draft_residual": <a Residual mapping>}     # under a drafter

A grid that states a draft width prices each step's drafter too, read for its
points' drafter values with `draft_residual`.

The grid states its timing scope (`scope`) and what its points hold
(`value`); `step_grid_fit.bundle`, which writes a bundle, also writes the grid
definition and the residual's sources, which loading does not read.

Its publisher states the readings `conditions` takes. The engine-config reading
is `keyed`, as JSON: the ladder the runner replays (none under
``enforce_eager``), the KV block size and dtype, the quantization with its
layer names as the model's runner remaps them (`quantization`), the TP and
DP widths, DP attention, expert parallelism, TBO, the MoE backend, EPLB and
DeepSeek-V4.1 decoder SWA bounded replay when
off ATOM's default, the index cache's dtype where ATOM resolved it apart from
the KV cache's, the performance environment, ATOM's prefill token buckets
in it when off ATOM's default too, and, under speculation, its
method and draft width. Loading refuses, naming what differs:

- a run at data-parallel width above one, since every bundle is measured at one;
- a keyed condition, the grid's graph mode, a timing scope other than the
  whole ``ModelRunner.forward``, or a value other than ``measure steps``'
  bench window (`step_grid_fit.VALUE`), none included;
- the gates, or any reading, through `ArtifactStore.load`;
- a rung of the ladder between the grid's lowest and highest measured rungs
  that was not measured;
- a configured bound above the one the grid was planned for.

Under expert parallelism the bench routed every MoE layer by a uniform draw
(`bench/routing.py`), so every step the loaded backend prices from its grid
names uniform routing among its assumed treatments; without it, none.

A decode corner past the KV pool the grid records is extrapolated at lookup,
and any other unmeasured corner refuses there, so neither needs a check here;
the run composes the step roofline below the backend for the steps it refuses.
The residual's recipe reads no chunk size, so none is keyed.

Grading a run reads its bundle with `read` instead, checking nothing against
the grading process: the run checked it against its own as it started, and its
summary records the digest of the bundle it priced with.
"""

import copy
import functools
import json
import os
import tempfile
from collections.abc import Mapping
from types import FunctionType, SimpleNamespace

from atom.benchmarks.results.metadata import performance_env
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    Conditions,
    GateState,
    Key,
    Kind,
    Row,
    Topology,
    machine_platform,
)
from atom.compass.backends.step_grid import StepGrid, StepGridBackend
from atom.compass.backends.step_grid_fit import VALUE
from atom.compass.backends.step_grid_plan import BOUNDS, SCOPE, keyed_of  # noqa: F401
from atom.compass.backends.step_grid_residual import Residual
from atom.compass.backends.step_linear import UNIFORM_ROUTING
from atom.compass.runner.overrides import schedulable_ladder

#: The bundle's member a run reads: a group's step has a single writer.
MEMBER = ("params", Topology().ranks()[0], "json")
#: Engine fields keyed only off their default, so keys written before they
#: were read do not move: ATOM's, or for a field in `DERIVED` the config's own
#: value of the field it names there. None, a field left unresolved, is default.
OFF_DEFAULT = (
    "moe_backend",
    "eplb_enable",
    "enable_decoder_swa_bounded_replay",
    "index_cache_dtype",
)
#: The index cache takes the KV cache's dtype unless ATOM resolves it apart.
DERIVED = {"index_cache_dtype": "kv_cache_dtype"}
#: ATOM's prefill token buckets, an environment variable `keyed` the same way.
BUCKETS = "ATOM_TOKEN_GRAPH_SIZES"
#: The quantization config's lists of layer names, which its runner remaps.
LAYERS = ("layer_pattern_specs", "exclude_layers")
LAYERS += tuple(f"online_{name}" for name in LAYERS)
#: The arch `remapping` declares: what it reads is a class constant on any arch.
REMAP_ARCH = "gfx950"


def keyed(config, ladder: bool = True) -> dict:
    """The conditions a bundle is measured under, as the resolved config holds them,
    `OFF_DEFAULT` only where it is off its default, and the performance
    environment of this process less the client's and the device selection,
    `BUCKETS` in it only off ATOM's default, and under speculation its method
    and draft width; the ladder only with `ladder`."""
    from atom.config import Config

    spec = getattr(config, "speculative_config", None)
    drafts = {"speculative": [spec.method, spec.num_speculative_tokens]} if spec else {}
    if ladder:
        drafts["ladder"] = [] if config.enforce_eager else schedulable_ladder(config)
    for name in OFF_DEFAULT:
        default = Config.__dataclass_fields__[name].default
        if name in DERIVED:
            default = getattr(config, DERIVED[name])
        if (value := getattr(config, name, None)) not in (None, default):
            drafts[name] = value
    return drafts | {
        "kv_cache_block_size": config.kv_cache_block_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "quantization": quantization(config),
        "tensor_parallel_size": config.tensor_parallel_size,
        "data_parallel_size": config.parallel_config.data_parallel_size,
        "enable_dp_attention": config.enable_dp_attention,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_tbo": config.enable_tbo,
        "enable_tbo_decode": config.enable_tbo_decode,
        "env": keyed_env(),
    }


def quantization(config) -> str:
    """The hash of `config`'s quantization config as its runner holds it, its
    layer names remapped to the model's modules as `ModelRunner.__init__` remaps
    them before it builds: on a copy where the runner has not, and as it is
    where the packed modules the remap leaves on it say it has, since a second
    remap can rename a name again. A config naming no layer has none to remap."""
    from atom.config import QuantizationConfig

    held = config.quant_config
    if (
        isinstance(held, QuantizationConfig)
        and not hasattr(held, "packed_modules_mapping")
        and any(getattr(held, name) for name in LAYERS)
    ):
        held = copy.deepcopy(held)
        architecture = config.hf_config.architectures[0]
        held.remap_layer_name(config.hf_config, **remapping(architecture))
    return held.compute_hash()


@functools.cache
def remapping(architecture: str) -> dict:
    """The keywords `ModelRunner.__init__` remaps with for the model class of
    `architecture`, read in a process of its own: the class's module imports
    aiter, whose probes need a device, here `REMAP_ARCH`."""
    from atom.compass.bench.ops import _child

    return _child(_remapping, architecture)


def _remapping(architecture: str) -> dict:
    from atom.compass.capture import driverless

    with tempfile.TemporaryDirectory(prefix="compass-remap-") as tmpdir:
        driverless.install(tmpdir, REMAP_ARCH)
        from atom.model_engine.model_runner import support_model_arch_dict
        from atom.utils import resolve_obj_by_qualname

        cls = resolve_obj_by_qualname(support_model_arch_dict[architecture])
    names = ("packed_modules_mapping", "quant_exclude_name_mapping")
    return {name: getattr(cls, name, {}) for name in names}


def keyed_env() -> dict:
    """This process's performance environment as `keyed` states it: less the
    client's variables and the device selection, `BUCKETS` in it only off
    ATOM's default."""
    return {
        k: v
        for k, v in sorted((performance_env() | _buckets_off_default()).items())
        if not (k.startswith("AIPERF_") or k.endswith("_VISIBLE_DEVICES"))
    }


def _buckets_off_default() -> dict:
    """`BUCKETS` as this process sets it when ATOM reads it off its default, else
    nothing: ATOM states the default only inside its reader, run here unset."""
    from atom.utils import envs

    read = envs.environment_variables[BUCKETS]
    unset = SimpleNamespace(getenv=lambda _, default=None: default)
    off = read() != FunctionType(read.__code__, read.__globals__ | {"os": unset})()
    return {BUCKETS: os.environ[BUCKETS]} if off else {}


def differ(there: dict, here: dict, where: str) -> list[str]:
    """Each field of two `keyed` engine configs that differs, with `there`'s
    value as `where` states it and `here`'s, a field both state as a mapping by
    each of its entries, so a performance variable by its own name."""
    named = []
    for name in {**there, **here}:
        a, b = there.get(name), here.get(name)
        if isinstance(a, dict) and isinstance(b, dict):
            named += differ(
                *({f"{name} {k}": v for k, v in c.items()} for c in (a, b)), where
            )
        elif a != b:
            named.append(f"{name} {a!r} in {where} and {b!r} here")
    return named


def conditions(config, machine) -> Conditions:
    """This run's readings: `readings` of this process's ATOM source, the
    served config.json and `keyed`."""
    from atom.compass.run import _file_digest, atom_source

    return readings(
        machine,
        atom_src=atom_source(),
        model=_file_digest(config.model, "config.json"),
        engine_config=json.dumps(keyed(config), sort_keys=True),
    )


def readings(machine, **inputs) -> Conditions:
    """The readings a run on `machine` takes: the platform the spec states, and
    this host's CPU; `inputs` state the ATOM source, the model and the engine
    config."""
    from atom.compass.run import host_cpu

    return Conditions.of(
        **machine_platform(machine),
        host_cpu=host_cpu(),
        **inputs,
    )


def load(spec: Mapping, config, machine) -> tuple[StepGridBackend, dict]:
    """The backend the bundle at the run file's key prices with, and the
    bundle's key and digest for the run summary; or a refusal naming what
    differs."""
    dp = config.parallel_config.data_parallel_size
    if dp != 1:
        raise ValueError(
            "a step grid bundle is measured at data-parallel width 1, and this "
            f"run has data_parallel_size {dp}"
        )
    fields = dict(spec)
    store = ArtifactStore(fields.pop("store"))
    key = Key.of(Kind.STEP_GRID_BUNDLE, **fields)
    entry = store.read(key)
    ours = keyed(config)
    treatments = (UNIFORM_ROUTING,) if ours["enable_expert_parallel"] else ()
    try:
        member = json.loads(entry.read_member(*MEMBER))
        backend = _backend(member, str(key), treatments)
        grid = backend.grid
        bounds = {name: member["bounds"][name] for name in BOUNDS}
        row = dict(entry.fingerprints[Row.STEP_GRID_BUNDLE].cells)
        recorded = json.loads(row[Axis.ENGINE_CONFIG].value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{key} does not load: {type(error).__name__}: {error}"
        ) from error
    pairs = {
        "graph mode": (grid.graph_mode, "eager" if config.enforce_eager else "replay"),
        "timing scope": (grid.scope, SCOPE),
        "value": (member["grid"].get("value"), VALUE),
        **{name: (recorded.get(name), value) for name, value in ours.items()},
    }
    differ = [
        f"{name} {theirs!r} in the bundle and {here!r} here"
        for name, (theirs, here) in pairs.items()
        if theirs != here
    ]
    if differ:
        raise ValueError(
            f"{key} was measured under other conditions: " + "; ".join(differ)
        )
    store.load(key, conditions=conditions(config, machine), gates=GateState.of())
    if grid.graph_mode == "replay":
        measured = sorted(rung for rung, cells in grid.decode.items() if cells.points)
        unmeasured = [
            r
            for r in ours["ladder"]
            if measured and measured[0] < r < measured[-1] and r not in measured
        ]
        if unmeasured:
            raise ValueError(
                f"{key} measured the decode rungs {measured}, and not the "
                f"ladder's rungs {unmeasured} between them"
            )
    over = [
        f"{name} {getattr(config, name)} is above the {bounds[name]}"
        for name in BOUNDS
        if getattr(config, name) > bounds[name]
    ]
    if over:
        raise ValueError(
            f"this run's {'; '.join(over)} the grid of {key} was planned for"
        )
    return backend, {"key": key.values, "digest": entry.digest}


def _backend(member: Mapping, bundle: str, treatments=()) -> StepGridBackend:
    """The backend of a bundle's member, under a drafter with the grid read for
    the drafter's value and its residual, naming `treatments` on its steps."""
    grid, residual = StepGrid(member["grid"]), Residual(member["residual"])
    drafter = None
    if grid.draft_width:
        drafter = (
            StepGrid(member["grid"], drafter=True),
            Residual(member["draft_residual"], drafter=True),
        )
    return StepGridBackend(grid, residual, bundle, drafter, treatments)


def read(spec: Mapping, digest: str) -> StepGridBackend:
    """The backend of the bundle a run priced with: the entry in the store the
    run file names that `ArtifactStore.read_recorded` finds by the run file's
    key and `digest`, the one the run summary records, and its member read back
    at the digest the entry states. Refused, naming the key, when the store
    holds no such entry or its member is missing or has changed."""
    fields = dict(spec)
    store = ArtifactStore(fields.pop("store"))
    entry = store.read_recorded(Kind.STEP_GRID_BUNDLE, fields, digest)
    return _backend(json.loads(entry.read_member(*MEMBER)), str(entry.key))
