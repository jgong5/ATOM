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
DP widths, DP attention, expert parallelism, TBO, the MoE backend, the
all-to-all backend, EPLB and DeepSeek-V4.1 decoder SWA bounded replay when
off ATOM's default, the index cache's dtype where ATOM resolved it apart from
the KV cache's, whether mori imports where it does not, MiniMax-M3's
indexer-only context parallelism when on, the model config's
``--hf-overrides`` when it states any, the performance environment, each
variable of `READ_OFF_DEFAULT` in it when ATOM reads it off its default too,
and, under speculation, its method and draft width, and the digest of the
draft's config.json where it is not the target's. Loading refuses, naming
what differs:

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

import functools
import importlib.util
import json
import os
import pickle
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
    "moe_all2all_backend",
    "eplb_enable",
    "enable_decoder_swa_bounded_replay",
    "index_cache_dtype",
)
#: The index cache takes the KV cache's dtype unless ATOM resolves it apart.
DERIVED = {"index_cache_dtype": "kv_cache_dtype"}
#: ATOM's prefill token buckets, an environment variable `keyed` the same way.
BUCKETS = "ATOM_TOKEN_GRAPH_SIZES"
#: ATOM's environment variables keyed only where ATOM reads them off their
#: default: `BUCKETS`, vocabulary-embedding replication (GLM-5.2's and an EAGLE3
#: draft's whole embedding, with no all-reduce after it), and the three that
#: decide, with the all-to-all backend and whether mori imports (`MORI`), whether
#: a MoE fuses its shared expert into aiter's top-k table (`model_ops/topK.py`):
#: two under DP attention, and one for an MXFP8 shared expert beside MXFP4
#: routed ones.
READ_OFF_DEFAULT = (
    BUCKETS,
    "ATOM_REPLICATE_VOCAB_EMBED",
    "ATOM_FUSE_SHARED_EXPERT",
    "ATOM_DISABLE_MORI_EP",
    "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4",
)
#: Whether mori, the all-to-all ATOM selects by default, imports: keyed only
#: where it does not, so keys written where it imports do not move.
MORI = "mori_imports"
#: The quantization config's lists of layer names, which its runner remaps.
LAYERS = ("layer_pattern_specs", "exclude_layers")
LAYERS += tuple(f"online_{name}" for name in LAYERS)
#: The arch `remapped` declares: what it reads is a class constant on any arch.
REMAP_ARCH = "gfx950"
#: The digest of a drafter's config.json, keyed only where it is not the
#: target's, which every key already holds as its model's.
DRAFT = "draft_model"


def keyed(config, ladder: bool = True) -> dict:
    """The conditions a bundle is measured under, as the resolved config holds them,
    `OFF_DEFAULT` only where it is off its default, `MORI` only where mori does
    not import, MiniMax-M3's indexer-only context parallelism only when on, the
    model config's overrides only when it states some, and the performance
    environment of this process less the client's and the device selection
    (`keyed_env`), and under speculation its method and draft width, and
    `DRAFT` (`_draft`); the ladder only with `ladder`."""
    from atom.config import Config

    spec = getattr(config, "speculative_config", None)
    drafts = {"speculative": [spec.method, spec.num_speculative_tokens]} if spec else {}
    drafts |= _draft(config, spec) if spec else {}
    if ladder:
        drafts["ladder"] = [] if config.enforce_eager else schedulable_ladder(config)
    if importlib.util.find_spec("mori") is None:
        drafts[MORI] = False
    for name in OFF_DEFAULT:
        default = Config.__dataclass_fields__[name].default
        if name in DERIVED:
            default = getattr(config, DERIVED[name])
        if (value := getattr(config, name, None)) not in (None, default):
            drafts[name] = value
    if getattr(getattr(config, "dcp_config", None), "indexer_dcp_only", False):
        drafts["indexer_dcp_only"] = True
    # ATOM applies them to the model config after it reads config.json.
    if overrides := getattr(config, "hf_overrides", None):
        drafts["hf_overrides"] = overrides
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


def _draft(config, spec) -> dict:
    """`DRAFT` as the drafter `spec` reads it: the digest of its checkpoint's
    config.json, the target's when it names none, as ATOM loads it, stated only
    where that is not the target's config.json."""
    from atom.compass.run import _file_digest

    draft, target = (
        _file_digest(model, "config.json")
        for model in (getattr(spec, "model", None) or config.model, config.model)
    )
    return {DRAFT: draft} if draft != target else {}


def quantization(config) -> str:
    """The hash of `config`'s quantization config as its runner holds it, its
    layer names remapped to the model's modules as `ModelRunner.__init__` remaps
    them before it builds: by `remapped` where the runner has not, and as it is
    where the packed modules the remap leaves on it say it has, since a second
    remap can rename a name again. A config naming no layer, or with no remap,
    has none to apply."""
    held = config.quant_config
    if (
        hasattr(held, "remap_layer_name")
        and not hasattr(held, "packed_modules_mapping")
        and any(getattr(held, name) for name in LAYERS)
    ):
        return remapped(pickle.dumps((held, config.hf_config)))
    return held.compute_hash()


@functools.cache
def remapped(pickled: bytes) -> str:
    """The hash of the quantization config `pickled` with its model's config,
    remapped as `ModelRunner.__init__` remaps it, in a process of its own: the
    model class's module imports aiter, whose probes need a device, here
    `REMAP_ARCH`, and a class may hold its packed modules in a property."""
    from atom.compass.bench.ops import _child

    return _child(_remapped, pickled)


def _remapped(pickled: bytes) -> str:
    from atom.compass.capture import driverless

    with tempfile.TemporaryDirectory(prefix="compass-remap-") as tmpdir:
        driverless.install(tmpdir, REMAP_ARCH)  # before anything imports aiter
        quant, hf_config = pickle.loads(pickled)
        from atom.model_engine.model_runner import support_model_arch_dict
        from atom.utils import resolve_obj_by_qualname

        arch = hf_config.architectures[0]
        cls = resolve_obj_by_qualname(support_model_arch_dict[arch])
    names = ("packed_modules_mapping", "quant_exclude_name_mapping")
    quant.remap_layer_name(hf_config, **{n: getattr(cls, n, {}) for n in names})
    return quant.compute_hash()


def keyed_env() -> dict:
    """This process's performance environment as `keyed` states it: less the
    client's variables and the device selection, each of `READ_OFF_DEFAULT` in
    it only where ATOM reads it off its default."""
    return {
        k: v
        for k, v in sorted((performance_env() | _off_default()).items())
        if not (k.startswith("AIPERF_") or k.endswith("_VISIBLE_DEVICES"))
    }


def _off_default() -> dict:
    """Each of `READ_OFF_DEFAULT` as this process sets it where ATOM reads it off
    its default: ATOM states each default only inside its reader, run here unset."""
    from atom.utils import envs

    unset = SimpleNamespace(getenv=lambda _, default=None: default)
    readers = {name: envs.environment_variables[name] for name in READ_OFF_DEFAULT}
    return {
        name: os.environ[name]
        for name, read in readers.items()
        if read() != FunctionType(read.__code__, read.__globals__ | {"os": unset})()
    }


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
