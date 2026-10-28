# SPDX-License-Identifier: MIT
"""The run file a simulated run hands ATOM, built from its command's flags.

`add_arguments` adds the flags `trace steps` and `run` share, and `build`
resolves them, store keys included, into the run file's mapping; the command
then sets ``out_dir`` and ``clock_endpoint`` and hands it to the server.

- The backend, exactly one, each flag named by its `run` run-file key:
  ``--step-stub FILE``, the stated `Coefficients`; ``--step-linear KEY``, a
  `step_linear_law` entry's params with its key's ``engine_config``, the entry
  loaded through `ArtifactStore.load` under `law_readings`;
  ``--step-grid KEY``, the store's root and a `step_grid_bundle` entry's key;
  ``--op-graph``, ``true``, and as ``price_list`` the fields of the
  `price_list` key this run's model, parallel layout, machine spec and ATOM
  tree name, when ``--store`` holds one there.
  A KEY is the token `Key.token` prints, resolved under ``--store``.
  ``backend`` holds the backend's ``digest`` and, for a store entry, its
  ``provenance``, with its report's as ``inputs``; a stub's, and an op graph's
  with no price list, is None, its digest over what the run file states.
- ``--host-laws KEY``, a `host_laws` entry's params, as ``host``; without it
  the run charges no host time.
- ``--step-linear`` and ``--host-laws`` repeat, one entry per engine config:
  the key then holds a list, each process installing its own config's entry,
  and ``backend``'s digest is over every law's in any order, their provenance
  ``entries``.
- ``--machine``, the spec; ``--store``'s root as ``store``; ``--on-refusal``;
  ``--step-roofline``; ``--padding none``, which prices a replayed graph's
  real rows.
- Not flags: `FIXED`, and ``data_parallel_size``, read from ATOM's arguments.
  The server serves client runs until it is stopped.

`prefill_decode` builds the one run file of a prefill and a decode instance,
which `scripts/compass/deploy.sh` hands its clock authority and both instances.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from atom.compass import run
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Reading,
    git_tree_root,
    machine_spec_digest,
    member_name,
    module_root,
)
from atom.compass.fit.publish import RANK
from atom.compass.run.step_grid import readings

#: Each backend flag's store kind; the step stub's coefficients are a file.
KINDS = {run.STEP_LINEAR: Kind.STEP_LINEAR_LAW, run.STEP_GRID: Kind.STEP_GRID_BUNDLE}
#: The API server's admission path, the lookaheads no measurement backs
#: (`single_engine_table`), and a simulated-time bound past any run's.
FIXED = {
    "admission_path": "serving",
    "ipc_s": 2.0**-14,
    "stream_s": 2.0**-12,
    "bound_s": 1e7,
}
#: A prefill-decode run's write-request lookahead, which no measurement backs.
KV_WRITE_REQ_S = 2.0**-10


def add_arguments(parser) -> None:
    """Add the run file's flags to `parser`."""
    backend = parser.add_mutually_exclusive_group(required=True)
    for key in (run.STEP_STUB, *KINDS):
        flag, stub = "--" + key.replace("_", "-"), key == run.STEP_STUB
        many = {"action": "append"} if key == run.STEP_LINEAR else {}
        backend.add_argument(flag, dest=key, metavar="FILE" if stub else "KEY", **many)
    backend.add_argument("--op-graph", dest=run.OP_GRAPH, action="store_true")
    parser.add_argument(
        "--host-laws", metavar="KEY", action="append", help="a host_laws token"
    )
    parser.add_argument("--machine", required=True, help="the machine spec")
    parser.add_argument("--store", required=True, help="the artifact store's root")
    parser.add_argument("--on-refusal", choices=("mark", "abort"), default="mark")
    parser.add_argument("--step-roofline", action="store_true")
    parser.add_argument("--padding", choices=("none",))


def build(args, atom_args) -> dict:
    """The run file of `args`, parsed with `add_arguments`, for a server
    started with ATOM's arguments `atom_args`."""
    store = ArtifactStore(Path(args.store).resolve())
    out = {
        **FIXED,
        "client_runs": None,
        "data_parallel_size": _dp_width(atom_args),
        "machine": json.loads(Path(args.machine).read_text()),
        "store": str(store.root),
        "on_refusal": args.on_refusal,
        "step_roofline": args.step_roofline,
    }
    # Read as the runner and `compass fit` read it, so a spec they would
    # refuse refuses before a server starts.
    machine = run._machine(out)
    if stub := getattr(args, run.STEP_STUB):
        out[run.STEP_STUB] = json.loads(Path(stub).read_text())
        out["backend"] = _stated(out[run.STEP_STUB])
    elif getattr(args, run.OP_GRAPH):
        out[run.OP_GRAPH] = True
        key = _price_list_key(atom_args, machine)
        if store.directory_for(key).exists():
            out[run.PRICE_LIST] = key.values
            out["backend"] = _backend(store.read(key))
        else:
            out["backend"] = _stated({run.OP_GRAPH: True})
    else:
        name = next(k for k in KINDS if getattr(args, k))
        tokens = getattr(args, name)
        tokens = [tokens] if isinstance(tokens, str) else tokens
        resolved = [resolve(store, name, t, machine, atom_args) for t in tokens]
        laws = [law for law, _ in resolved]
        backends = [_backend(entry) for _, entry in resolved]
        out[name], out["backend"] = laws[0], backends[0]
        if len(laws) > 1:
            digests = " ".join(sorted(b["digest"] for b in backends)).encode()
            out[name], out["backend"] = laws, {
                "digest": "sha256:" + hashlib.sha256(digests).hexdigest(),
                "provenance": {
                    "entries": [b["provenance"] for b in backends],
                    "inputs": [
                        i for b in backends for i in b["provenance"]["inputs"] or ()
                    ],
                },
            }
    if args.host_laws:
        hosts = [
            resolve(store, "host", k, machine, atom_args)[0] for k in args.host_laws
        ]
        out["host"] = hosts if len(hosts) > 1 else hosts[0]
    if args.padding:
        out["padding"] = args.padding
    return out


def resolve(store, name: str, token: str, machine, atom_args) -> tuple:
    """The run file's `name` value for the entry `token` names in `store`, a
    key of `KINDS` or ``host``, and that entry: a bundle's key and host laws'
    params read, a step linear law's params with its key's engine config
    loaded under `law_readings`."""
    key = store.key_from_token(KINDS.get(name, Kind.HOST_LAWS), token)
    if name == run.STEP_LINEAR:
        conditions = law_readings(key, machine, atom_args)
        entry = store.load(key, conditions=conditions, gates=GateState.of())
        engine_config = json.loads(key.values["engine_config"])
        return {**_member(entry, "params"), "engine_config": engine_config}, entry
    entry = store.read(key)
    if name == run.STEP_GRID:
        return {"store": str(store.root), **key.values}, entry
    return _member(entry, "params"), entry


def prefill_decode(args, prefill_args, decode_args, kv_link) -> dict:
    """The run file `build` builds from `args` for one prefill and one decode
    instance, started with ATOM's arguments `prefill_args` and `decode_args`:
    each side's DP width read from its own, the router's per-request forward
    cost the machine spec's ``interconnect.router_relay_s``, the KV transfers
    priced on the spec's `kv_link`, ``intra_node`` for two instances on one node
    and ``inter_node`` across two, and `KV_WRITE_REQ_S`. A price list the
    prefill side names that is not the decode side's refuses."""
    out = build(args, prefill_args)
    if run.PRICE_LIST in out:
        decode = _price_list_key(decode_args, run._machine(out)).values
        if decode != out[run.PRICE_LIST]:
            raise ValueError(
                f"--op-graph: the prefill side's price list is {out[run.PRICE_LIST]} "
                f"and the decode side's {decode}; one run file names one"
            )
    del out["data_parallel_size"]
    out[run.DP_WIDTHS["P"]] = _dp_width(prefill_args)
    out[run.DP_WIDTHS["D"]] = _dp_width(decode_args)
    return out | {
        "router_s": run._machine(out).value("interconnect.router_relay_s"),
        "kv_write_req_s": KV_WRITE_REQ_S,
        "kv_link": kv_link,
    }


def law_readings(key: Key, machine, atom_args):
    """The readings a run on `machine` from this ATOM tree, serving the model
    ATOM's arguments `atom_args` name, loads the `step_linear_law` at `key`
    under: its engine config as `key` states it, the one the runner then holds
    the resolved config to."""
    return readings(
        machine,
        atom_src=Reading.of_source_root(git_tree_root("atom", module_root("atom"))),
        model=_model_digest(_engine_args(atom_args).model),
        engine_config=key.values["engine_config"],
    )


def _model_digest(model: str) -> str:
    """`run._file_digest` of the config.json of the model `model` names, found
    or fetched as ATOM's loader finds it, so a hub id with no local copy reads
    the one its server will load; one it cannot find or fetch refuses."""
    from transformers.utils import cached_file

    try:
        why = cached_file(model, "config.json") is None and "there is none"
    except OSError as error:
        why = error
    if why:
        raise ValueError(
            f"--model {model}: its config.json is neither local nor fetchable: {why}"
        )
    return run._file_digest(model, "config.json")


def _dp_width(atom_args) -> int:
    """The DP ranks ATOM's arguments run: under DP-attention every TP rank is
    one, as `run.frontend` counts."""
    engine = _engine_args(atom_args)
    ranks = engine.tensor_parallel_size if engine.enable_dp_attention else 1
    return engine.data_parallel_size * ranks


def _stated(value) -> dict:
    """The ``backend`` of a backend the run file states whole as `value`."""
    stated = json.dumps(value, sort_keys=True).encode()
    return {
        "digest": "sha256:" + hashlib.sha256(stated).hexdigest(),
        "provenance": None,
    }


def _backend(entry) -> dict:
    """The ``backend`` of the store entry `entry`: its digest and provenance,
    with its report's provenance as ``inputs``."""
    reported = member_name("report", RANK, "json") in entry.members
    inputs = _member(entry, "report").get("provenance") if reported else None
    provenance = {**entry.provenance.as_json(), "inputs": inputs}
    return {"digest": entry.digest, "provenance": provenance}


def _price_list_key(atom_args, machine) -> Key:
    """The `price_list` key of the model and parallel layout ATOM's arguments
    serve, on `machine`, from this ATOM tree."""
    engine = _engine_args(atom_args)
    tree = git_tree_root("atom", module_root("atom"))
    return Key.of(
        Kind.PRICE_LIST,
        model=_model_digest(engine.model),
        dp=engine.data_parallel_size,
        pp=engine.pipeline_parallel_size,
        pcp=engine.prefill_context_parallel_size,
        tp=engine.tensor_parallel_size,
        enable_expert_parallel=engine.enable_expert_parallel,
        enable_dp_attention=engine.enable_dp_attention,
        decode_context_parallel_size=engine.decode_context_parallel_size,
        machine_spec=machine_spec_digest(machine),
        source_root=f"{tree.revision_kind}-{tree.revision}",
    )


def _member(entry, stem: str):
    """`entry`'s `stem` member, as `fit` publishes one, read as JSON."""
    return json.loads(entry.read_member(stem, RANK, "json"))


def _engine_args(atom_args):
    """ATOM's arguments as its own parser reads them."""
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    return parser.parse_known_args(atom_args)[0]


def pd_side(atom_args, refuse: bool = False) -> str | None:
    """``P`` or ``D`` for a prefill or a decode instance, else None; `refuse` raises."""
    transfer = _engine_args(atom_args).kv_transfer_config
    side = run._pd_side(SimpleNamespace(kv_transfer_config=json.loads(transfer)))
    if refuse and side is not None:
        raise ValueError(
            f"--kv-transfer-config {transfer} makes this a "
            f"{'prefill' if side == 'P' else 'decode'} instance, which shares a clock "
            "with its peer: start it with compass run --join, as deploy.sh does"
        )
    return side


def engine_only(atom_args) -> list[str]:
    """ATOM's arguments `atom_args` less the server's, which a build that
    parses ATOM's engine arguments alone refuses."""
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(add_help=False))
    left, kept = iter(parser.parse_known_args(atom_args)[1]), []
    drop = next(left, None)
    for arg in atom_args:
        if arg == drop:
            drop = next(left, None)
        else:
            kept.append(arg)
    return kept
