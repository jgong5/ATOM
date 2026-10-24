# SPDX-License-Identifier: MIT
"""`compass measure ops`: each leaf of an `op_graph` timed alone on one device, cold.

    compass measure ops (--width W | --prefill T B)... [--context C] --llc-mib M
        --gpu G --store S --out DIR -- <ATOM args>
    compass measure ops --plan RECORD [--plan RECORD]... --llc-mib M --gpu G
        --store S --out DIR -- <ATOM args>
    compass measure ops --coverage RECORD [--coverage RECORD]... --store S
        --out DIR -- <ATOM args>

Each shape names the `op_graph` `compass trace ops` published in S for the
model and TP width ATOM's arguments resolve. A point is one leaf of that graph,
its instances grouped (`leaves`), at the shape and `--context`: a decode
step's rows each hold `--context` tokens, and a prefill's rows, split as the
capture split them, follow `--context` cached tokens each, none when it is not
given. Rows the KV pool cannot hold at once are ragged (`point_rows`), and the
point records its rows.

ATOM's runner is built from ATOM's arguments on each GPU `--gpu` names, one
rank each of the layout the arguments resolve (`ranks`), its parameters on the
meta device and with no warmup (`_runner`): no forward of the model, its LM
head or its sampler runs. Each point reaches the runner as a batch
(`ReplayRule`) that `ModelRunner.prepare_model` prepares, so ATOM builds its
metadata and forward context; for DeepSeek-V4, the CSA top-k section its
forward writes is written from ids spread over each token's compressed rows
(`Stub._top_k`). A `Stub` calls each leaf from its contract:
an activation is random at its shape, dtype and stride, except a router's
top-k expert ids, drawn uniformly on every copy, past top-k the shared
experts' ids ATOM fuses into them (`_route`), and the `NM_OUTPUT` slot at
one KV split, a view at the start of twice its span; a weight is random at its
shape, dtype and stride, one tensor per occurrence of the leaf in the step;
a metadata tensor is a
view of ATOM's live metadata (`meta`), whose name the point records, or of
the token ids ATOM prepared (`TOKEN_IDS`), or of the field of DeepSeek-V4's
HCA decode workspace an `HCA_DECODE` slot names, or of the buffer
`V4_METADATA` names for a DeepSeek-V4 slot, refused when it is not live; an
integer the capture
found affine in a decode step's context is taken at the point's; a module or
context key is the one each instance names, its module's meta parameters
moved to the device and filled as `--load_dummy xavier` fills them (`_fill`),
and a runner attribute this runner's own; a KV tensor is a view of a tensor
ATOM's KV allocation bound of its layout (`_kv`), refused when none is, whose
name the point records, the same on every copy; an object is rebuilt from its class
and state as the capture's replay rebuilds one, its tensors built as any other
argument's, its activations fresh on every copy. A leaf whose operator cannot
run so refuses by name. On one GPU at a
DP-attention layout, the runner is one DP rank's own, TP 1 and DP 1, the
graphs read and the keys written keep the layout, and a collective leaf
refuses: it is measured across the layout's ranks.
A collective leaf is one family over its message bytes, decode and prefill
alike (`exchange`), which the plan ladders, densifies and covers.

Cold, by copies. A call's bytes are its tensor arguments' spans, a KV
argument's as many entries of its first dimension, each the span of the
others, as the batch's state slots, or its blocks when the model keeps no
per-request state, or, of a decode attention leaf that states its layer's
window and compress ratio, the plane rows the batch's blocks hold at that
ratio, a RoPE table's (`TABLES`) as many rows as the point's tokens, the
tensors of each module it names, of one whose first dimension is the pool's
blocks the batch's blocks, and, for a leaf that names a context key, the KV and
per-request state its layer holds for the batch. One graph runs the leaf K
times, K `CALLS` or the calls that fill `GRAPH_NS` at the device time of the
kernels of one eager call of copy 0, the fewer, at least one, cycling over its
copies: N built ones: M, the least count whose bytes pass `COLD` times the
last-level cache, raised for a context key to its I instances (at most K),
then, up to the leaf's occurrences when it takes a weight (at most K), copy
c mod N with the c-th weights on that copy's other arguments.
Copy c < N takes fresh activations below M and copy c mod M's from there,
weight c mod the more of M and the leaf's occurrences, and
instance c mod I at placement (c div I) mod P, its metadata and module
tensors cloned past the first round: a placement is the point's batch on KV
blocks and state slots of its own, as many as the pool and a KV registry
entry of per-request state hold, past what the state slots hold on the first
placements' slots again, for which ATOM prepares the forward context
again, it and the metadata views cloned once each, and only where they differ
from the last placement's. A KV argument
places a leaf's copies as a context key does. A second graph runs
copy 0 K times, warm; with one copy, or K under `CALLS`, there is none.
Before each capture, and once the runner sized its KV pool, the memory the
bench freed goes back to the driver, and the bench waits, at most `CLEAR_S`,
until the card's VRAM in use falls by it (`_release`); a point records the
seconds waited since the last point and how many waits the cap ended.
Each graph is replayed `WARMUPS` times, then `REPEATS` times timed, the host
idle `REST` times a replay's device time after each replay;
a repeat is the replay's device time over K, and a graph that captured no
kernel times 0. The kernels one eager call of copy 0 launches, memory copies
aside, are the point's kernel identity. When `ReplayRule` refuses a batch, its
leaves refuse; an error in ATOM's preparation fails the run, naming the point.
The points run in a process of their own on each GPU, which dies when its
device faulted (`_alive`); one that dies ends every rank, keeps the points
every rank finished, the point it was measuring refuses, and new ones measure
the rest; two in a row that die having finished no point fail the run.

DIR gets run.json, the run's identity, and ops.jsonl, one line per point. S
gets each measured point as an `op_measurement`, keyed by the model, the
parallel layout, the digest of the machine spec of the device, stack and PTL
state run.json reads, the source root and the point.

`--plan` measures, in rounds, the points the steps of record directories call
for (`record`'s or `trace steps`' output), capturing each `op_graph` S lacks:
every leaf at every rung of ATOM's decode ladder at the steps' lowest and
highest row context (`_rows`); then each leaf of a kind those show depends on
context (`_dependent`) over a ladder (`_ladder`) of the steps' row contexts,
and each prefill family (`_family`) over a ladder of their token counts, chunk
remainders among them, one of a kind that depends on context at each at the
most cached tokens of steps of its rows, and over a ladder of them once those
points and the fresh ones show the kind still does; then the point halfway
between two neighbours of a family, or of a prefill leaf's token count over
cached tokens, that launched different kernels or, at a decode width, either
side of a point off the line through theirs, or whose cold times differ where
the leaf's kind steps under one kernel between two adjacent contexts at some
width (`_densify`), until they are adjacent; a collective's message bytes bend
as a decode width does. DIR gets each round's ops.jsonl in `round<n>`, and
every point held in plan.jsonl.

`--coverage` counts the steps of record directories that DIR's plan.jsonl
covers (`coverage`), with the `op_graph`s of the arch DIR's run.json records,
and prints the count by phase and why each leaf kind left the rest out.
"""

import argparse
import bisect
import contextlib
import copy
import dataclasses
import datetime
import functools
import gc
import hashlib
import importlib
import inspect
import itertools
import json
import multiprocessing
import os
import re
import statistics
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

import torch

from atom.compass import ir
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    roots_for,
)
from atom.compass.drive.gpu import add_gpu_flag
from atom.utils.arg_parser import FlexibleArgumentParser

PROG = "compass measure ops"
#: The timing protocol's version; it changes whenever a count below, the
#: copies a graph reads or what a capture waits for change.
PROTOCOL = 13
WARMUPS = 3
REPEATS = 20
#: Calls a graph holds, and so the copies it cycles over at most. One count for
#: every leaf a graph of which fits `GRAPH_NS`: a replay's fixed cost adds the
#: same share to every such price.
CALLS = 256
#: Device time a graph's calls fill at most: a longer call takes fewer calls,
#: at least one, so a capped graph holds at least 5 ms: a replay's fixed cost,
#: about 19 us on MI308X, is under 0.4% of it.
GRAPH_NS = 10_000_000
#: Times over the last-level cache the copies' bytes pass: just past it, part of
#: them stays there, and an MI308X GEMM read up to 7% fast.
COLD = 2
#: Host idle after each replay, as a share of its device time. Replayed back to
#: back on an MI308X, the lm_head GEMM ran at a 1100 MHz memory and 1550 MHz
#: fabric clock, against 1300 and 1800 MHz with an idle after each call, at the
#: same socket power, and read up to 2.55% over ATOM's decode step by block
#: median. A power limit is the suspected cause. With this idle after a replay
#: `GRAPH_NS` bounds, it read 0.22% to 0.58% under the step.
REST = 0.1
#: Seconds a release waits at most for the driver to clear what it returned
#: (`_release`), and the VRAM in use over the release's target that ends it:
#: on MI308X a release of 78 to 208 MiB left VRAM in use where it was for 3 s,
#: while 2.4 to 51 GiB fell by the release's bytes.
CLEAR_S, CLEAR_SLACK = 3.0, 256 << 20
#: The release waits since a point last recorded them: seconds, and how many
#: ended at `CLEAR_S`.
_waited: Counter = Counter()
#: Metadata arguments, by operator and slot, that are the token ids ATOM passes
#: the model's forward; `_meta`'s name rule binds an embedding's `indices` to
#: another index buffer.
TOKEN_IDS = {("aten.embedding.default", "indices")}
#: Weight slots a leaf reads a row a token of: RoPE's cos and sin tables,
#: indexed by each token's position (or by each compressed position, at most
#: one a token), never read whole.
TABLES = {"cos_cache", "sin_cache", "cos", "sin"}
#: The operator whose metadata slots are named as the fields of DeepSeek-V4's
#: HCA decode workspace: aiter defines its schema and the workspace together.
HCA_DECODE = "aiter.mla_decode_v4_ps_asm.default"
#: DeepSeek-V4's metadata arguments, by operator and slot, a dict argument's
#: item by its key (`slot.<key>`), and the live buffer ATOM passes each, at the
#: compress ratio the leaf takes or states (`{ratio}`), that ratio's
#: attention kind (`{kind}`) and its destination rows' class (`{dest}`):
#: `_meta`'s rules bind them to a tighter buffer of their dtype. A leaf whose
#: buffer is not live refuses.
#: A decode's `split_indptr` is its split plan's, `split_plan_<kind>.1`. The
#: paged-decode index leaves write their indptrs into the clone that the
#: leaves timed after them at a placement read.
_NM = "aiter.mla:mla_decode_fwd_v4_nm"
_INDICES = "atom.model_ops.v4_kernels.paged_decode_indices"
_WRITE = f"{_INDICES}:write_v4_paged_decode_indices"
_CSA = "aiter.ops.flydsl.kernels.fused_compress_attn:flydsl_fused_compress_attn"
_HCA = "aiter.ops.flydsl.kernels.fused_compress_attn_hca:flydsl_hca_compress_attn"
_STATE = "atom.model_ops.v4_kernels.state_writes:update_compressor_states"
_QKN = "aiter._fused_qk_norm_rope_group_quant_kernel.default"
_QKN_BF16 = "aiter.ops.flydsl.kernels.qk_norm_rope_quant:flydsl_qk_norm_rope_quant"
_BF16 = "atom.model_ops.v4_kernels.paged_decode:_sparse_attn_v4_paged_decode_triton"
_DESTS = {0: "dense", 4: "csa", 128: "hca"}
V4_METADATA = {
    (_NM, "kv_indptr"): "kv_indptr_{kind}",
    (_NM, "split_indptr"): "split_plan_{kind}.1",
    (_WRITE, "state_slot_per_seq"): "v4_meta_state_slot_out",
    **{
        (_WRITE, f"dest_rows.{ratio}"): f"v4_swa_dest_{dest}"
        for ratio, dest in _DESTS.items()
    },
    **{
        (f"{_INDICES}:{fn}", f"{kind}_indptr"): f"kv_indptr_{kind}"
        for fn in ("build_v4_paged_decode_indptr", "write_v4_paged_decode_indices")
        for kind in ("swa", "csa", "hca")
    },
    (HCA_DECODE, "kv_indptr"): "kv_indptr_{kind}",
    (_BF16, "kv_indptr"): "kv_indptr_{kind}",
    (_CSA, "state_slot_mapping"): "state_slot_in",
    (_CSA, "plan_gpu"): "v4_compress_plan_{ratio}",
    (_HCA, "state_slot_mapping"): "state_slot_in",
    (_HCA, "plan_gpu"): "v4_compress_plan_{ratio}",
    (_STATE, "state_slot_mapping"): "state_slot_out",
    (_STATE, "write_plan"): "v4_write_plan_{ratio}",
    (_QKN, "batch_id_per_token"): "batch_id_per_q_token",
    (_QKN, "swa_dest_row"): "v4_swa_dest_{dest}",
    (_QKN_BF16, "batch_id_per_token"): "batch_id_per_q_token",
    (_QKN_BF16, "swa_dest_rows"): "v4_swa_dest_{dest}",
}
V4_KINDS = {0: "swa", 4: "csa", 128: "hca"}
#: The one leaf and slot built past its span: aiter's `mla_decode_fwd_v4_nm`
#: at one KV split writes its BF16 `output` at the FP32 split stride, twice
#: the span, and past an exact buffer kills the process (aiter defect, #1977).
NM_OUTPUT = (_NM, "output")
#: The seed of a copy's drawn expert ids, with the copy's index.
SEED = 0
#: The rank `trace ops` writes an `op_graph`'s members at, which `graph` reads.
RANK = Topology().ranks()[0]
#: The variable naming the file a measuring process appends each point's
#: record to as it finishes it.
JOURNAL = "COMPASS_OPS_JOURNAL"


def main(argv: list[str]) -> None:
    from atom.compass.bench import ranks
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture.forward import decode, prefill
    from atom.compass.drive.gpu import hip_visible, kfd_pid, quiet
    from atom.compass.drive.identity import identity
    from atom.compass.run.step_grid import keyed
    from atom.model_engine.arg_utils import EngineArgs

    cut = argv.index("--") if "--" in argv else len(argv)
    if "--coverage" in argv[:cut]:
        return print(json.dumps(_coverage_main(argv[:cut], argv[cut + 1 :]), indent=1))
    args = _parser().parse_args(argv[:cut])
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    if not shapes and not args.plan:
        raise ValueError("name a shape: --width or --prefill, or --plan")
    if shapes and args.plan:
        raise ValueError("--plan places every point; name no --width or --prefill")
    if args.context is None and any(s.phase == "decode" for s in shapes):
        raise ValueError("a decode shape needs --context, each row's tokens")
    engine = EngineArgs.add_cli_args(FlexibleArgumentParser(prog=f"{PROG} --"))
    engine = engine.parse_args(argv[cut + 1 :])
    world = engine.tensor_parallel_size * engine.data_parallel_size
    # DP attention makes each TP rank a DP rank, and one GPU can run one alone
    alone = engine.enable_dp_attention and len(args.gpu) == 1
    if world != len(args.gpu) and not alone:
        raise ValueError(
            f"--gpu names {len(args.gpu)} GPUs and ATOM's arguments resolve "
            f"{world} ranks; the op bench runs one on each, or one DP-attention "
            "rank on one GPU"
        )
    os.environ["HIP_VISIBLE_DEVICES"] = hip_visible(args.gpu)
    kfd_pid()  # before atom_config opens a GPU, so quiet leaves this process out
    config = atom_config(argv[cut + 1 :])
    ranks.layout(config)
    if config.compilation_config.level:
        raise ValueError(
            f"--level {config.compilation_config.level} compiles the model, whose "
            "forward the bench never runs; measure at --level 0"
        )
    _eager(config)
    store = ArtifactStore(args.store)
    todo = [
        (s, args.context, leaves(graph(store, config, s)["region"])) for s in shapes
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine.json").write_text(json.dumps(keyed(config)))
    command = [*PROG.split(), *argv]
    with quiet(args.gpu, out):
        run = identity(out / "engine.json", config.model, command, args.gpu)
        (out / "run.json").write_text(json.dumps(run))
        key, held = keys(config, run), 0
        if args.plan:
            command, atom_args = " ".join(command), argv[cut + 1 :]
            return plan(store, key, config, run, args, command, out, atom_args)
        for shape, context, found in todo:
            for sig in list(found):
                if store.directory_for(key(shape, context, sig)).exists():
                    del found[sig]
                    held += 1
        print(f"{held} points already in the store", flush=True)
        todo = [point for point in todo if point[2]]
        results = _measured(config, todo, args.llc_mib << 20, len(args.gpu))
    publish(store, key, config, run, " ".join(command), results, out)
    refused = [
        f"{r['op']} at {r['shape']}: {r['refused']}" for r in results if "refused" in r
    ]
    print(f"{len(results) - len(refused)} of {len(results)} points measured")
    if refused:
        raise ValueError(f"{len(refused)} points refused:\n" + "\n".join(refused))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--width", type=int, action="append", default=[], help="a decode width"
    )
    p.add_argument(
        "--prefill",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("T", "B"),
        help="a prefill of T tokens over B sequences",
    )
    p.add_argument(
        "--context",
        type=int,
        help="each decode row's tokens, or the tokens cached before each prefill row",
    )
    p.add_argument(
        "--plan",
        action="append",
        default=[],
        metavar="RECORD",
        help="a record directory whose steps the planned points cover",
    )
    p.add_argument(
        "--llc-mib",
        type=int,
        required=True,
        help="the device's last-level cache in MiB: 256 on MI300X, MI308X and "
        "MI355X (AMD Infinity Cache, CDNA 3 and CDNA 4 white papers)",
    )
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument(
        "--out",
        required=True,
        help="where run.json and ops.jsonl go; with --plan, each round's "
        "ops.jsonl goes in round<n> and every point held in plan.jsonl",
    )
    return p


def graph(store, config, shape, arch=None) -> dict:
    """The `op_graph` `trace ops` published for `config` at `shape` on `arch`,
    this GPU's when it is None, of a step that produces output, as JSON: a
    prefill that produces none runs the same leaves less the LM head and the
    sampler, so this graph's points cover both. A miss where the store holds
    the shape under another engine config refuses naming what differs."""
    from atom.compass.artifacts import ArtifactRefusal, Rule
    from atom.compass.capture.forward import missed, structure

    key = Key.of(
        Kind.OP_GRAPH,
        structure=structure(config, shape.phase, device=arch),
        shape=str(shape),
    )
    if not store.directory_for(key).exists() and (why := missed(store, key)):
        raise ArtifactRefusal(
            Rule.RESOLUTION,
            f"no op_graph holds {shape} under this engine config{why}",
            "compass trace ops captures one under this process's environment",
        )
    return json.loads(store.read(key).read_member("graph", RANK, "json"))


def _ref(value) -> bool:
    return isinstance(value, dict) and value.keys() in ({"context"}, {"module"})


def signature(op: dict) -> str:
    """A leaf's identity across instances: its operator, arguments and outputs,
    a context key or module path by its kind alone, and what the capture
    states beside its arguments when it states anything."""
    args = {s: {k: None for k in v} if _ref(v) else v for s, v in op["args"].items()}
    leaf = {"op": op["op"], "args": args, "out": op["out"]}
    leaf |= {"stated": op["stated"]} if "stated" in op else {}
    return json.dumps(leaf, sort_keys=True, separators=(",", ":"))


def leaves(region: dict, env=None, found=None) -> dict[str, tuple[dict, list]]:
    """Each leaf of a graph's JSON region by `signature`: one of its ops, its
    times in the step under ``occurrences``, and its instances in run order,
    each the key or path every reference slot binds."""
    env, found = env or {}, {} if found is None else found
    if "op" in region:
        bound = {}
        for slot, value in _values(region["args"]):
            if _ref(value):
                ref = ir.ContextRef(next(iter(value.values())))
                bound[slot] = ref.bind(**{n: env[n] for n in ref.index_names}).key
        leaf = region | {"occurrences": 0}
        op, instances = found.setdefault(signature(region), (leaf, []))
        op["occurrences"] += 1
        instances += [bound] if bound not in instances else []
    elif "repeat" in region:
        for i in range(region["count"]):
            index = {region["index"]: region["start"] + region["step"] * i}
            leaves(region["repeat"], {**env, **index}, found)
    else:
        for item in region.get("seq", region.get("par", ())):
            leaves(item, env, found)
    return found


def measure(config, todo, llc: int, rank: int = 0) -> list[dict]:
    """Each point's record, shape by shape: its timing, or why it refused; at
    `rank` of `config`'s layout, over every rank's points (`ranks.union`);
    each also appended to the `JOURNAL` file as it is finished, when one is
    named, suffixed `.<rank>` over several ranks."""
    from atom.compass.bench import ranks
    from atom.compass.bench.replay import ReplayRule

    runner, journal = _runner(config, rank), os.environ.get(JOURNAL)
    try:
        group = ranks.cpu_group()
        if journal and group is not None:
            journal += f".{torch.distributed.get_rank()}"
        stub, rule, got = Stub(runner, llc, journal), ReplayRule(runner.config), []
        todo = ranks.union(todo, group)
        with torch.inference_mode():
            for shape, context, found in todo:
                got += stub.point(rule, shape, context, found)
        return got
    finally:
        runner.exit()


def _runner(config, rank: int):
    """ATOM's runner at TP `rank` of `config`'s DP rank, its parameters built on
    the meta device and no warmup run, its KV pool allocated at the fewest
    blocks a rank has room for, sized by ATOM with the parameters' bytes held
    on the device and then freed for the weights each leaf builds; the runner
    attributes the allocation set are in its `kv_held`; and, when the runner
    defers its output, one sampled-id handoff to the host on ATOM's copy
    stream, as ATOM's postprocess makes one after every forward. One DP rank
    of several run alone is built as ATOM's simulated DP attention builds it:
    DP 1, its experts cut as across the ranks."""
    from torch.distributed import TCPStore

    from atom.compass.bench.ranks import agreed, cpu_group
    from atom.model_engine.model_runner import ModelRunner, RapidServeModelRunner

    class MetaRunner(ModelRunner):
        def _build_and_load_model(self, model_class):
            with RapidServeModelRunner._init_weight_params_on_meta():
                self.model = model_class(self.config)
            torch.set_default_device(None)

        def _maybe_warmup(self):
            pass

    pc, store = config.parallel_config, None
    if not pc._managed_distributed_store:  # one rank: the store is its own
        if pc.data_parallel_size > 1:
            # ponytail: ATOM's top-k buffers then hold one rank's batched
            # tokens, not the rows the layout gathers: a leaf past them refuses
            config = copy.deepcopy(config)
            config.dp_logical_size = pc.data_parallel_size
            pc = config.parallel_config
            pc.data_parallel_size = pc.data_parallel_size_local = 1
        store = TCPStore(
            pc.data_parallel_master_ip,
            pc.data_parallel_base_port,
            is_master=True,
            wait_for_workers=False,
        )
        pc.data_parallel_base_port, pc._managed_distributed_store = store.port, True
    runner = MetaRunner(rank, config)
    runner.store = store
    nbytes = sum(p.nbytes for p in runner.model.parameters())
    held = torch.empty(nbytes, dtype=torch.uint8, device=runner.device)
    blocks = runner.get_num_blocks()["num_kvcache_blocks"]
    del held
    _release()  # the pool takes the room the hold had
    config.num_kvcache_blocks = agreed(blocks, cpu_group())
    before = dict(vars(runner))
    runner.allocate_kv_cache(config.num_kvcache_blocks)
    # what the allocation bound on the runner: `trace ops` reads a tensor
    # made there as KV
    runner.kv_held = {k: v for k, v in vars(runner).items() if before.get(k) is not v}
    # with no handoff made, a Qwen3.8-27B GDN decode leaf read 5% under a
    # runner that had made one
    ids, done = runner.tokenID_processor, torch.cuda.Event()
    if ids.is_deferred_out:
        done.record()
        zero = torch.zeros(1, dtype=torch.int32, device=runner.device)
        ids.send_to_cpu_async(zero, ids.token_ids_cpu, done)
        ids.recv_async_output(ids.token_ids_cpu)
    return runner


class Stub:
    """The leaves of a point timed beside ATOM's runner, on the forward context
    and live metadata ATOM prepared for each placement of its batch. `alone`
    is the layout's ranks when the runner runs one of them by itself, and then
    a collective leaf refuses."""

    def __init__(self, runner, llc: int, journal=None) -> None:
        from atom.model_engine.kv_block import STATE_SLOT_CLASS

        self.runner, self.llc, self.journal = runner, llc, journal
        self.alone = getattr(runner.config, "dp_logical_size", 0)
        self.slots = runner.pool_plan.entries.get(STATE_SLOT_CLASS, 0)
        self.built, self.leaves, self.placed = {}, {}, []
        self.cold_copies, self.clones, self.kv, self.state = {}, {}, {}, set()
        self.modules = dict(runner.model.named_modules())
        self.contexts = runner.config.compilation_config.static_forward_context
        self.moe = sorted(
            {
                (m.global_num_experts, m.top_k)
                for m in self.modules.values()
                if hasattr(m, "global_num_experts")
            }
        )
        # ATOM's top-k weights and ids buffers when its MoE layers fuse shared
        # experts into top-k (`init_aiter_topK_meta_data`), live metadata a
        # leaf may read: past top-k, the shared experts' ids, under EP one
        # more, and the fake expert's on rows another rank owns
        topk = sys.modules.get("atom.model_ops.topK")
        meta = getattr(topk, "aiter_topK_meta_data", None)
        self.topk = dict(
            zip(("topk_weights", "topk_ids"), meta if meta and self.moe else ())
        )
        self.shared = self.topk.get("topk_ids")
        self.loads = []
        # a DeepSeek-V4 CSA layer: its forward writes the top-k section of the
        # CSA index buffer, and `_prepare` runs none
        self.csa = next(
            (
                m
                for m in self.modules.values()
                if getattr(m, "indexer", None) is not None
                and hasattr(m, "_fill_csa_paged_compress")
            ),
            None,
        )

    def _prepare(self, batch) -> None:
        """`batch` prepared by ATOM's runner as its forward prepares one, no
        forward run, and a snapshot of the forward context it built and the
        live metadata each leaf names, beside the KV `point` found for it."""
        from atom.utils.forward_context import get_forward_context

        self.runner._gate_staging_reuse()
        input_ids = self.runner.prepare_model(batch)[0]
        self.runner._mark_staging_h2d_enqueued()
        fc, live = get_forward_context(), {"input_ids": input_ids, **self.topk}
        if self.csa is not None:
            self._top_k(fc, len(input_ids))
        for name, buffer in self.runner.forward_vars.items():
            live[name] = getattr(buffer, "gpu", buffer)
        for owner in (fc.attn_metadata, fc.context):
            live |= {k: v for k, v in vars(owner).items() if torch.is_tensor(v)}
        device = self.runner.device.type
        live = {
            k: v
            for k, v in live.items()
            if torch.is_tensor(v) and v.device.type == device
        }
        # DeepSeek-V4's persistent HCA decode workspace, one per device: a slot
        # of `HCA_DECODE` named as one of its fields is a view of that field
        hca = sys.modules.get("atom.model_ops.v4_kernels.hca_persist")
        work = getattr(hca, "_workspaces", {}).get(self.runner.device.index)
        work = work._asdict() if work is not None else {}
        # a decode split plan, `(splits, split_indptr)`: only a slot
        # `V4_METADATA` names takes it, so `_meta`'s rules never offer it
        named = live | {
            f"{k}.1": v[1]
            for k, v in vars(fc.attn_metadata).items()
            if k.startswith("split_plan_") and v
        }
        meta = dict(self.kv)
        for sig, (op, _) in self.leaves.items():
            ratio = op["args"].get("ratio", op.get("stated", {}).get("compress_ratio"))
            for path, spec in _specs(op["args"]):
                if spec["role"] == "metadata":
                    token = (op["op"], path) in TOKEN_IDS
                    pool = {"input_ids": input_ids} if token else live
                    what, field = "metadata", path.split(".")[0]
                    if op["op"] == HCA_DECODE and field in work:
                        pool, what = {field: work[field]}, "HCA workspace"
                    if wanted := V4_METADATA.get((op["op"], _keyed(op["args"], path))):
                        what = wanted.format(
                            ratio=ratio,
                            kind=V4_KINDS.get(ratio),
                            dest=_DESTS.get(ratio),
                        )
                        pool = {what: named[what]} if what in named else {}
                    try:
                        if not pool:
                            raise ValueError(f"ATOM passes {path} {what}: not live")
                        name, view = _meta(spec, path, pool, what)
                        # leaves share a view's clone: DeepSeek-V4's decode
                        # leaves view 2.7 GB of it at a placement
                        meta[sig, path] = name, _deep(view, {}, self.clones)
                    except ValueError as error:  # the leaf refuses, not the step
                        meta[sig, path] = str(error), None
        # one clone across placements of what none changed: a leaf may cache
        # what it prepares on the host by its identity (linear attention's
        # chunk indices), and a graph over placements replays all of it
        memo: dict = {}
        snap = copy.copy(fc)
        snap.attn_metadata = _deep(fc.attn_metadata, memo, self.clones)
        snap.context = _deep(fc.context, memo, self.clones)
        self.placed.append((snap, meta))

    def _top_k(self, fc, tokens) -> None:
        """The CSA top-k section of the placement's index buffer written as
        ATOM's forward writes it, from top-k ids over the compressed rows each
        of its `tokens` sees: every row when it sees no more than top-k, else
        one drawn from each of top-k equal runs of them, seeded by `SEED` and
        the placement."""
        # ponytail: spread over the rows, ascending; the indexer picks where
        # its scores fall, which no placement has
        positions, k = fc.context.positions[:tokens], self.csa.indexer.index_topk
        seen = (positions.long() + 1) // self.csa.compress_ratio
        run = torch.arange(k, device=positions.device)
        lo, hi = run * seen[:, None] // k, (run + 1) * seen[:, None] // k
        seed = torch.Generator(positions.device)
        seed.manual_seed(SEED * CALLS + len(self.placed))
        draw = torch.rand(lo.shape, generator=seed, device=positions.device)
        ids = torch.where(seen[:, None] > k, lo + (draw * (hi - lo)).long(), run)
        self.csa._fill_csa_paged_compress(
            fc.attn_metadata, ids.int(), fc.context.positions, tokens
        )

    def point(self, rule, shape, context, found) -> list[dict]:
        """Each leaf of `found` timed at `shape` on `point_rows`, or every leaf
        refused when `rule` refuses the batch."""
        from atom.utils.forward_context import _forward_kv_cache_context

        pool = self.runner.config.num_kvcache_blocks
        rows = point_rows(shape, context, pool, rule.block_size)
        view = {"rows": rows, "produces_output": True}
        blocks = sum(-(-row[1] // rule.block_size) for row in rows)
        kv = _forward_kv_cache_context.kv_cache_data
        held = _held(self.runner, kv)
        self.state = _state(self.runner, held)
        self.kv = {
            (s, path): _kv(spec, path, held)
            for s, (op, _) in found.items()
            for path, spec in _specs(op["args"])
            if spec["role"] == "kv"
        }
        room = [pool // blocks]
        room += [t.k_cache.shape[0] // len(rows) for t in kv.values() if _stateful(t)]
        plans = {
            s: self._plan(op, i, kv, blocks, len(rows), shape.tokens, s)
            for s, (op, i) in found.items()
        }
        need = max(p for _, _, p, _ in plans.values())
        self.leaves, self.placed, self.context, self.clones = found, [], context, {}
        self.cold_copies = {s: self._cold(size) for s, (_, size, *_) in plans.items()}
        for p in range(max(min(need, *room), 1)):
            slot = p * len(rows)
            if self.slots >= len(rows):  # past the state slots, take them again
                slot %= self.slots - self.slots % len(rows)
            try:
                batch = rule.batch(view, p * blocks, slot)
            except ValueError as error:  # the rule refuses, before any preparation
                refused = [
                    {"shape": str(shape), "context": context, "op": op["op"]}
                    | {"leaf": sig, "refused": f"ValueError: {error}"}
                    for sig, (op, _) in found.items()
                ]
                return _journal(self.journal, refused)
            try:
                self._prepare(batch)
            except Exception as error:
                error.add_note(f"measuring {shape} at context {context}")
                raise
        got = []
        for sig, (op, inst) in found.items():
            n, size, _, (by_block, by_slot) = plans[sig]
            names = {p: m[0] for (s, p), m in self.placed[0][1].items() if s == sig}
            record = {"shape": str(shape), "context": context, "op": op["op"]}
            record |= {"leaf": sig, "metadata": names, "rows": rows}
            record |= {"exchange": ex} if (ex := exchange(op)) else {}
            try:
                record |= self._time(op, sig, inst, n)
                # a KV argument is one view every copy reads at its placement:
                # its bytes count once a placement, a slot's once however many
                # placements wrap onto it
                built = min(n, record.get("copies", n))
                at = min(len(self.placed), -(-built // len(inst)))
                wrap = self.slots // len(rows) if self.slots >= len(rows) else at
                record["footprint_bytes"] = (
                    built * (size - by_block - by_slot)
                    + at * by_block
                    + min(at, wrap) * by_slot
                )
            except Exception as error:  # noqa: BLE001 - a refusal names what stopped it
                record["refused"] = f"{type(error).__name__}: {error}"
                _alive(self.runner.device)
            copies = record.get("copies", n)
            print(f"{shape}: {op['op']} over {copies} copies {names or ''}", flush=True)
            record |= {"instances": len(inst), "placements": len(self.placed)}
            got += _journal(self.journal, [record])
        self.placed = []
        return got

    def _plan(self, op, instances, kv, blocks, rows, tokens=None, sig=None):
        """A leaf's copies, its bytes per call, the placements its context
        keys and KV arguments take, and its KV arguments' bytes per call read
        by blocks and by state slots; `tokens` defaults to `rows`, a
        decode's."""
        need, size, shared = 1, 0, [0, 0]
        for slot, s in _specs(op["args"]):
            itemsize = _dtype(s["dtype"]).itemsize
            nbytes = _span(s["shape"], s["stride"]) * itemsize
            if s["role"] == "weight" and slot in TABLES:
                entry = _span(s["shape"][1:], s["stride"][1:]) * itemsize
                nbytes = entry * min(tokens or rows, s["shape"][0])
            if s["role"] == "kv":
                # ponytail: its first dim taken as blocks or state slots, or as
                # plane rows when the leaf states its layer's attention geometry;
                # beside state slots a block view is one block a request, the
                # one a decode compressor writes its row to
                unit = rows if self.slots else blocks
                slotted = int(self.kv.get((sig, slot), ("",))[0] in self.state)
                stated = op.get("stated", {})
                if "window_size" in stated and (ratio := stated["compress_ratio"]):
                    # the rows the batch's blocks hold at its layer: a row
                    # per `ratio` tokens; its window rows are left out
                    unit = blocks * self.runner.config.kv_cache_block_size // ratio
                    slotted = 0
                    # of which a request reads at most its indexer's top-k
                    if top := stated.get("index_topk"):
                        unit = min(unit, rows * top)
                # an entry is the span of its trailing dims: a view may be
                # strided by a whole state slot or by every layer's block
                entry = _span(s["shape"][1:], s["stride"][1:]) * itemsize
                nbytes, need = entry * unit, None
                shared[slotted] += nbytes
            size += nbytes
        for key in instances[0].values():
            module = self.contexts.get(key)
            if hasattr(module, "layer_num"):  # a layer of ATOM's KV registry
                held = kv.get(f"layer_{module.layer_num}")
                unit = rows if _stateful(held) else blocks
                # ponytail: a tensor's first dim is its blocks or its state slots
                size += sum(t.nbytes // t.shape[0] * unit for t in _fields(held))
                need = None
            elif (module := self.modules.get(key, module)) is not None:
                pool = self.runner.config.num_kvcache_blocks
                for t in _own(module):
                    # ponytail: a first dim of the pool's blocks is a view of
                    # the KV pool (DeepSeek-V4's Indexer), read at the batch's
                    if t.dim() and t.shape[0] == pool:
                        size += t.nbytes // pool * blocks
                        need = None
                    else:
                        size += t.nbytes
        n = self._cold(size)
        if need is None:  # cycle over every instance's state, which is held anyway
            n = min(CALLS, max(n, len(instances)))
        return n, size, need or -(-n // len(instances)), shared

    def _cold(self, size) -> int:
        """Copies of `size` bytes a call whose bytes pass `COLD` times the cache."""
        return min(CALLS, COLD * self.llc // size + 1) if size else 1

    def _time(self, op, sig, instances, n) -> dict:
        """`op` timed in a graph of K calls, `CALLS` or as many as fill
        `GRAPH_NS`, cold over at most `n` copies, past the first `m` its point
        sized to the cache (`cold_copies`) each on copy `c mod m`'s
        activations, then copy `c mod n` with the c-th weight up to the
        leaf's occurrences when it takes one, at most K; and, when K is `CALLS`,
        warm over one; with its kernels. Every rank's graph holds the most K
        any rank's call gives, and a leaf any rank refuses, every rank does."""
        from atom.compass.bench.ranks import agreed, cpu_group, refused_together
        from atom.utils import forward_context

        if self.alone and exchange(op):
            raise ValueError(
                f"a collective over the layout's {self.alone} ranks, measured "
                "across them, not on a runner that runs one alone"
            )
        fn, m = _function(op["op"]), self.cold_copies.get(sig, n)
        self.loads, self.built = [], {}
        with refused_together():
            calls = [self._copy(op, sig, instances, 0)]
        # one weight can read a few percent off another of its spec: cycle one
        # per occurrence of the leaf, on the n copies' other arguments
        weighs = any(s["role"] == "weight" for _, s in _specs(op["args"]))
        held = [op["occurrences"]] if weighs else []
        try:
            forward_context._forward_context = calls[0][1]
            with refused_together():
                fn(**calls[0][0])  # compiles and tunes outside the graphs
            kernels, ns = _kernels(fn, calls[0][0])
            k = min(CALLS, max(int(GRAPH_NS // ns), 1)) if ns else CALLS
            # a collective's call holds its wait on the last rank to arrive,
            # whose own call is the shortest: every graph takes its calls
            k = agreed(k, cpu_group(), most=True)
            n = min(n, k)
            with refused_together():
                calls += [
                    self._copy(op, sig, instances, c) for c in range(1, min(n, m))
                ]
                calls += [
                    self._copy(op, sig, instances, c, calls[c % m]) for c in range(m, n)
                ]
                for c in range(n, min(k, max([n, *held]))):
                    calls.append(self._reweigh(op, sig, calls[c % n], c))
            n = len(calls)
            cycle = [calls[i % n] for i in range(k)]
            cold = _replay(fn, cycle)
            warm = _replay(fn, calls[:1] * k) if n > 1 and k == CALLS else None
        finally:
            forward_context._forward_context = forward_context.ForwardContext()
        medians = statistics.median(cold), warm and statistics.median(warm)
        return {
            "kernels": kernels,
            "cold_ns": medians[0],
            "warm_ns": medians[1],
            "warm_over_cold": medians[1] / medians[0] if warm and medians[0] else None,
            "cold_repeats": cold,
            "warm_repeats": warm,
            "copies": n,
            "calls": len(cycle),
            "llc_bytes": self.llc,
            "protocol": PROTOCOL,
            "release_wait_s": _waited.pop("s", 0.0),
            "release_capped": _waited.pop("capped", 0),
            "routing": _routing(self.loads) if self.loads else None,
        }

    def _copy(self, op, sig, instances, c, like=None):
        """Copy `c` of `op`'s call: its keyword arguments and forward context,
        on the activations of copy `like` when given."""
        from atom.compass.capture.forward import rebuild_object, referenced

        instance = instances[c % len(instances)]
        p = c // len(instances) % len(self.placed)
        snap, meta = self.placed[p]
        fresh = c >= len(self.placed) * len(instances)

        def build(path, value, old):
            if isinstance(value, list):
                return [
                    build(f"{path}.{i}", v, old and old[i]) for i, v in enumerate(value)
                ]
            if isinstance(value, dict) and self._routed(value, sig, path):
                return self._route(value, c)
            if isinstance(value, dict) and value.get("role") == "weight":
                # each cache-sized copy takes a weight of its own, as it does activations
                count = max(op["occurrences"], self.cold_copies.get(sig, 1))
                return self._weight(value, c, count)
            if isinstance(value, dict) and "object" in value:
                return rebuild_object(
                    value["object"], build(path, value["state"], None)
                )
            if (
                isinstance(value, dict)
                and value.get("role") == "activation"
                and old is not None
            ):
                return old
            if _ref(value):
                key, module = instance[path], "module" in value
                held = (
                    referenced(self.runner, key) if module else self.contexts.get(key)
                )
                if isinstance(held, torch.nn.Module):
                    _fill(held, self.runner.device)
                return (_module_copy(held) if fresh else held) if module else key
            if isinstance(value, dict) and "affine" in value:
                return ir.Affine(*value["affine"]).at(self.context)
            if (op["op"], path) == NM_OUTPUT and op["args"].get("num_kv_splits") == 1:
                # the captured view at the start of the kernel's write extent
                n = _span(value["shape"], value["stride"])
                device = self.runner.device
                whole = _random({**value, "shape": [2 * n], "stride": [1]}, device)
                return whole.as_strided(value["shape"], value["stride"])
            if isinstance(value, dict):
                return self._tensor(value, meta.get((sig, path)), fresh)
            if isinstance(value, str) and value.startswith("torch."):
                return getattr(torch, value[6:], value)
            if isinstance(value, str) and value.split(":")[0] == "cuda":
                return self.runner.device  # the capture ran on device 0
            return value

        args = op["args"].items()
        return {s: build(s, v, like[0][s] if like else None) for s, v in args}, snap

    def _routed(self, spec, sig, path) -> bool:
        """Whether `spec` at `sig`'s `path` is a router's top-k expert ids: an
        integer activation, or a metadata view `_meta` bound to ATOM's ids
        buffer, a row per token, a column per expert an MoE layer of the model
        takes, or per column of ATOM's ids buffer with its fused shared
        experts."""
        widths = {k for _, k in self.moe}
        if self.shared is not None:
            widths.add(self.shared.shape[1])
        bound = self.placed[0][1].get((sig, path), ("",))[0]
        return (
            (spec.get("role") == "activation" or bound == "topk_ids")
            and len(spec["shape"]) == 2
            and not _dtype(spec["dtype"]).is_floating_point
            and spec["shape"][1] in widths
        )

    def _route(self, spec, c):
        """Copy `c`'s expert ids for `spec`: each row's experts drawn
        independently and uniformly, distinct, from the model's routed experts,
        seeded by `SEED` and `c`, and past top-k the row's ids in ATOM's buffer;
        the copy's expert loads go to `loads`."""
        if len(self.moe) > 1:
            raise ValueError(
                f"the model's MoE layers route (experts, top-k) {self.moe}: "
                f"which one draws {_text(spec)} is not known"
            )
        [(experts, k)] = self.moe
        device = self.runner.device
        seed = torch.Generator(device).manual_seed(SEED * CALLS + c)
        # double, not the default dtype ATOM's runner sets to the model's: a
        # bfloat16 or float32 draw ties, and the device's top-k breaks a tie
        # differently from call to call
        draw = torch.rand(
            spec["shape"][0], experts, generator=seed, device=device, dtype=torch.double
        )
        ids = draw.topk(k).indices
        self.loads.append(torch.bincount(ids.flatten(), minlength=experts).tolist())
        out = _random(spec, device)
        out[:, :k] = ids
        if out.shape[1] > k:
            out[:, k:] = self.shared[: len(out), k:]
        return out

    def _reweigh(self, op, sig, like, c):
        """Copy `like` of `op`'s call with the c-th weight of each weight
        argument's spec and expert ids of its own, its other arguments and
        forward context shared."""

        def build(path, value, old):
            if isinstance(value, list):
                return [
                    build(f"{path}.{i}", v, o)
                    for i, (v, o) in enumerate(zip(value, old))
                ]
            if isinstance(value, dict) and value.get("role") == "weight":
                return self._weight(value, c, op["occurrences"])
            if isinstance(value, dict) and self._routed(value, sig, path):
                return self._route(value, c)
            return old

        kwargs, snap = like
        return {s: build(s, v, kwargs[s]) for s, v in op["args"].items()}, snap

    def _tensor(self, spec, meta, fresh):
        role = spec["role"]
        if role == "activation":
            return _random(spec, self.runner.device)
        if role in ("metadata", "kv"):
            if meta[1] is None:
                raise ValueError(meta[0])
            return _clone(meta[1]) if fresh and role == "metadata" else meta[1]
        raise ValueError(f"a {role} tensor argument {_text(spec)} is not built")

    def _weight(self, spec, c, count):
        """Weight `c mod count` of `spec`, random: the `count` of a leaf are
        built once, as slices of one tensor, each marked preshuffled when ATOM
        marked the captured one, so the operator picks the kernel it picked."""
        if (_spec_of(spec), 0) not in self.built:
            n = _span(spec["shape"], spec["stride"])
            dtype, device = _dtype(spec["dtype"]), self.runner.device
            whole = torch.empty(count * n, dtype=dtype, device=device)
            for i in range(count):
                w = whole[i * n : (i + 1) * n]
                w.copy_(_random({**spec, "shape": [n], "stride": [1]}, device))
                view = w.as_strided(spec["shape"], spec["stride"])
                if spec.get("is_shuffled"):
                    view.is_shuffled = True
                self.built[_spec_of(spec), i] = view
        return self.built[_spec_of(spec), c % count]


def _alive(device) -> None:
    """Ends this process when `device` faulted: the fault is sticky, so every
    later call there fails, and a peer rank would wait on this one's next
    collective until it times out. `_measured` refuses the leaf it died on."""
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001 - any error here is the device's
        os._exit(1)


def _routing(loads) -> dict:
    """The drawn copies' imbalance, recorded and not priced: each copy's
    experts that took a token, and the most tokens one expert took."""
    return {
        "seed": SEED,
        "experts": len(loads[0]),
        "activated": [sum(n > 0 for n in load) for load in loads],
        "max_load": [max(load) for load in loads],
    }


def point_rows(shape, context, pool: int, block_size: int) -> list[list]:
    """A point's rows, `[query, context, decodes]` each, longest first: a decode
    row holds `context` tokens; a prefill row its query tokens, the capture's
    split, past `context` cached ones. When they need more than the `pool`'s
    KV blocks, the first keeps its context and each other shares what it
    leaves, as the rows of a ragged step do."""
    decode = shape.phase == "decode"
    per, extra = divmod(shape.tokens, shape.seqs)
    past = context - 1 if decode else context or 0
    rows = [
        [q, q + past, decode] for q in (per + (i < extra) for i in range(shape.seqs))
    ]
    blocks = [-(-row[1] // block_size) for row in rows]
    if sum(blocks) > pool and len(rows) > 1:
        share = (pool - blocks[0]) // (len(rows) - 1) * block_size
        for row in rows[1:]:
            row[1] = min(row[1], max(share, row[0]))
    return rows


def _specs(args):
    """Each tensor argument in `args`, by its slot path."""
    return [(p, v) for p, v in _values(args) if isinstance(v, dict) and "role" in v]


def _keyed(args, path) -> str:
    """A slot path's argument, a dict argument's item `slot.i.1` named by its
    key, `slot.<key>`."""
    slot, *rest = path.split(".")
    value = args.get(slot)
    if rest and isinstance(value, dict) and value.get("object") == "builtins:dict":
        return f"{slot}.{value['state'][int(rest[0])][0]}"
    return slot


def _values(args, path=""):
    """Each argument value in `args` by its slot path, `slot`, `slot.i` in a
    list, `slot.i.1` for an object's i-th attribute or item."""
    items = args.items() if isinstance(args, dict) else enumerate(args)
    for slot, value in items:
        where = f"{path}.{slot}" if path else str(slot)
        if isinstance(value, dict) and "object" in value:
            value = value["state"]
        if isinstance(value, list):
            yield from _values(value, where)
        else:
            yield where, value


def _meta(spec, slot, live, what="metadata") -> tuple[str, torch.Tensor]:
    """The live metadata tensor a metadata argument is a view of, and its name:
    one of its dtype with room for its span: one whose name the slot holds or
    that holds the slot's first, then one of its shape and stride, then the
    tightest."""
    need = _span(spec["shape"], spec["stride"])
    fits = [
        (name, t)
        for name, t in live.items()
        if str(t.dtype) == spec["dtype"] and _room(t) >= need
    ]
    if not fits:
        raise ValueError(f"ATOM's live {what} has no room for {slot} {_text(spec)}")
    slot = slot.split(".")[0]
    name, t = min(
        fits,
        key=lambda f: (
            not (f[0] in slot or slot in f[0]),
            _spec(f[1]) != _spec_of(spec),
            _room(f[1]),
        ),
    )
    return name, t.as_strided(spec["shape"], spec["stride"], t.storage_offset())


def _kv(spec, slot, live) -> tuple[str, torch.Tensor | None]:
    """The live KV tensor a KV argument is a view of, and its name, picked as
    `_meta` picks among those whose layout past the first dimension's size
    is its own (`_layout`), when any is: the capture's pool sized that one;
    else among the last layers, `name.i`, of tensors stacked by layer whose
    layer has that layout: ATOM passes a layer of DeepSeek-V4's compressor
    state, and another field's base at this one's slot stride reaches past
    the pool; or why none fits, and None: a tensor of another layout is
    another buffer than the one ATOM passes."""
    want = _layout(spec["shape"], spec["stride"])
    own = {n: t for n, t in live.items() if _layout(t.shape, t.stride()) == want}
    own = own or {
        f"{n}.{len(t) - 1}": t[-1]
        for n, t in live.items()
        if t.dim() > 2 and len(t) and _layout(t[-1].shape, t[-1].stride()) == want
    }
    if not own:
        return f"ATOM's KV holds no tensor laid out as {slot} {_text(spec)}", None
    try:
        return _meta(spec, slot, own, "KV")
    except ValueError as error:  # the leaf refuses, not the point
        return str(error), None


def _layout(shape, stride) -> list:
    """Each dimension's size and stride, but the first's size and any
    dimension of one."""
    dims = zip(range(len(shape)), shape, stride)
    return [(n if i else None, s) for i, n, s in dims if n != 1 or not i]


def _held(runner, registry) -> dict[str, torch.Tensor]:
    """Each tensor ATOM's KV allocation bound, by name: the runner attributes
    it set (`kv_held`), a list's items as `name.i`, each field of a KV
    `registry` entry as `layer_<n>.field`, and the block views of the KV
    planes it bound on DeepSeek-V4's attention compressors, which no runner
    attribute holds, as `<module>.kv_cache` and `<module>.kv_cache_rope`; an
    indexer's compressor writes a layer of a pool the runner holds."""
    named = [
        (f"{key}.{field}", t)
        for key, entry in (registry or {}).items()
        for field, t in vars(entry).items()
    ]
    model = getattr(runner, "model", None)
    for path, module in model.named_modules() if model is not None else ():
        if path.endswith("attn.compressor"):
            named += [
                (f"{path}.{a}", getattr(module, a, None))
                for a in ("kv_cache", "kv_cache_rope")
            ]
    for name, value in runner.kv_held.items():
        if isinstance(value, (list, tuple)):
            named += [(f"{name}.{i}", t) for i, t in enumerate(value)]
        else:
            named.append((name, value))
    return {name: t for name, t in named if torch.is_tensor(t)}


def _state(runner, held) -> set[str]:
    """The names of the `held` tensors that hold per-request state, each also
    as `_kv` names its last layer: the fields of an arena of a request's state
    an entry that the runner holds (DeepSeek-V4's compressor state). A KV
    registry entry of it caps a point's placements at its slots instead."""
    from atom.model_ops.attentions.pool_layout.entry_arena import SplitEntryMajorArena

    state = []
    for arena in runner.kv_held.values():
        if isinstance(arena, SplitEntryMajorArena):
            state += [p.view(f.name) for p in arena.arenas for f in p.fields]
    at = {(t.data_ptr(), t.shape, t.stride()) for t in state}
    return {
        n
        for name, t in held.items()
        if (t.data_ptr(), t.shape, t.stride()) in at
        for n in (name, f"{name}.{len(t) - 1}")
    }


def _function(name: str):
    """The callable a leaf names: `ns.op.overload`, or a lifted `module:qualname`;
    a functional collective waited on, as the in-place one ATOM calls is, on
    this rank's group of the coordinator it was captured on (`_on_coordinator`);
    a lifted collective that takes a group on the one ATOM calls it on
    (`DP_GROUP`), refused when ATOM's group is not known."""
    if ":" in name:
        module, qualname = name.split(":")
        fn, path = importlib.import_module(module), qualname.split(".")
    else:
        namespace, rest = name.split(".", 1)
        fn, path = getattr(torch.ops, namespace), rest.split(".")
    for attr in path:
        fn = getattr(fn, attr)
    if name.startswith(COLLECTIVES):
        if name in DP_GROUP:  # the capture drops the group
            from aiter.dist.parallel_state import get_dp_group

            return functools.partial(fn, group=get_dp_group())
        if ":" in name and "group" in inspect.signature(fn).parameters:
            raise ValueError(f"{name} takes a group no leaf name maps to")
        if name.startswith("_c10d_functional."):
            wait, fn = torch.ops._c10d_functional.wait_tensor, _on_coordinator(name, fn)
            return lambda **kwargs: wait(fn(**kwargs))
    return fn


def _on_coordinator(name: str, fn):
    """`fn` on this rank's device group of the aiter coordinator a call's
    `group_name` holds (`tp:0`), whatever torch named that group here; refused
    naming the coordinator when this rank has none by that name."""

    def call(**kwargs):
        from aiter.dist.parallel_state import _groups

        ref = _groups.get(coordinator := kwargs["group_name"])
        if (group := ref and ref()) is None:
            raise ValueError(
                f"{name} on coordinator {coordinator!r}, which rank "
                f"{torch.distributed.get_rank()} does not hold: {sorted(_groups)}"
            )
        return fn(**kwargs | {"group_name": group.device_group.group_name})

    return call


def _kernels(fn, kwargs) -> tuple[list[str], float]:
    """The kernels one call of `fn` launches, in order, and their device time
    in ns: its device activity except memory copies, which the profiler
    records at some calls of a leaf and misses at others; started on every
    rank together (`_together`)."""
    from torch.profiler import ProfilerActivity, profile

    _together()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(**kwargs)
        torch.cuda.synchronize()
    device = [
        e
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith("Memcpy")
    ]
    device.sort(key=lambda e: e.time_range.start)
    ns = 1e3 * sum(e.time_range.elapsed_us() for e in device)
    return [e.name for e in device], ns


def _replay(fn, calls) -> list[float]:
    """Each timed replay of one graph of `calls`, in ns per call, each started
    on every rank together (`_together`), the capture refused on every rank
    when any rank's refused, and nothing it left read by a later one
    (`_contained`), captured once the memory earlier leaves freed is cleared
    (`_release`). Over several ranks, a decode step's graph is captured as
    ATOM captures one, each collective on the buffers it registered."""
    from atom.compass.bench.ranks import cpu_group, refused_together
    from atom.utils import forward_context

    registered = cpu_group() is not None and not calls[0][1].context.is_prefill
    if registered:
        from aiter.dist.parallel_state import graph_capture
    _release()  # the capture's own release then has nothing to return
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with (
            refused_together(),
            graph_capture() if registered else contextlib.nullcontext(),
            _contained(torch.cuda.graph(graph, capture_error_mode="thread_local")),
        ):
            for kwargs, snap in calls:
                forward_context._forward_context = snap
                fn(**kwargs)
    if any("Graph is empty" in str(w.message) for w in caught):
        return [0.0] * REPEATS
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for i in range(WARMUPS + REPEATS):
        _together()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if i >= WARMUPS:
            times.append(ms * 1e6 / len(calls))
        time.sleep(ms * REST / 1e3)
    return times


def _release() -> None:
    """The caching allocator's free memory returned to the driver, then a wait,
    at most `CLEAR_S`, until the card's VRAM in use falls by it, added to
    `_waited`. The driver clears VRAM a process returns in the background,
    about 31 GiB/s on MI308X, and a memory-bound leaf timed during the clear
    read up to 1.20 of its time after it."""
    used, held = _vram_used(), torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    if used is None:
        return
    target = used - (held - torch.cuda.memory_reserved()) + CLEAR_SLACK
    start = time.monotonic()
    while (over := _vram_used() > target) and time.monotonic() - start < CLEAR_S:
        time.sleep(0.005)
    _waited["s"] += time.monotonic() - start
    _waited["capped"] += over


def _vram_used() -> int | None:
    """The bytes in use on this process's GPU as its driver counts them, or
    None when the process has no GPU in use."""
    from atom.compass.drive.identity import PCI

    if not torch.cuda.is_initialized():
        return None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return int(Path(PCI, bus, "mem_info_vram_used").read_text())


@contextlib.contextmanager
def _contained(capture):
    """`capture` entered with no garbage collection, and when it fails,
    undone what it leaves behind for later leaves: its stream current, the
    generator in capture, and the capture HIP leaves open on its stream when
    a call invalidated it. A failed capture's graph can sit in a reference
    cycle, and a graph freed during a capture synchronizes the device, which
    ends the process."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        with capture:
            yield
    except BaseException:
        if torch.cuda.current_stream() == capture.capture_stream:
            capture.stream_ctx.__exit__(None, None, None)
        with torch.cuda.stream(capture.capture_stream):
            stuck = torch.cuda.is_current_stream_capturing()
        if stuck:
            _end_capture(capture.capture_stream)
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        generator.graphsafe_set_state(generator.clone_state())
        raise
    finally:
        if collecting:
            gc.enable()


def _end_capture(stream) -> None:
    """End the capture HIP keeps open on `stream` after its end refused it, by
    beginning and ending an empty one on it, through the HIP runtime this
    process loaded."""
    import ctypes

    maps = Path("/proc/self/maps").read_text().splitlines()
    hip = ctypes.CDLL(next(m.split()[-1] for m in maps if "libamdhip64" in m))
    handle, graph = ctypes.c_void_p(stream.cuda_stream), ctypes.c_void_p()
    if hip.hipStreamBeginCapture(handle, 2) or hip.hipStreamEndCapture(  # relaxed
        handle, ctypes.byref(graph)
    ):
        raise RuntimeError(f"the capture HIP left open on {stream} does not end")
    hip.hipGraphDestroy(graph)


def _together() -> None:
    """Over several ranks, a barrier on the host, then one all-reduce each
    rank's stream waits on, so what is launched next starts on every rank's
    device as on the others': a collective's time holds no peer's lateness."""
    from atom.compass.bench.ranks import cpu_group

    if (group := cpu_group()) is None:
        return
    from aiter.dist.parallel_state import get_world_group

    torch.distributed.barrier(group=group)
    device, flag = get_world_group().device_group, torch.zeros(1, device="cuda")
    torch.distributed.all_reduce(flag, group=device)


def _span(shape, stride) -> int:
    """Elements a view of `shape` and `stride` reaches from its offset."""
    if 0 in shape:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride))


def _room(t) -> int:
    return t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()


def _dtype(text: str) -> torch.dtype:
    return getattr(torch, text.removeprefix("torch."))


def _spec(t) -> tuple:
    return tuple(t.shape), str(t.dtype), tuple(t.stride())


def _spec_of(spec: dict) -> tuple:
    return tuple(spec["shape"]), spec["dtype"], tuple(spec["stride"])


def _text(spec: dict) -> str:
    return f"{spec['shape']} {spec['dtype']} stride {spec['stride']}"


def _random(spec, device) -> torch.Tensor:
    """A tensor of `spec`'s shape, dtype and stride, random over its span when
    floating, else zero: an integer one may index a dimension of one. Packed
    fp4, which no cast reaches, takes random bytes: every e2m1 code is finite."""
    dtype = _dtype(spec["dtype"])
    n = _span(spec["shape"], spec["stride"])
    if dtype == torch.float4_e2m1fn_x2:
        flat = torch.randint(256, (n,), dtype=torch.uint8, device=device).view(dtype)
    elif dtype.is_floating_point:
        flat = torch.randn(n, device=device).mul_(0.1).to(dtype)
    else:
        flat = torch.zeros(n, device=device, dtype=dtype)
    return flat.as_strided(spec["shape"], spec["stride"])


def _clone(t) -> torch.Tensor:
    """`t` on storage of its own, its span copied, its shape and stride kept."""
    n = _span(t.shape, t.stride())
    flat = t.as_strided((n,), (1,), t.storage_offset()).clone()
    return flat.as_strided(t.shape, t.stride())


def _deep(x, memo, clones=None):
    """`x` with every tensor it reaches through dataclasses, ATOM objects,
    dicts and lists cloned, aliases kept; with `clones`, a tensor whose values
    have not changed since the last clone of its live view takes that clone."""
    if id(x) in memo:
        return memo[id(x)]
    if torch.is_tensor(x):
        key = x.data_ptr(), x.shape, x.stride(), x.dtype
        twin = None if clones is None else clones.get(key)
        if twin is None or not torch.equal(twin, x):
            twin = _clone(x)
        memo[id(x)] = twin
        if clones is not None:
            clones[key] = twin
    elif isinstance(x, dict):
        memo[id(x)] = {k: _deep(v, memo, clones) for k, v in x.items()}
    elif isinstance(x, list):
        memo[id(x)] = [_deep(v, memo, clones) for v in x]
    elif dataclasses.is_dataclass(x) or type(x).__module__.startswith("atom.utils"):
        if not hasattr(x, "__dict__"):
            return x
        memo[id(x)] = twin = copy.copy(x)
        vars(twin).update({k: _deep(v, memo, clones) for k, v in vars(x).items()})
    else:
        return x
    return memo[id(x)]


def _own(module):
    """The tensors `module` holds itself: attributes, parameters and buffers."""
    held = [*vars(module).values(), *module._parameters.values()]
    return [t for t in [*held, *module._buffers.values()] if torch.is_tensor(t)]


def _fill(module, device) -> None:
    """`module`'s meta parameters moved to `device` and filled as ATOM's
    `--load_dummy xavier` fills a model's."""
    meta = [
        (m, n, p)
        for m in module.modules()
        for n, p in m._parameters.items()
        if p is not None and p.is_meta
    ]
    if not meta:
        return
    from atom.model_loader.loader import initialize_dummy_weights

    for m, n, p in meta:
        t = torch.empty_like(p, device=device)
        m._parameters[n] = torch.nn.Parameter(t, requires_grad=False)
    initialize_dummy_weights(module, "xavier")


def _module_copy(module):
    """`module` with its own tensors, parameters and buffers cloned."""
    twin = copy.copy(module)
    for name, value in vars(module).items():
        if torch.is_tensor(value):
            vars(twin)[name] = _clone(value)
        elif name in ("_parameters", "_buffers"):
            vars(twin)[name] = {
                k: v if v is None else _clone(v) for k, v in value.items()
            }
    return twin


def _stateful(held) -> bool:
    return bool(getattr(held, "per_request_state", False))


def _fields(held):
    for value in vars(held).values() if held is not None else ():
        if torch.is_tensor(value) and value.dim():
            yield value


def keys(config, run: dict):
    """The `op_measurement` key of a point, `(shape, context, leaf)`, measured
    under `config` on the platform `run` reads."""
    from atom.compass.artifacts.keys import LAYOUT, platform_key
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.memory import key_fields

    fields = run_fields(run)
    platform = platform_key(fields["device"], run["stack"], run["ptl"]).canonical()
    layout = {k: v for k, v in key_fields(config).items() if k in LAYOUT}

    def key(shape, context, leaf) -> Key:
        point = json.dumps([str(shape), context, leaf]).encode()
        return Key.of(
            Kind.OP_MEASUREMENT,
            model=fields["model"],
            **layout,
            machine_spec="sha256:" + hashlib.sha256(platform.encode()).hexdigest(),
            source_root=fields["source_root"],
            point="sha256:" + hashlib.sha256(point).hexdigest(),
        )

    return key


def publish(store, key, config, run: dict, command: str, results, out: Path):
    """Write `results` to ``ops.jsonl`` in `out` and each measured one to
    `store` at its `key`."""
    from atom.compass.fit.publish import conditions, ranks
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.step_grid import keyed

    readings = conditions(
        {**run_fields(run), "engine_config": keyed(config)}
        | {k: run[k] for k in ("stack", "host_key", "ptl")}
    )
    roots = roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / "ops.jsonl").open("w") as lines:
        for record in results:
            at = key(record["shape"], record["context"], record["leaf"])
            lines.write(json.dumps(record | {"token": at.token()}) + "\n")
            if "refused" not in record:
                body, every = json.dumps(record).encode(), ranks(at)
                store.publish(
                    at,
                    provenance=Provenance(command, now, roots),
                    topology=every[0].topology,
                    conditions=readings,
                    gates=GateState.of(),
                    members={member_name("point", r, "json"): body for r in every},
                )


#: A ladder keeps a value only this many times the last it kept, and the largest.
LADDER_STEP = 1.25
#: The spread of a leaf's cold times across contexts, or a prefill leaf's
#: across rows, over the largest, past which its kind depends on them; a
#: repeat's spread stays within 4%.
CONTEXT_MOVES = 0.10
#: How far the line through a decode point's two neighbours may price it off its
#: measured cold time, over that time as `fit ops` reads a held-out error, when
#: the three launched one kernel, before the plan halves the gaps either side;
#: and how far a gap's ends' cold times may differ, over the lower, before the
#: plan halves it where the leaf's kind steps (`_apart`). A collective's point
#: over its message bytes is read as a decode one.
BENDS = 0.05
#: The spread of a leaf's cold times across contexts, or a prefill leaf's
#: across rows, in ns, at or under which its kind does not depend on them
#: whatever its share: one 3.5 us elementwise kernel on MI308X reads in two
#: modes about 0.5 us apart, either one at any shape.
CONTEXT_FLOOR_NS = 1000


def plan(store, key, config, run, args, command: str, out: Path, atom_args=()) -> None:
    """Measure the points the steps of `--plan`'s record directories call for,
    round by round, and write every point held to plan.jsonl in `out`."""
    from atom.compass.capture import population_steps
    from atom.compass.capture.forward import decode
    from atom.compass.runner.overrides import schedulable_ladder

    steps = list(population_steps(args.plan))
    contexts = [
        c for s, line in steps if s.phase == "decode" for c in _rows(line, s.seqs)
    ]
    if not contexts:
        raise ValueError("the --plan records hold no decode step to read contexts of")
    rungs = [decode(r) for r in schedulable_ladder(config)]
    graphs, done, tried = {}, {}, set()

    def measure_round(want, n):
        dependent = _dependent(done.values())
        _load(store, config, args.store, {s for s, _ in want}, graphs, None, atom_args)
        todo = []
        for (shape, context), families in want.items():
            found = {}
            for sig, (op, instances) in graphs[shape].items():
                fam = _family(shape, context, op["op"], sig, dependent)[0]
                if (
                    (str(shape), context, sig) in done
                    or families
                    and fam not in families
                ):
                    continue
                if store.directory_for(at := key(shape, context, sig)).exists():
                    entry = store.read(at)
                    held = entry.read_member("point", entry.topology.ranks()[0], "json")
                    done[str(shape), context, sig] = json.loads(held)
                else:
                    found[sig] = op, instances
            todo += [(shape, context, found)] if found else []
        print(f"round {n}: {sum(len(f) for *_, f in todo)} points to measure")
        gpus = len(run["device"])
        results = _measured(config, todo, args.llc_mib << 20, gpus)
        (out / f"round{n}").mkdir(exist_ok=True)
        publish(store, key, config, run, command, results, out / f"round{n}")
        done.update(((r["shape"], r["context"], r["leaf"]), r) for r in results)

    measure_round(
        {(s, c): None for s in rungs for c in (min(contexts), max(contexts))}, 0
    )
    dependent, want = _dependent(done.values()), defaultdict(set)
    for shape in rungs:
        for sig, (op, _) in graphs[shape].items():
            if dependent.get(op["op"], True) and not exchange(op):
                for c in _ladder(contexts):
                    want[shape, c].add(_family(shape, c, op["op"], sig, dependent)[0])
    prefills, cached = set(), defaultdict(list)
    for s, line in steps:
        if s.phase == "prefill":
            prefills.add(s)
            cached[s.seqs] += _cached(line)
    _load(store, config, args.store, prefills, graphs, None, atom_args)
    seen = defaultdict(dict)
    for shape in sorted(prefills):
        for sig, (op, _) in graphs[shape].items():
            fam, tokens = _family(shape, None, op["op"], sig, dependent)
            seen[fam].setdefault(tokens, (shape, op["op"], sig))

    def cached_ladder(top_only: bool) -> dict:
        """The points of each prefill family over its rows' cached tokens, of
        a kind `_dependent` reads as depending on context: only the most, its
        reading's far end, while the kind has no cached point to read. None whose
        first row, the cached tokens and its share of the token count, holds
        more than `max_model_len` tokens, a row ATOM's runner cannot take."""
        dependent, want = _dependent(done.values()), defaultdict(set)
        for at in seen.values():
            for tokens in _ladder(at):
                shape, op, sig = at[tokens]
                ladder = _ladder(cached[shape.seqs])
                room = config.max_model_len - -(-shape.tokens // shape.seqs)
                for c in ladder[-1:] if top_only else ladder:
                    fam = _family(shape, c, op, sig, dependent)[0]
                    if c and c <= room and _cut(fam):
                        want[shape, c].add(fam)
        return want

    for fam, at in seen.items():
        for tokens in _ladder(at):
            want[at[tokens][0], None].add(fam)
    for point, fams in cached_ladder(top_only=True).items():
        want[point] |= fams
    n = 1
    while want:
        measure_round(want, n)
        n += 1
        want = cached_ladder(top_only=False) if n == 2 else {}
        want = want or _densify(
            _families(done.values(), _dependent(done.values())), tried
        )
    (out / "plan.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in done.values())
    )
    kinds = Counter(r["op"] for r in done.values() if "refused" not in r)
    print(json.dumps({"points per leaf kind": dict(kinds.most_common())}, indent=1))
    print(f"{sum(kinds.values())} of {len(done)} points measured in {n} rounds")


def _coverage_main(argv: list[str], atom_args: list[str]) -> dict:
    from atom.compass.bench import ranks
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture import driverless

    p = argparse.ArgumentParser(prog=f"{PROG} --coverage")
    p.add_argument("--coverage", action="append", required=True, metavar="RECORD")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="the --plan run's --out")
    args = p.parse_args(argv)
    out = Path(args.out)
    lines = (out / "plan.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    store = ArtifactStore(args.store)
    arch = json.loads((out / "run.json").read_text())["device"][0]["arch"]
    if arch not in driverless.DEVICES:
        declared = sorted(driverless.DEVICES)
        raise ValueError(f"no declared device for {arch}; declared: {declared}")
    with tempfile.TemporaryDirectory() as tmpdir:
        # aiter asks rocminfo for an arch at import, which the config reaches.
        driverless._declare_arch(tmpdir, arch)
        config = atom_config(atom_args)
        ranks.layout(config)
        _eager(config)
        names = args.coverage
        return coverage(store, args.store, config, names, records, arch, atom_args)


def _eager(config) -> None:
    """Refuse `config` unless ATOM runs it eager: `trace ops` captures and keys
    every `op_graph` eager, so a graph-mode key names no graph it publishes."""
    if not config.enforce_eager:
        raise ValueError(
            "ATOM's arguments run the model in CUDA graphs, and trace ops "
            "publishes eager op_graphs only; measure with --enforce-eager"
        )


def coverage(store, root, config, names, records, arch, atom_args=()) -> dict:
    """How the measured points among `records` cover the steps of the record
    directories `names`: the steps and those covered by phase, and per leaf
    kind the steps it left out by why. A step is covered when each leaf of its
    `op_graph` lies on a point or between two of one kernel (`_bracket`): a
    decode leaf of a kind that depends on context over its rows' lowest to
    highest context (`_rows`), any other decode leaf at its width, a prefill
    leaf at its tokens, and one of a kind that depends on context over its
    rows' cached tokens too (`_bracket_cached`). The graphs are `arch`'s."""
    from atom.compass.capture import population_steps

    steps = list(population_steps(names))
    dependent = _dependent(records)
    axes, cuts = {}, defaultdict(dict)
    for fam, points in _families(records, dependent).items():
        axes[fam] = [p[0] for p in points], [p[1] for p in points]
        if cut := _cut(fam):
            cuts[cut[0]][cut[1]] = axes[fam]
    graphs, report, out = {}, defaultdict(Counter), defaultdict(Counter)
    _load(store, config, root, {s for s, _ in steps}, graphs, arch, atom_args)
    families = {
        s: [
            (op["op"], *_family(s, None, op["op"], sig, dependent))
            for sig, (op, _) in g.items()
        ]
        for s, g in graphs.items()
    }
    for shape, line in steps:
        whys = {}
        for name, fam, tokens in families[shape]:
            moves = dependent.get(name, True)
            cut = _cut(fam)
            if cut and cut[0] in cuts:
                at, (lo, hi) = cuts[cut[0]], _cached(line)
                why = (
                    "a cached row, measured on fresh rows only"
                    if hi and at.keys() == {0}
                    else _bracket_cached(at, tokens, lo, hi)
                )
            elif fam not in axes:
                why = "no measured point"
            elif shape.phase == "decode" and not _exchanged(fam):
                why = _bracket(*axes[fam], *_rows(line, shape.seqs)) if moves else ""
            else:
                why = _bracket(*axes[fam], tokens, tokens)
            if why:
                whys[name] = why
        report[shape.phase].update(steps=1, covered=not whys)
        for name, why in whys.items():
            out[name][why] += 1
    total = sum(report.values(), Counter())
    return {
        "steps": total["steps"],
        "covered": total["covered"],
        "share": total["covered"] / max(total["steps"], 1),
        "by phase": {phase: dict(c) for phase, c in report.items()},
        "left out, by leaf kind": {k: dict(v) for k, v in sorted(out.items())},
    }


def _rows(line, rung: int) -> tuple[int, int]:
    """A step's lowest and highest row context: at least 1, the least over
    `k`, the powers of two up to the rung, of `rowmod(k)` per row of the
    `rung` a class `i mod k` holds (`k = 1`: the mean over the rung's rows,
    padding included); and the max."""
    from atom.compass.backends.step_linear import rowmod

    contexts = [row[1] for row in line["rows"]]
    ks = [1 << i for i in range(rung.bit_length())]
    lo = min(rowmod(contexts, k) // -(-rung // k) for k in ks)
    return max(lo, 1), max(contexts)


def _cached(line) -> tuple[int, int]:
    """A step's rows' mean and max cached tokens."""
    cached = [row[1] - row[0] for row in line["rows"]]
    return sum(cached) // len(cached), max(cached)


def _ladder(values) -> list:
    """`values` in order, each kept only `LADDER_STEP` past the last kept,
    and the largest."""
    kept = []
    for v in sorted(set(values)):
        if not kept or v >= kept[-1] * LADDER_STEP:
            kept.append(v)
    return kept + [v] if kept[-1] != v else kept


def _shape(text: str):
    from atom.compass.capture.forward import decode, prefill

    counts = [int(n) for n in re.findall(r"=(\d+)", text)]
    return prefill(*counts) if text.startswith("prefill") else decode(*counts)


def _abstract(value, n: int):
    """`value` with every int equal to `n` replaced by ``"T"``."""
    # ponytail: an unrelated int equal to n (a stride of 1 at T = 1) is
    # abstracted too, so that shape's leaf joins another family and its
    # points bracket nothing; symbolic slots in the contract would end it.
    if isinstance(value, dict):
        return {k: _abstract(v, n) for k, v in value.items()}
    if isinstance(value, list):
        return [_abstract(v, n) for v in value]
    return "T" if type(value) is int and value == n else value


#: Leaf kinds that exchange rows between ranks (`exchange`).
COLLECTIVES = (
    "atom.model_ops.moe:",
    "mori.ops.dispatch_combine:",
    "aiter.all_reduce_.",
    "_c10d_functional.all_",
    "_c10d_functional.broadcast.",
    "_c10d_functional.reduce_scatter",
)
#: Lifted collectives every ATOM caller passes aiter's DP group.
DP_GROUP = ("atom.model_ops.moe:all_gatherv", "atom.model_ops.moe:reduce_scatterv")


def exchange(op: dict) -> list | None:
    """A collective leaf's family, decode and prefill alike, and its message
    bytes, the axis it is priced over: the family is its kind, each tensor
    argument's dtype and row width, its output rows' widths and its arguments
    neither tensors nor numbers; the bytes the largest of its tensors and its
    outputs, at its first tensor's dtype. None for any other leaf."""
    import math

    if not op["op"].startswith(COLLECTIVES):
        return None
    tensors = [s for _, s in _specs(op["args"])]
    size = _dtype(tensors[0]["dtype"]).itemsize
    nbytes = max(
        [math.prod(t["shape"]) * _dtype(t["dtype"]).itemsize for t in tensors]
        + [math.prod(o) * size for o in op["out"]]
    )
    fam = {
        "op": op["op"],
        "tensors": [[t["dtype"], t["shape"][1:], t["stride"][1:]] for t in tensors],
        "out": [o[1:] for o in op["out"]],
        "flags": {s: v for s, v in op["args"].items() if isinstance(v, (bool, str))},
    }
    return [json.dumps(["exchange", fam], sort_keys=True), nbytes]


def _exchanged(fam: str) -> bool:
    return json.loads(fam)[0] == "exchange"


def _family(shape, context, op: str, sig: str, dependent: dict):
    """The family a leaf's point belongs to, and where on its axis: a decode
    leaf at its width over context; a prefill leaf over its tokens, at its rows
    unless its kind's prefill points show it does not move with them, and at
    its rows and cached tokens when its kind depends on context, as its
    prefill points read it, else its decode points, else by default; a
    collective over its message bytes (`exchange`)."""
    if op.startswith(COLLECTIVES):
        return tuple(exchange(json.loads(sig)))
    if shape.phase == "decode":
        return json.dumps([str(shape), sig]), context
    leaf = _abstract(json.loads(sig), shape.tokens)
    if not dependent.get(("prefill", op), dependent.get(op, True)):
        rows = shape.seqs if dependent.get(("rows", op), True) else None
        return json.dumps(["prefill", rows, leaf], sort_keys=True), shape.tokens
    fam = ["prefill", shape.seqs, leaf, context or 0]
    return json.dumps(fam, sort_keys=True), shape.tokens


def _cut(fam: str):
    """A prefill family of a kind that depends on context as the family over
    every cached-token count, and its count; None for any other."""
    *base, cached = json.loads(fam)
    return (json.dumps(base, sort_keys=True), cached) if len(base) == 3 else None


def _dependent(records) -> dict:
    """Whether each leaf kind timed at two contexts of one shape depends on
    context: its kernels differ there, or its cold times by more than
    `CONTEXT_MOVES` of the largest and `CONTEXT_FLOOR_NS`. A decode kind, read
    at two row contexts, is keyed by its name; a prefill kind, read fresh and
    over cached tokens, by ``("prefill", name)``. Keyed ``("rows", name)``,
    whether a prefill kind timed at two row counts of one token count and
    cached tokens moves with its rows."""
    by = defaultdict(list)
    for r in records:
        if "refused" in r:
            continue
        if r["shape"].startswith("decode"):
            by[r["op"], r["shape"], r["leaf"]].append(r)
        else:
            by[("prefill", r["op"]), r["shape"], r["leaf"]].append(r)
            at = _shape(r["shape"]).tokens, r["context"]
            by[("rows", r["op"]), at, r["leaf"]].append(r)
    kinds = {}
    for (kind, *_), points in by.items():
        if len(points) > 1:
            cold = [p["cold_ns"] for p in points]
            kernels = {tuple(p["kernels"]) for p in points}
            spread = max(cold) - min(cold)
            moved = spread > max(CONTEXT_MOVES * max(cold), CONTEXT_FLOOR_NS)
            kinds[kind] = kinds.get(kind, False) or moved or len(kernels) > 1
    return kinds


def _families(records, dependent) -> dict:
    """Each family's measured points, `(at, kernels, record)` in `at` order."""
    found = defaultdict(list)
    for r in records:
        if "refused" not in r:
            shape = _shape(r["shape"])
            fam, at = _family(shape, r["context"], r["op"], r["leaf"], dependent)
            found[fam].append((at, tuple(r["kernels"]), r))
    return {fam: sorted(points, key=lambda p: p[0]) for fam, points in found.items()}


def _bracket(ats, kernels, lo, hi) -> str:
    """'' if `lo` to `hi` lies on or between two points, `ats` in order, that
    launched the same `kernels` as every point between them; else why not."""
    band = _band(ats, lo, hi)
    if band is None:
        return "outside the measured range"
    return "between two kernels" if len(set(kernels[band])) > 1 else ""


def _bracket_cached(cuts: dict, tokens, lo, hi) -> str:
    """`_bracket` over two axes: '' if `lo` to `hi` cached tokens lie on or
    between two of `cuts`' cached-token counts, and `tokens` on or between two
    points of each count from the one to the other, every point of one kernel;
    else why not."""
    cached = sorted(cuts)
    if (band := _band(cached, lo, hi)) is None:
        return "outside the measured range"
    kernels = set()
    for ats, ks in (cuts[c] for c in cached[band]):
        if (inner := _band(ats, tokens, tokens)) is None:
            return "outside the measured range"
        kernels.update(ks[inner])
    return "between two kernels" if len(kernels) > 1 else ""


def _band(ats, lo, hi) -> slice | None:
    """The points of `ats`, in order, from the last at or below `lo` to the
    first at or above `hi`; None when either is missing."""
    i, j = bisect.bisect_right(ats, lo) - 1, bisect.bisect_left(ats, hi)
    return None if i < 0 or j == len(ats) else slice(min(i, j), max(i, j) + 1)


def _densify(families, tried: set) -> dict:
    """The point halfway between each two neighbours of a family, or of a
    prefill leaf's token count over cached tokens (`_cut`), more than one
    apart, unless `tried`, that launched different kernels or, in a decode
    family, lie either side of a point `_bent` off the line through theirs or
    are `_apart` where the leaf's kind steps: some decode family of the kind
    holds two adjacent contexts of one kernel that are `_apart`; between two
    points of a collective's family, the step of the rows halfway, a decode
    when the farther is one, else a prefill of one sequence. A collective's
    family is `_bent` too, each message size at the mean of its readings, as
    `fit ops` merges them."""
    from atom.compass.capture.forward import decode, prefill

    steps = {
        p[2]["op"]
        for points in families.values()
        for p, q in itertools.pairwise(points)
        if p[2]["shape"].startswith("decode") and q[0] - p[0] == 1
        if p[1] == q[1] and _apart(p, q)
    }
    across = defaultdict(list)
    for fam, points in families.items():
        if cut := _cut(fam):
            for tokens, kernels, record in points:
                across[cut[0], tokens].append((cut[1], kernels, record))
    want = defaultdict(set)
    for fam, points in [*families.items(), *across.items()]:
        points = sorted(points, key=lambda p: p[0])
        decoding = points[0][2]["shape"].startswith("decode")
        if isinstance(fam, str) and _exchanged(fam):
            sizes = [[*g] for _, g in itertools.groupby(points, key=lambda p: p[0])]
            mean = (statistics.mean(p[2]["cold_ns"] for p in g) for g in sizes)
            at = ((*g[0][:2], {"cold_ns": ns}) for g, ns in zip(sizes, mean))
            bent = _bent([*at]) if len(sizes) > 2 else ()  # no line under three
        else:
            bent = _bent(points, steps) if decoding else ()
        for (a, ka, record), (b, kb, other) in itertools.pairwise(points):
            mid = (a + b) // 2
            if ka == kb and a not in bent or b - a < 2 or (fam, mid) in tried:
                continue
            tried.add((fam, mid))
            shape = _shape(record["shape"])
            if isinstance(fam, tuple):  # over cached tokens at one token count
                new = json.dumps([*json.loads(fam[0]), mid], sort_keys=True)
                want[shape, mid].add(new)
            elif _exchanged(fam):
                far = _shape(other["shape"])
                rows = (shape.tokens + far.tokens) // 2
                step = decode(rows) if far.phase == "decode" else prefill(rows, 1)
                want[step, other["context"]].add(fam)
            elif shape.phase == "decode":
                want[shape, mid].add(fam)
            else:
                want[prefill(mid, shape.seqs), record["context"]].add(fam)
    return want


def _bent(points, steps=()) -> set:
    """The lower ends of the gaps either side of each middle point of three
    neighbours that launched one kernel and whose cold time the line through
    the outer two's prices more than `BENDS` off, over that cold time; and,
    for a leaf of a kind in `steps`, of each gap whose ends are `_apart`."""
    ends = set()
    if steps and points[0][2]["op"] in steps:
        ends = {p[0] for p, q in itertools.pairwise(points) if _apart(p, q)}
    for p, q, r in zip(points, points[1:], points[2:]):
        y = [point[2]["cold_ns"] for point in (p, q, r)]
        line = y[0] + (y[2] - y[0]) * (q[0] - p[0]) / (r[0] - p[0])
        if p[1] == q[1] == r[1] and abs(line - y[1]) > BENDS * y[1]:
            ends |= {p[0], q[0]}
    return ends


def _apart(p, q) -> bool:
    """Whether two points' cold times differ by more than `BENDS` of the
    lower: the line across them may then price a point between them more
    than `BENDS` off its cold time."""
    lo, hi = sorted(point[2]["cold_ns"] for point in (p, q))
    return hi - lo > BENDS * lo


def _load(store, config, root, shapes, graphs: dict, arch=None, atom_args=()) -> None:
    """Each of `shapes`' leaves into `graphs`, its op_graph captured first when
    the store has none."""
    from atom.compass.artifacts import ArtifactRefusal

    missing = []
    for shape in sorted(shapes - graphs.keys()):
        try:
            graphs[shape] = leaves(graph(store, config, shape, arch)["region"])
        except ArtifactRefusal:
            missing.append(shape)
    if missing:
        _capture(config, missing, root, arch, atom_args)
    for shape in missing:
        graphs[shape] = leaves(graph(store, config, shape, arch)["region"])


def _capture(config, shapes, root, arch=None, atom_args=()) -> None:
    """`trace ops` of `shapes` under ATOM's arguments `atom_args` into the store
    at `root`, in a process of its own that sees no GPU."""
    from atom.compass import capture
    from atom.compass.capture.forward import device_arch
    from atom.model_engine.arg_utils import EngineArgs

    engine = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(atom_args)
    argv = ["--model", config.model, "--tp", str(engine.tensor_parallel_size)]
    argv += ["--arch", arch or device_arch()]
    for s in shapes:
        argv += ["--prefill", str(s.tokens)] if s.phase == "prefill" else ["--width"]
        argv.append(str(s.seqs))
    # The capture sets HIP_VISIBLE_DEVICES itself, and HIP then holds memory on
    # the GPU it names; ROCm's own mask, below HIP's, hides every GPU.
    with patch.dict(os.environ, ROCR_VISIBLE_DEVICES=""):
        _child(capture.main, [*argv, "--store", str(root), "--", *atom_args])


def _journal(path, records: list[dict]) -> list[dict]:
    """`records`, each first appended to the file at `path` when one is named."""
    if path:
        with open(path, "a") as lines:
            lines.writelines(json.dumps(r) + "\n" for r in records)
    return records


def _measured(config, todo, llc: int, gpus: int = 1) -> list[dict]:
    """`measure` of `todo` on `gpus` GPUs (`ranks.measure`), a child of its own
    on each. A launch whose child dies keeps the points every rank finished;
    the first it left, the one it died measuring, refuses, and another launch
    measures the rest: the pool ends every rank once one dies. Two launches in
    a row that die having finished no point raise: a child itself is failing,
    not a point."""
    from atom.compass.bench import ranks

    got, empty = [], False
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp, "points.jsonl")
        paths = (
            [journal] if gpus == 1 else [Path(f"{journal}.{r}") for r in range(gpus)]
        )
        while todo:
            for path in paths:
                path.write_text("")
            try:
                with patch.dict(os.environ, {JOURNAL: str(journal)}):
                    return got + ranks.measure(config, todo, llc, gpus)
            except BrokenProcessPool:
                new = ranks.merge(
                    [[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
                )
                if not new and empty:
                    raise
                got, empty = got + new, not new
            done = {(r["shape"], r["context"], r["leaf"]) for r in got}
            todo = [
                (s, c, {k: v for k, v in found.items() if (str(s), c, k) not in done})
                for s, c, found in todo
            ]
            todo = [point for point in todo if point[2]]
            if todo:  # the first point it left is the one it died measuring
                shape, context, found = todo[0]
                sig = next(iter(found))
                op = found.pop(sig)[0]["op"]
                print(f"{shape}: {op}: the measuring process died", flush=True)
                got.append(
                    {"shape": str(shape), "context": context, "op": op}
                    | {"leaf": sig, "refused": "the measuring process died"}
                )
                todo = todo if found else todo[1:]
    return got


def _child(fn, *args):
    """`fn(*args)` in a spawned process of its own: the device is free again
    at its end, and a capture's process-wide patches end with it."""
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as child:
        return child.submit(fn, *args).result()
