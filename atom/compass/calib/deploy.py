# SPDX-License-Identifier: MIT
"""A design config: the deployment, the law's columns, the noise and the
acceptance criteria, and the segments.

A config is a JSON document:

    {"name": ..., "about": ...,
     "deployment": {
        "from_record": RECORD,           # optional: every field below a
                                         # record states (`from_record`)
        "dp": 4,                         # data-parallel ranks (DP attention)
        "tp": 1, "dcp": 1,               # a rank's TP and DCP widths
        "rungs": [1, 2, ..., 64],        # decode capture sizes a rank
        "max_num_seqs": 64,              # a rank's running requests
        "max_num_batched_tokens": 32768, # a rank's prefill tokens a step
        "chunk": 65536,                  # --attn-prefill-chunk-size
        "block_size": 128,               # KV block
        "max_model_len": 1048576,
        "kv_tokens": N | {"run_json": PATH},  # a rank's KV pool, in tokens
                                         # of its sequences: num_kvcache_blocks
                                         # x block_size of a record's run.json
                                         # engine, times its DCP width
        "drafts": 5, "acceptance": 3.51, # speculative drafts and the forced
                                         # acceptance length (0 and 1: none)
        "prefix_unit": 128,              # tokens a prefix-cache hit comes in
                                         # (the block, or a state checkpoint)
        "kda_state": false,              # a KDA model: views state its state
                                         # work (`Deployment.kda`)
        "prefill_buckets": [...],        # prefill token-graph buckets
        "moe_routing": [E, K],           # routed experts and top-k, or null
        "trace_model": "...",            # the traces' model name
        "trace_block": 64},              # tokens a trace hash id stands for
     "columns": {"decode": [...], "decode_variants": [[...], ...],
                 "prefill": [...], "prefill_variants": [[...], ...],
                 "min_spread": 0.2},
     "noise": {"decode": s, "prefill": s},       # relative hardware noise a step
     "acceptance": {"vif": 10, "g": {"decode": 0.01, "prefill": 0.02},
                    "g_z": 2.0, "rung_steps": 60, "levels": 3},
     "range": {"min_context": 1024, "min_prompt": 512},
     "timing": {"server_start_s": 450, "segment_overhead_s": 20},
     "segments": [{"name": ..., "kind": ..., ...}, ...]}

`columns` is the law's column set, an input: `decode` the columns every
decode rung fits and `prefill` the prefill vector's, each starting with the
intercept `step`; `<kind>_variants` the additions the form-selection rule
chooses among (a decode rung's MoE column, an opt-in column), each checked on
its own; `min_spread` the fit's `moe_imbalance` floor. A
column this tree's step linear backend prices is counted by it (`columns`);
a newer opt-in column plugs in by name.

`acceptance` may only tighten LLM-Booster#739's frozen criteria, `ACCEPTANCE`
(`TIGHTER` says which way each tightens): a looser bound, or a criterion it
does not know, is refused, unless `LOOSEN_ENV` is ``1``, a path for tests
whose small designs hold fewer steps a rung.

A deployment field given beside `from_record` overrides the record's
(`parse`); `load`'s `record` instead replaces the config's own fields with
those a record states, so a design follows a deployment change with no edit.

A segment is a name, unique in its config, a kind and that kind's
parameters; its sizes are plain numbers or relative to the deployment
(`count`, `tokens`), so a design stated that way follows a deployment change
with no edit. A config may `extend` another (`resolve`), inserting segments
into it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

#: The worst relative prediction error LLM-Booster#739 allows a vector, in
#: percent as it states it.
G_PERCENT = {"decode": 1, "prefill": 2}
#: The acceptance criteria LLM-Booster#739 freezes before a record, which a
#: config may tighten and never loosen (`acceptance`).
ACCEPTANCE = {
    "vif": 10.0,
    "g": {kind: pct / 100 for kind, pct in G_PERCENT.items()},
    "g_z": 2.0,
    "rung_steps": 60,
    "levels": 3,
}
#: Which way each criterion tightens: -1, a lower bound is tighter (the worst
#: VIF and G allowed); +1, a higher one (G's standard errors, a rung's decode
#: steps, a column's levels).
TIGHTER = {"vif": -1, "g": -1, "g_z": 1, "rung_steps": 1, "levels": 1}
#: The variable that, set to ``1``, lets a config loosen `ACCEPTANCE`: for
#: tests alone, whose small designs hold fewer decode steps a rung.
LOOSEN_ENV = "COMPASS_CALIB_TEST_LOOSEN"
TIMING = {"server_start_s": 450.0, "segment_overhead_s": 20.0}
RANGE = {"min_context": 1024, "min_prompt": 512}


@dataclass(frozen=True)
class Deployment:
    """What the engine can reach, from its configuration alone."""

    dp: int
    rungs: tuple[int, ...]
    max_num_seqs: int
    max_num_batched_tokens: int
    chunk: int
    block_size: int
    max_model_len: int
    kv_tokens: int
    drafts: int = 0
    acceptance: float = 1.0
    prefix_unit: int = 0
    prefill_buckets: tuple[int, ...] = ()
    moe_routing: tuple[int, int] | None = None
    trace_model: str = "calibration"
    trace_block: int = 64
    tp: int = 1
    dcp: int = 1
    #: A model with KDA layers: each row of a step holds a recurrent state, and a
    #: view states the step's state work as `BatchView.kda` (`kda`), which the
    #: law's KDA prefill columns read. Off: views state none, as before.
    kda_state: bool = False

    def __post_init__(self) -> None:
        if self.dp < 1 or not self.rungs or list(self.rungs) != sorted(set(self.rungs)):
            raise ValueError(f"dp {self.dp}, rungs {self.rungs}: rungs ascend, dp >= 1")
        if self.drafts and self.acceptance < 1.0:
            raise ValueError(f"acceptance {self.acceptance} is below one token a step")
        if self.prefix_unit < 0:
            raise ValueError(f"prefix_unit {self.prefix_unit}: a hit is whole tokens")
        if not self.prefix_unit:
            object.__setattr__(self, "prefix_unit", self.block_size)

    @property
    def query(self) -> int:
        """The tokens a decode row verifies a step: the drafts and one."""
        return self.drafts + 1

    @property
    def per_step(self) -> float:
        """The tokens a decode row emits a step on average."""
        return self.acceptance if self.drafts else 1.0

    def rung(self, rows: int) -> int | None:
        """The capture rung a step of `rows` rows on its widest rank replays;
        None (eager) past the widest."""
        return next((r for r in self.rungs if r >= rows), None)

    def rows_of(self, rung: int) -> range:
        """The widest-rank row counts that replay `rung`."""
        i = self.rungs.index(rung)
        return range((self.rungs[i - 1] if i else 0) + 1, rung + 1)

    def kda(self, rows) -> tuple[int, int, int, int] | None:
        """`BatchView.kda` of a step of `rows` (`RequestShape`s) under
        `kda_state`, else None: every row holds a state; no midstep save and no
        fork (a prefix hit resumes through a paged checkpoint restore); one move
        for each prefill row resuming past its first token (that restore, or the
        store of the checkpoint its previous chunk ended on). On GLM-5.3-Flash's
        step A record (TP4, MTP3) this gave a prefill step's real moves exactly
        on 88% of its 392 prefill steps (correlation 0.85)."""
        if not self.kda_state:
            return None
        resumed = sum(
            1 for r in rows if not r.decode and r.context_tokens > r.query_tokens
        )
        return (len(rows), 0, 0, resumed)

    def cached(self, tokens: int) -> int:
        """The prefix-cache hit on `tokens` of history: whole `prefix_unit`s."""
        return tokens // self.prefix_unit * self.prefix_unit


@dataclass(frozen=True)
class Config:
    name: str
    deployment: Deployment
    columns: dict
    noise: dict
    acceptance: dict
    range: dict
    timing: dict
    segments: list
    about: str = ""
    raw: dict = field(default_factory=dict, compare=False)


#: A count relative to the deployment (`count`): rows, requests, concurrency.
COUNT = ("seqs_share", "per_rank", "dp", "minus", "plus")
#: A token size relative to the deployment (`tokens`).
TOKENS = ("kv_share", "len_share", "budget_share", "bucket_share", "over")
#: The share of the model length a share of the KV pool is held under.
LEN_CAP = 0.9


def whole(tokens: float, dep: Deployment) -> int:
    """`tokens` rounded to whole trace blocks, at least one."""
    return dep.trace_block * max(1, round(tokens / dep.trace_block))


def count(spec, dep: Deployment) -> int:
    """A count: a number as given, or ``{"seqs_share": f, "per_rank": k,
    "dp": bool, "minus": m, "plus": p}``, `f` of a rank's `max_num_seqs` and
    `k` more, times the DP ranks with `dp` (a group's total), less `m`, plus `p`."""
    if not isinstance(spec, dict):
        return spec
    if extra := set(spec) - set(COUNT):
        raise ValueError(f"a count states {sorted(extra)}; it takes {list(COUNT)}")
    width = dep.dp if spec.get("dp") else 1
    base = spec.get("seqs_share", 0) * dep.max_num_seqs + spec.get("per_rank", 0)
    n = round(base * width) - spec.get("minus", 0) + spec.get("plus", 0)
    if n < 1:
        raise ValueError(f"the count {spec} is {n} at this deployment")
    return n


def tokens(spec, dep: Deployment, over: int = 1) -> int:
    """A token size: a number as given, or the least of what a dict states,
    in whole trace blocks: `kv_share` of the KV pool split `over` ways (by
    default a drain group's rows), held under `len_share` (default `LEN_CAP`)
    of the model length; `len_share` of the model length, under the pool split
    `over` ways; `budget_share` of the step budget; `bucket_share` of the
    largest token-graph bucket."""
    if not isinstance(spec, dict):
        return spec
    if extra := set(spec) - set(TOKENS):
        raise ValueError(
            f"a token size states {sorted(extra)}; it takes {list(TOKENS)}"
        )
    over = spec.get("over", over)
    sizes = []
    if "kv_share" in spec:
        sizes.append(spec["kv_share"] * dep.kv_tokens / over)
        sizes.append(spec.get("len_share", LEN_CAP) * dep.max_model_len)
    elif "len_share" in spec:
        sizes += [spec["len_share"] * dep.max_model_len, dep.kv_tokens / over]
    if "budget_share" in spec:
        sizes.append(spec["budget_share"] * dep.max_num_batched_tokens)
    if "bucket_share" in spec:
        if not dep.prefill_buckets:
            raise ValueError(f"{spec}: the deployment states no token-graph buckets")
        sizes.append(spec["bucket_share"] * max(dep.prefill_buckets))
    if not sizes:
        raise ValueError(f"the token size {spec} states no share")
    return whole(min(sizes), dep)


def kv_tokens(spec, block_size: int) -> int:
    """A rank's KV pool in tokens of its sequences: the number, or a run.json's
    engine `num_kvcache_blocks` times its block, times the decode context
    parallel width its serve line ran (`pool_tokens`; `$VAR`s in the path
    expand)."""
    if isinstance(spec, int):
        return spec
    path = Path(os.path.expandvars(spec["run_json"]))
    run = json.loads(path.read_text())
    engine = run["engine"]
    return pool_tokens(
        int(engine["num_kvcache_blocks"]),
        int(engine.get("kv_cache_block_size", block_size)),
        int(_serve_args(run).decode_context_parallel_size),
    )


def pool_tokens(blocks: int, block_size: int, dcp: int) -> int:
    """The sequence tokens a rank's KV pool of `blocks` holds. ATOM's
    `num_kvcache_blocks` is one GPU's pool; with decode context parallelism
    (`--decode-context-parallel-size`, `dcp` > 1) each of the `dcp` GPUs holds
    `block_size` tokens of a sequence in a block of its own (a sequence's
    tokens are spread over them), so the pool holds `dcp` times one GPU's
    share. KDA state checkpoints, stored in the same pool, are evictable and
    take no running capacity (`page_unit_checkpoint.PageUnitCheckpointStore`):
    they are not taken off."""
    return blocks * block_size * dcp


def _serve_args(run: dict):
    """A run.json's serve line (``argv`` after ``--``, or all of it) read by
    ATOM's own parser, so an option the line omits takes ATOM's default."""
    from atom.compass.drive.runfile import _engine_args

    argv = [str(a) for a in run.get("argv") or ()]
    return _engine_args(argv[argv.index("--") + 1 :] if "--" in argv else argv)


#: The keys of a routed-expert count and of a top-k in a model's config.json,
#: in the order they are read (`top_k` alone is a sampling default there).
EXPERTS = ("n_routed_experts", "num_local_experts", "num_experts")
TOP_K = ("num_experts_per_tok", "num_experts_per_token", "moe_topk")


def from_record(record) -> dict:
    """The deployment fields a record states, by the names `parse` reads: its
    run.json's engine (over its rank 0's ``dp0.engine.json``) for the ladder,
    the scheduler's bounds, the block, the KV pool (in sequence tokens, so
    times the DCP width: `pool_tokens`) and the token-graph buckets, the DP and
    TP widths and the drafter's width; its serve command line
    (run.json ``argv`` after ``--``), read by ATOM's own parser, so an option
    the line omits takes ATOM's default, for the prefill chunk, the DCP width
    and the forced acceptance (else one plus the mean accepted drafts of
    ``dp0.acceptance.json``); and the model's config.json, where it can be
    read, for the routed experts and top-k, and for a model with KDA layers
    (``linear_attn_config``) the unit its prefix hits land on, refused at a
    checkpoint interval below one token (`resume_unit`: the interval, finer
    where the record's requests resume off it), and `kda_state`. `$VAR`s in the
    path expand. A field the record does not state is left out."""
    record = Path(os.path.expandvars(str(record)))
    if not (record / "run.json").exists() and (record / "real" / "run.json").exists():
        record = record / "real"  # a calibration run's: its server's record
    run = json.loads((record / "run.json").read_text())
    steps = record / "steps"
    engine = {}
    if (steps / "dp0.engine.json").exists():
        engine = json.loads((steps / "dp0.engine.json").read_text())
    engine |= run.get("engine") or {}
    args = _serve_args(run)
    out = {}
    for name, key in (
        ("max_num_seqs", "max_num_seqs"),
        ("max_num_batched_tokens", "max_num_batched_tokens"),
        ("max_model_len", "max_model_len"),
        ("block_size", "kv_cache_block_size"),
        ("dp", "data_parallel_size"),
        ("tp", "tensor_parallel_size"),
    ):
        if key in engine:
            out[name] = int(engine[key])
    if "ladder" in engine:
        out["rungs"] = [int(r) for r in engine["ladder"]]
    if "num_kvcache_blocks" in engine and "block_size" in out:
        out["kv_tokens"] = pool_tokens(
            int(engine["num_kvcache_blocks"]),
            out["block_size"],
            int(args.decode_context_parallel_size),
        )
    if "prefill_buckets" in engine:
        out["prefill_buckets"] = [int(b) for b in engine["prefill_buckets"] or ()]
    out["chunk"] = int(args.attn_prefill_chunk_size)
    out["dcp"] = int(args.decode_context_parallel_size)
    spec = engine.get("speculative")
    out["drafts"] = int(spec[1]) if spec else 0
    if spec and args.spec_decode_acceptance_length is not None:
        out["acceptance"] = float(args.spec_decode_acceptance_length)
    elif spec and (steps / "dp0.acceptance.json").exists():
        seen = json.loads((steps / "dp0.acceptance.json").read_text())["distribution"]
        total = sum(seen.values())
        if total:
            out["acceptance"] = 1 + sum(int(k) * n for k, n in seen.items()) / total
    model = Path(args.model) / "config.json"
    if model.exists():
        hf = json.loads(model.read_text())
        hf = hf.get("text_config", hf)
        experts = next((hf[k] for k in EXPERTS if k in hf), None)
        top_k = next((hf[k] for k in TOP_K if k in hf), None)
        if experts and top_k:
            out["moe_routing"] = [int(experts), int(top_k)]
        if (hf.get("linear_attn_config") or {}).get("kda_layers"):
            interval = int(args.state_checkpoint_interval_tokens)
            if interval < 1:
                raise ValueError(
                    f"{record}: --state-checkpoint-interval-tokens {interval} "
                    "keeps no checkpoint every so many tokens, so a KDA model's "
                    "prefix hit lands on no fixed unit"
                )
            out["prefix_unit"] = resume_unit(steps, interval)
            out["kda_state"] = True
    return out


def resume_unit(steps: Path, interval: int) -> int:
    """The tokens a KDA model's prefix hits come in on the record whose step
    lines `steps` holds: the greatest common divisor of its state checkpoint
    `interval` and every request's resume point, the history its first
    scheduled prefill row starts on (``batch``'s ``n - q``), on each DP rank's
    lines. A model that resumes only at interval rungs keeps the interval; one
    that resumes at prompt-end checkpoints (GLM-5.3-Flash, step A's smoke
    record: 95 of 104 hits off the 8192-token interval) gets the KV block those
    land on. A record without step lines keeps the interval."""
    unit = interval
    for path in sorted(steps.glob("dp*.jsonl")):
        if not path.stem[2:].isdigit():
            continue  # dp<r>.jsonl, not its cost or handoff lines
        seen = set()
        with path.open() as lines:
            for line in lines:
                step = json.loads(line)
                if step.get("is_dummy_run"):
                    continue
                rows = step.get("rows") or ()
                for (req, q, n), row in zip(step.get("batch") or (), rows):
                    if row[2] or req in seen:
                        continue
                    seen.add(req)
                    if n > q:
                        unit = math.gcd(unit, n - q)
    return unit


def deployment_of(d: dict, record=None) -> dict:
    """A config's deployment block as plain fields: `from_record`'s under the
    block's own (which override it), or, given `record`, that record's over the
    block's (which it replaces wherever it states a field)."""
    d = dict(d)
    stated = from_record(d.pop("from_record")) if "from_record" in d else {}
    d = {**stated, **d}
    if record is not None:
        d = {**d, **from_record(record)}
    return d


def parse(raw: dict, record=None) -> Config:
    d = deployment_of(raw["deployment"], record)
    block = int(d["block_size"])
    routing = d.get("moe_routing")
    deployment = Deployment(
        dp=int(d["dp"]),
        rungs=tuple(int(r) for r in d["rungs"]),
        max_num_seqs=int(d["max_num_seqs"]),
        max_num_batched_tokens=int(d["max_num_batched_tokens"]),
        chunk=int(d["chunk"]),
        block_size=block,
        max_model_len=int(d["max_model_len"]),
        kv_tokens=kv_tokens(d["kv_tokens"], block),
        drafts=int(d.get("drafts", 0)),
        acceptance=float(d.get("acceptance", 1.0)),
        prefix_unit=int(d.get("prefix_unit", 0)),
        prefill_buckets=tuple(int(b) for b in d.get("prefill_buckets") or ()),
        moe_routing=tuple(routing) if routing else None,
        trace_model=str(d.get("trace_model", "calibration")),
        trace_block=int(d.get("trace_block", 64)),
        tp=int(d.get("tp", 1)),
        dcp=int(d.get("dcp", 1)),
        kda_state=bool(d.get("kda_state", False)),
    )
    columns = raw["columns"]
    for kind in ("decode", "prefill"):
        if not columns.get(kind) or columns[kind][0] != "step":
            raise ValueError(f"columns.{kind} starts with the intercept, step")
    acceptance = criteria(raw.get("acceptance", {}))
    names = [s["name"] for s in raw["segments"]]
    if len(set(names)) != len(names):
        raise ValueError(f"segment names repeat: {names}")
    return Config(
        name=raw["name"],
        deployment=deployment,
        columns={k: list(v) if isinstance(v, list) else v for k, v in columns.items()},
        noise=dict(raw["noise"]),
        acceptance=acceptance,
        range={**RANGE, **raw.get("range", {})},
        timing={**TIMING, **raw.get("timing", {})},
        segments=list(raw["segments"]),
        about=raw.get("about", ""),
        raw={**raw, "deployment": {**d, "kv_tokens": deployment.kv_tokens}},
    )


def criteria(given: dict) -> dict:
    """`ACCEPTANCE` with the bounds a config's `acceptance`, `given`, states,
    each only tightening #739's (`TIGHTER`): a looser one, or a criterion
    `ACCEPTANCE` does not name, is refused unless `LOOSEN_ENV` is ``1``."""
    out = {**ACCEPTANCE, **given, "g": {**ACCEPTANCE["g"], **given.get("g", {})}}
    if unknown := sorted(set(given) - set(ACCEPTANCE)):
        raise ValueError(f"acceptance states {unknown}; it takes {list(ACCEPTANCE)}")
    if unknown := sorted(set(given.get("g", {})) - set(ACCEPTANCE["g"])):
        raise ValueError(f"acceptance.g states {unknown}; it takes {list(G_PERCENT)}")
    looser = [
        f"{name} {value} (#739: {frozen})"
        for name, value, frozen in (
            *((f"g.{k}", out["g"][k], v) for k, v in ACCEPTANCE["g"].items()),
            *((k, out[k], v) for k, v in ACCEPTANCE.items() if k != "g"),
        )
        if (value - frozen) * TIGHTER[name.split(".")[0]] < 0
    ]
    if looser and os.environ.get(LOOSEN_ENV) != "1":
        raise ValueError(
            f"acceptance loosens #739's frozen criteria: {', '.join(looser)}; a "
            "config may only tighten them"
        )
    return out


def resolve(raw: dict, here: Path) -> dict:
    """`raw` over the config it `extends` (a path beside it): its own entries
    replace the base's, and each of its `insert` entries, ``{"after": NAME or
    null, "segments": [...]}``, puts segments after the base's segment NAME
    (null: first)."""
    if "extends" not in raw:
        return raw
    base = resolve(json.loads((here / raw["extends"]).read_text()), here)
    own = {k: v for k, v in raw.items() if k not in ("extends", "insert")}
    out = {**base, **own}
    segments = list(out["segments"])
    for ins in raw.get("insert", ()):
        names = [s["name"] for s in segments]
        at = 0 if ins.get("after") is None else names.index(ins["after"]) + 1
        segments[at:at] = ins["segments"]
    out["segments"] = segments
    return out


def load(path, record=None) -> Config:
    """The config at `path`, its deployment `record`'s where given (`parse`)."""
    path = Path(path)
    return parse(resolve(json.loads(path.read_text()), path.parent), record)
