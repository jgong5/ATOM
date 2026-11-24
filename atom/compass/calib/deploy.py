# SPDX-License-Identifier: MIT
"""A design config: the deployment, the law's columns, the noise and the
acceptance criteria, and the segments (#739 section 2).

A config is a JSON document:

    {"name": ..., "about": ...,
     "deployment": {
        "dp": 4,                         # data-parallel ranks (DP attention)
        "rungs": [1, 2, ..., 64],        # decode capture sizes a rank
        "max_num_seqs": 64,              # a rank's running requests
        "max_num_batched_tokens": 32768, # a rank's prefill tokens a step
        "chunk": 65536,                  # --attn-prefill-chunk-size
        "block_size": 128,               # KV block
        "max_model_len": 1048576,
        "kv_tokens": N | {"run_json": PATH},  # a rank's KV pool, in tokens:
                                         # num_kvcache_blocks x block_size of
                                         # a record's run.json engine
        "drafts": 5, "acceptance": 3.51, # speculative drafts and the forced
                                         # acceptance length (0 and 1: none)
        "prefix_unit": 128,              # tokens a prefix-cache hit comes in
                                         # (the block, or a state checkpoint)
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
its own (`ident.variants`); `min_spread` the fit's `moe_imbalance` floor. A
column this tree's step linear backend prices is counted by it (`columns`);
a newer opt-in column plugs in by name.

A segment's kind is one `design.GENERATORS` names, its other entries that
generator's parameters; or ``copy``, ``{"name": ..., "kind": "copy", "of":
NAME}``, the segment NAME generated again under its own name (an anchor
repeated through a record: its traces are new, so it hits no cache the first
left). A config may `extend` another (`resolve`), inserting segments into it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

#: The acceptance criteria #739 section 2.2 freezes, which a config may tighten.
ACCEPTANCE = {
    "vif": 10.0,
    "g": {"decode": 0.01, "prefill": 0.02},
    "g_z": 2.0,
    "rung_steps": 60,
    "levels": 3,
}
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

    def __post_init__(self) -> None:
        if self.dp < 1 or not self.rungs or list(self.rungs) != sorted(set(self.rungs)):
            raise ValueError(f"dp {self.dp}, rungs {self.rungs}: rungs ascend, dp >= 1")
        if self.drafts and self.acceptance < 1.0:
            raise ValueError(f"acceptance {self.acceptance} is below one token a step")
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


def kv_tokens(spec, block_size: int) -> int:
    """A rank's KV pool in tokens: the number, or a run.json's engine
    `num_kvcache_blocks` times its block (`$VAR`s in the path expand)."""
    if isinstance(spec, int):
        return spec
    path = Path(os.path.expandvars(spec["run_json"]))
    engine = json.loads(path.read_text())["engine"]
    return int(engine["num_kvcache_blocks"]) * int(
        engine.get("kv_cache_block_size", block_size)
    )


def parse(raw: dict) -> Config:
    d = dict(raw["deployment"])
    block = int(d["block_size"])
    routing = d.get("moe_routing")
    deployment = Deployment(
        dp=int(d["dp"]),
        rungs=tuple(int(r) for r in d["rungs"]),
        max_num_seqs=int(d["max_num_seqs"]),
        max_num_batched_tokens=int(d["max_num_batched_tokens"]),
        chunk=int(d.get("chunk", d["max_num_batched_tokens"])),
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
    )
    columns = raw["columns"]
    for kind in ("decode", "prefill"):
        if not columns.get(kind) or columns[kind][0] != "step":
            raise ValueError(f"columns.{kind} starts with the intercept, step")
    acceptance = {**ACCEPTANCE, **raw.get("acceptance", {})}
    acceptance["g"] = {**ACCEPTANCE["g"], **raw.get("acceptance", {}).get("g", {})}
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
        raw=raw,
    )


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


def load(path) -> Config:
    path = Path(path)
    return parse(resolve(json.loads(path.read_text()), path.parent))
