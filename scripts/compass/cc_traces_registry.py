"""What each of the six cc-traces cells is configured with, in one place.

`cc_traces_plan.py` prints the commands a cell needs and `cc_traces_run.py`
executes them, and both have so far taken the modelled side's oracle
configuration as free text on the command line: a factory qualname and a list of
`KEY=VALUE` strings the operator types. That is the part of the matrix most
expensive to get wrong -- an artifact resolved from the wrong width prices a
different deployment and nothing in the run says so -- and it is also the part
no test could reach, because it did not exist anywhere in the tree.

So the configuration is written down here, per width, resolved against a root
directory, and the plan reads it. `SOURCE_ONLY_SERVING.md` remains the prose
account of why each option is what it is; this is the same set as data, and
`tests/compass/test_cc_traces_registry.py` holds the two against each other.

Two things are deliberately *not* defaulted:

* **Costs.** Four of the protocol's §5 terms -- `capture`, `calibration`,
  `derivation`, `load` -- are inputs no step of the harness measures. They are
  listed here as owed, with who owes them, rather than carried as a zero that
  would make `replay_ratio` read better than it is.
* **Artifacts that do not exist yet.** `required_artifacts` names every file a
  width needs; `check` reports the ones absent from the root as absent. A cell
  whose artifacts are incomplete is not runnable, and saying so before a lease
  is bought is the entire point.

    python scripts/compass/cc_traces_registry.py --root /workspace/results/poc
    python scripts/compass/cc_traces_registry.py --root ... --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from atom.compass.core.artifacts import resolve_rank_path

#: The matrix, in the order a report should read.
TPS = (1, 2, 4)
CLASSES = ("short", "long")
CELLS = tuple(f"tp{tp}-{klass}" for tp in TPS for klass in CLASSES)

MODEL = "Qwen/Qwen3.8-27B"

#: The factory a served modelled run names. `_build_oracle` imports this
#: qualname and offers it `rank_coords` because its signature asks for them.
ORACLE = "atom.compass.runtime.source_oracle.source_cost_oracle"

#: Selected on the server command line rather than through the oracle: the
#: aggregation is a property of how one process stands in for a group, not of
#: the price composition. `rank0` is the parser default and would price the
#: rank that calls itself, so acceptance names the other one. What the maximum
#: is and is not is in `atom/compass/runtime/predict.py`.
RANK_AGGREGATION = "slowest"

#: Every width shares these. `allocation=native` takes the CPU scheduler's own
#: block and state assignment for the step being priced; `carry_allocation=1`
#: reuses a capture's and is inadmissible for acceptance, so it is absent here
#: and `check` refuses a configuration that reintroduces it.
SHARED_OPTIONS = (
    ("model", MODEL),
    ("device", "meta"),
    ("block_size", "16"),
    ("max_model_len", "262144"),
    ("position_rows", "3"),
    ("cudagraph_mode", "full"),
    ("head", "1"),
    # `prefill-interp`, not `prefill-cells` and not the older `conc-v2`. Runs 5
    # and 6 -- the only two replays whose composition was actually exercised --
    # passed `source-27b-tp1-prefill-cells`, the source carrying the measured
    # 1-sequence prefill cells, and run 7 died on it: an exact-match lookup of
    # five cells cannot answer a final chunk, whose token count is
    # `prompt mod chunk_budget` and so arbitrary. `prefill-interp` carries
    # those five cells byte for byte and wins where keys collide; it adds
    # unpaced source anchors and interpolates only between adjacent
    # measurements of the same (sequences, produces_output) group, refusing
    # outside their span. `conc-v2` has never been selected by any run.
    ("regions", "source-27b-tp1-prefill-interp"),
    # Family-price mode, explicitly on. Without it `gap_ratio(None)` is None,
    # `_price_library` builds a plain `PriceLibrary` with no curves, and every
    # parametric family -- the head row ladder below above all -- is dead
    # weight: each refusal reads "no entry for this signature" with no
    # parametric reason attached. Run 5 omitted it and that is what it cost.
    ("interpolate", "1"),
    ("require_complete", "1"),
    ("allocation", "native"),
    ("derive", "1"),
    # The deployment the ragged attention laws are asked under. Without it
    # `_declared_scope` returns {}, `_request_scope` returns None, and every
    # ragged request is silent on the keys the laws are specific about: run 5
    # declared nothing and its 64 attention operators were all refused by name
    # while agreeing on backend, KV layout, geometry and treatment.
    #
    # `xacq`, not `b2acq`, which is what run 6 declared. The two SCOPE records
    # differ only in KV variants -- 16 against 32, and the blocks-per-variant
    # and pointers that follow -- and `kv_regions` is a treatment field, so
    # they are two deployments and the V=16 law is fitted for this one.
    ("attention_scope", "{root}/xacq/SCOPE.json"),
)

#: Options no acceptance cell may carry, with why. Checked rather than trusted:
#: a diagnostic option that survives into a matrix run is the failure mode this
#: registry exists to catch.
INADMISSIBLE = {
    "carry_allocation": (
        "reuses the template's block and state assignment and declares it "
        "unmeasured; acceptance needs the scheduler's own"
    ),
}

#: Where each width's artifacts live under the root. TP1 reads the source-width
#: files directly; TP2 and TP4 read a directory of links in
#: `resolve_rank_path`'s own convention, because the pricing artifacts are
#: named `<stem>.tp<width>.r<rank>.json` and that convention never produces it.
#: `SOURCE_ONLY_SERVING.md` has the loop that builds the links.
_TP1 = "{root}/g4/src1"
_WIDE = "{root}/serving/src_tp{tp}"

#: The corrected decode-32 capture, and the prices acquired against it.
#: `g4/src1` keeps every byte it had -- it is the evidence for what the old
#: binder refused and why -- and nothing here overwrites it: the seeds are
#: new files under new names, and the repriced signatures are their own list.
_SRC2C = "{root}/g4/src2c"
_SRC2P = "{root}/g4/src2p"

#: The rest of the source-width price tree. Two decode-32 pairs under `_TP1`
#: were this file's whole TP=1 list; the run that was actually exercised
#: staged the forty-odd files below and refused far less. They are named here
#: rather than left in `agent_scratch/stage/dev_serve5.sh` so that the
#: registry is the price list rather than a subset of one.
_CARD = "{root}/g4/card"
_PC = "{root}/pricing_coverage"
_HG = "{root}/headgrid"

#: The body row ladder acquired to close the 32..640 and 640..16384 gaps.
_LAD = "{root}/pricing_coverage/bodyladder"

#: The measurement repetition every price in the list below is read from. One
#: value for the whole list on purpose: mixing repetitions across a family is
#: how a curve acquires a step nobody measured.
_REP = "rep3"

#: Head-GEMM row ladder widths. `_HEAD_ROWS` is the original ladder; it stops
#: at 16 and resumes at 24 because those widths do not dispatch the same
#: kernel (MT64x16x256_MI16x16x1 against MT128x32x128_MI32x32x1), and the
#: library refused everything between rather than interpolate across the
#: switch. `_HEAD_ROWS_NATIVE` is the boundary scan that located it: the
#: switch is at 17, so [1,16] and [17,32] are two dispatch-supported regimes,
#: each readable inside.
#:
#: M=20 and M=27 are deliberately absent from both. 20 was a source holdout
#: and was refused, and 27 was the blind far-regime holdout predicted at
#: 1.134536e-03 and then measured at 1.144763e-03. Staging either here would
#: convert the only out-of-sample evidence the head curve has into a lookup.
_HEAD_ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
_HEAD_ROWS_NATIVE = (17, 18, 19, 21, 22)

#: The V=16 cached-unified campaign: the nine admitted designs and the graph
#: each was priced against, from `mixed_freeze/v2/FROZEN.json` sha256
#: a247addcdfa4cfd3a8efa08f2fecbe17039b9fc6036c6461525abcaf9d5a2e9e. Four
#: repeats per design, 36 records; `MANIFEST.json` lists 40 and excludes all
#: four P4 repeats, which are a cold diagnostic and not a cached training
#: point.
#:
#: K1's graph is `b1_C1`, not `b1_K1`. That is the freeze's own pairing and is
#: copied rather than regularised.
#:
#: The law these fit -- `unified.prefill.cached`, `paired_work`
#: 2.192033574230628e-10 and `history_rows` 1.1275685856067372e-07 per unit,
#: with `calls` and `query_rows` held at zero by active bounds -- carries known
#: small-design residuals: K1 -11.514%, K2 -31.884%, K3 +28.052%, against
#: within 1.6% for the six larger designs. It is a candidate with that
#: limitation recorded, not a clean fit, and its contribution is to be read off
#: the end-to-end gate rather than argued from the component.
_XACQ = "{root}/xacq/training"

#: The B2 GDN training pairs, from `mixed_freeze/v1/FROZEN.json` sha256
#: 6f7c0c8e71f0b616fd0f0d42b65235016317e581322dc84661fb23222813e3f1. These are
#: the fitted `gdn.prefill` family and nothing here supersedes them: v2 states
#: outright that "No GDN or cold-MHA candidate is modified", and its
#: `preserved_unchanged` block names this file read-only. Two body-prefill
#: files above do not substitute for them -- those carry the serving
#: deployment's own GDN observations, which is a different thing.
#:
#: G1's rep1 is absent by the freeze's own rule -- cold first use -- so the
#: repeat sets are not uniform. 39 pairs.
_B2ACQ = "{root}/b2acq/training"
_B2_GDN = (
    ("C1", (1, 2, 3, 4)),
    ("C2", (1, 2, 3, 4)),
    ("C3", (1, 2, 3, 4)),
    ("G1", (2, 3, 4)),
    ("G2", (1, 2, 3, 4)),
    ("G3", (1, 2, 3, 4)),
    ("G4", (1, 2, 3, 4)),
    ("G5", (1, 2, 3, 4)),
    ("G6", (1, 2, 3, 4)),
    ("G7", (1, 2, 3, 4)),
)

#: The cold-MHA companion of the V=16 campaign: `xacq` P4, four repeats,
#: `realized_cache` `["graph"]`, 0 histories against 16384 queries, graph
#: `b1_G4`. v2 excludes it from the cached fit and says why -- "P4 is the cold
#: diagnostic bridge; it is not a cached training point and never enters this
#: fit" -- which is exactly what makes it this deployment's cold observation.
#:
#: B2's own cold MHA (P1-P4 under `b2acq`) is V=32. `kv_regions` is a treatment
#: field and the declared deployment is the V=16 one, so B2's cold set is not
#: this deployment's cold measurement; loading it beside this one would put two
#: treatments in the cold domain, which is the ambiguity the composition
#: refuses. It is archived below rather than dropped silently.
_V16_COLD = (("P4", "{root}/b1graphs/b1_G4.reduced.json"),)

_V16_DESIGNS = (
    ("K1", "{root}/b1graphs/b1_C1.reduced.json"),
    ("K2", "{root}/b1graphs/b1_K2.reduced.json"),
    ("K3", "{root}/b1graphs/b1_K3.reduced.json"),
    ("K4", "{root}/b1graphs/b1_K4.reduced.json"),
    ("K5", "{root}/b1graphs/b1_K5.reduced.json"),
    ("X2", "{root}/xgraphs/x_X2.reduced.json"),
    ("X3", "{root}/xgraphs/x_X3.reduced.json"),
    ("X4", "{root}/xgraphs/x_X4.reduced.json"),
    ("X5", "{root}/xgraphs/x_X5.reduced.json"),
)
_V16_REPEATS = (1, 2, 3, 4)

#: The cached-MHA inputs this book used to carry, kept as a name rather than
#: as a load. Both are still on disk and both remain the evidence for what
#: they measured; neither is in `options` any more, because each is its own
#: `kv_regions` treatment and loading them beside V=16 is the ambiguity the
#: composition refuses. Listed so that "archived" is a fact in the registry
#: and not only in a handoff.
ARCHIVED_CACHED_MHA = (
    ("{root}/pricing_coverage/agg_att_q16384_c16384.tp1.json",
     "unified.prefill.cached, kv_regions=1"),
    ("{root}/pricing_coverage/p_caseb/AGG_caseb_unified_q16384_c32768.json",
     "unified.prefill.cached, kv_regions=32"),
    ("{root}/b2acq/training/PRICE_K{1..5}.rep{1..4}.json",
     "unified.prefill.cached, kv_regions=32 -- the superseded B2 population, "
     "20 files, still frozen in mixed_freeze/v1"),
    ("{root}/b2acq/training/PRICE_P{1..4}.rep{1..4}.json",
     "unified.prefill.COLD, kv_regions=32 -- 16 files; archived for the same "
     "treatment reason, not because the cold candidate changed"),
)


def _tp1_prices() -> tuple:
    """The source-width price list, as `prices:graph:regime` specs.

    Verbatim in content from the run whose composition was exercised; the
    paths are `{root}`-relative here and were absolute there.
    """
    spec = []

    def add(prices: str, graph: str = "") -> None:
        spec.append(f"{prices}:{graph}:unregistered")

    # The two decode-32 pairs, against the corrected src2c graphs.
    #
    # The src1 records are unchanged and still on disk: what changed is which
    # graph the step binds. `40ebae73` found that the CLI capture flags were
    # written to `provenance.execution` and not to the BatchSpec actually
    # traced, so the src1 body graph recorded a decode whose attention chain
    # was not the deployed one. CC retraced it as src2c and repriced the 16
    # MHA signatures that move with the corrected graph.
    #
    # So the body price list comes in two pieces, and the binding differs on
    # purpose:
    #
    #   * the src1 list *unbound*. Its 2423 other signatures did not change,
    #     and they match by signature. Binding it to the src2c graph would
    #     claim it was measured against a graph it never saw.
    #   * the 16 repriced signatures bound to the src2c body graph, which is
    #     the graph they were measured against and the one the step binds.
    #
    # No key is substituted inside any old record. Where both pieces offer a
    # signature the bound one is the specific match.
    add(f"{_TP1}/p27bdec32.tp1.r0.json")
    add(f"{_SRC2P}/prices/p27bdec32attn.tp1.r0.{_REP}.json",
        f"{_SRC2C}/b27dec32.tp1.r0.json")
    add(f"{_TP1}/p27hdec32.tp1.r0.json", f"{_SRC2C}/h27dec32.tp1.r0.json")

    # The 640- and 16384-token prefill cells.
    p640 = f"{_PC}/p640"
    add(f"{p640}/p27_body_640.tp1.r0.{_REP}.json",
        f"{p640}/graphs/b27_tp1_r0_pref_body_640.json")
    add(f"{p640}/p27_head_640.tp1.r0.{_REP}.json",
        f"{p640}/graphs/b27_tp1_r0_pref_head_640.json")
    # No graph: a narrowed `--only` job prices the LM-head GEMM alone.
    add(f"{p640}/p27_headgemm3_640.tp1.r0.{_REP}.json")
    add(f"{_PC}/p16384/p27_body_16384.tp1.r0.{_REP}.json",
        f"{_PC}/p16384/graphs/b27_tp1_r0_pref_body_16384.json")

    # The rest of the body row ladder.
    #
    # `RowSupport` refuses a lookup whose row count falls in a gap between
    # adjacent measured rows wider than `max_gap_ratio`, and the body rows
    # this list carried were the decode ladder 1,2,4,8,16,32 plus the two
    # prefill cells above. Run 8 asked for 9216 rows and was refused -- "falls
    # in the unsampled gap 640..16384 (x25.6)" -- for 42 signatures, which
    # left 2118 of 2443 operators unpriced. A short final chunk would have hit
    # 32..640 (x20) the same way. Fixing `interpolate=1` restored the declared
    # bound to its 2.0 default and closed neither gap: that takes measurement.
    #
    # 512 and 1024 were acquired alongside 640 as its interpolation brackets
    # and have been on disk since; they were simply never named here. The rest
    # were measured by `agent_scratch/stage/job_bodyladder.sh`, on the
    # snapshot that priced 16384.
    #
    # Adjacent ratios across the axis are now 32:64, 64:128, 128:256, 256:512,
    # 1024:2048, 2048:4096, 4096:8192 and 8192:16384 all exactly 2.0, plus
    # 512:640 at 1.25 and 640:1024 at 1.6. The rule refuses `ratio >
    # max_gap_ratio`, so exactly 2.0 is inside support; every rung is a power
    # of two, which is the boundary the kernels dispatch on, and 640 is a
    # measured deployment shape rather than a ladder rung.
    for n in (512, 1024):
        add(f"{p640}/p27_body_{n}.tp1.r0.{_REP}.json",
            f"{p640}/graphs/b27_tp1_r0_pref_body_{n}.json")
    for n in (64, 128, 256, 2048, 4096, 8192):
        add(f"{_LAD}/p27_body_{n}.tp1.r0.{_REP}.json",
            f"{_LAD}/graphs/b27_tp1_r0_pref_body_{n}.json")

    # The long-context cells and the mixed step.
    for stem in ("ctx_b32_c4096", "ctx_b32_c16384", "mix_b32"):
        add(f"{_PC}/long/p27_{stem}.tp1.r0.{_REP}.json",
            f"{_CARD}/b27_tp1_r0_{stem}.json")

    # The one-sequence body ladder, and the cell at its context boundary.
    add(f"{_PC}/g1b/p27_ctx_b32_c1151.tp1.r0.{_REP}.json",
        f"{_CARD}/b27_tp1_r0_ctx_b32_c1151.json")
    for batch in (1, 2, 4, 8, 16, 32):
        add(f"{_PC}/g1b/p27_lad_b{batch}.tp1.r0.{_REP}.json",
            f"{_CARD}/b27_tp1_r0_lad_b{batch}.json")

    # The cached-MHA calibration inputs: the V=16 campaign, and nothing else.
    #
    # What used to stand here was two aggregated attention files, and asking
    # the library rather than their names showed they were two *different*
    # cached populations -- `agg_att_q16384_c16384` at `kv_regions=1` and
    # `AGG_caseb_unified_q16384_c32768` at `kv_regions=32`, 16 observations
    # each. `kv_regions` is a treatment field, so those are two laws by
    # identity; with the B2 V=32 set that is three, and a deployment that
    # declares none of them is refused for ambiguity rather than priced.
    #
    # Replacing both with the V=16 set leaves exactly one cached treatment in
    # the book, so `_treatment_for` resolves without a declaration. The cold
    # MHA observations are NOT touched: they come from the two body-prefill
    # files above at `('cache', 'graph')`, a different regime, and dropping
    # the cached files leaves them exactly as they were.
    for design, graph in _V16_DESIGNS:
        for rep in _V16_REPEATS:
            add(f"{_XACQ}/PRICE_{design}.rep{rep}.json", graph)

    # The cold-MHA observation of the same deployment, and the fitted B2 GDN
    # family. Neither is supplied by the body-prefill files above: those carry
    # the serving deployment's own attention observations, under the serving
    # scope, and the two ragged laws this book has to answer from were fitted
    # from these.
    for design, graph in _V16_COLD:
        for rep in _V16_REPEATS:
            add(f"{_XACQ}/PRICE_{design}.rep{rep}.json", graph)
    for design, repeats in _B2_GDN:
        for rep in repeats:
            add(f"{_B2ACQ}/PRICE_{design}.rep{rep}.json",
                f"{{root}}/b1graphs/b1_{design}.reduced.json")

    # The head row ladder, both passes per width: the whole-graph job prices
    # the three metadata operators and a narrowed `--only` job takes the LM
    # head GEMM. Without these the head GEMM at any width below 32 refuses as
    # an extrapolation from the single measured width.
    for rows, sub in ((_HEAD_ROWS, ""), (_HEAD_ROWS_NATIVE, "_native")):
        for m in rows:
            for pass_ in ("all", "gemm"):
                add(f"{_HG}/prices{sub}/head_h16384_m{m}.{pass_}.{_REP}.json",
                    f"{_HG}/graphs{sub}/head_h16384_m{m}.json")
    return tuple(spec)


#: The record a modelled server builds its `Config` from, per width.
#:
#: `ReplayModelRunner._check_parallel_contract` refuses a target whose own
#: `tensor_parallel_size` is not the width being replayed, and it refuses
#: rather than warns: the block count and pool entries were sized at the
#: captured width, and handing them to a wider scheduler lets it admit a
#: workload the target cannot hold. So there is no one target for the matrix.
#: TP=1 replays the captured source-width record, which is of the width it
#: claims and is what the source oracle was built from. TP=2 and TP=4 replay
#: a record derived at that width from the memory model. The pool every one
#: of them is sized from is the analytical profile below, not the record.
_CAPTURED_TARGET = "{root}/poc/g5_27b/target.json"
_DERIVED_TARGET = "{root}/serving/src_tp{tp}/target.tp{tp}.json"

#: The memory profile every width sizes its pool from, including TP=1. The
#: acceptance question is whether every non-KV term and the KV budget survive
#: the move from measured to analytical, so a replay sized from a captured
#: count is not evidence for it at any width -- TP=1 may *bootstrap* from the
#: captured target (it is a record of its own width, and the source oracle is
#: built from that capture), but the budget it actually replays under has to
#: be the derived one. `memory_blocks` then reports `source-derived` for all
#: three cells.
#:
#: These are MEMORY's own files at the path they already publish them to, not
#: a delivery convention invented here: a profile is per *width*, never rank
#: resolved.
#:
#: `profile_r21` rather than the older `profile/`: the r21 set carries
#: `capture_history`, so a profile records what was captured beside the terms
#: it derives, and its calibration files are the ones MEMORY's 8ef965dc gate
#: reads. The older directory is left in place -- past diagnostics cite it by
#: path and must keep resolving -- but nothing in the frozen plan selects it.
_PROFILE = "{root}/memval/capture_replay/profile_r21/profile.tp{tp}.json"


def replay_target(tp: int, root) -> str:
    """The target this width replays against, resolved against `root`."""
    template = _CAPTURED_TARGET if tp == 1 else _DERIVED_TARGET
    return template.format(root=str(Path(root)), tp=tp)


def memory_model(tp: int, root) -> str:
    """The profile this width sizes its pool from.

    Every width, TP=1 included. Without it the replay takes the block and
    state counts verbatim out of the target record and publishes the budget as
    `captured`, which is the measured number the acceptance is supposed to be
    testing the analytical one against.
    """
    return _PROFILE.format(root=str(Path(root)), tp=tp)


def per_width_options(tp: int) -> tuple:
    """The options this width adds to `SHARED_OPTIONS`, unresolved."""
    if tp == 1:
        return (
            ("tp", "1"),
            ("replay_target", _CAPTURED_TARGET),
            ("price", ",".join(_tp1_prices())),
            # src2c, not src1: the templates are what the step binds, and the
            # src1 capture recorded a decode whose attention chain was not
            # the deployed one. See `_tp1_prices` for what that changed and
            # what it deliberately did not.
            ("template", f"{_SRC2C}/b27dec32.tp1.r0.json"),
            ("head_template", f"{_SRC2C}/h27dec32.tp1.r0.json"),
        )
    body = f"{_WIDE}/p27bdec32.json:{_WIDE}/b27dec32.json:unregistered"
    head = f"{_WIDE}/p27hdec32.json:{_WIDE}/h27dec32.json:unregistered"
    # Both all-reduce lists are loaded on purpose: they hold the same signature
    # under different registration regimes and the graph selects between them,
    # so which one answers must not depend on load order.
    prices = (body, head, f"{_WIDE}/ar_capture.json",
              f"{_WIDE}/ar_plain.json", f"{_WIDE}/ag_prices.json")
    return (
        ("tp", str(tp)),
        ("replay_target", _DERIVED_TARGET),
        ("price", ",".join(prices)),
        ("template", f"{_WIDE}/b27dec32.json"),
        ("head_template", f"{_WIDE}/h27dec32.json"),
    )


def options(tp: int, root) -> list:
    """The `KEY=VALUE` strings for this width, resolved against `root`."""
    root = str(Path(root))
    return [f"{key}={value.format(root=root, tp=tp)}"
            for key, value in SHARED_OPTIONS + per_width_options(tp)]


def option_paths(tp: int, root) -> dict:
    """Every file this width's options name, by the role it plays.

    Read back out of `options` rather than listed again, so the two cannot
    drift: a path that stops being an option stops being required in the same
    edit. These are the paths as *written*; what a rank actually opens is
    `resolution`, below.
    """
    found = {}
    for item in options(tp, root):
        key, _, value = item.partition("=")
        if key == "price":
            for n, spec in enumerate(value.split(",")):
                parts = spec.split(":")
                found[f"price[{n}].prices"] = parts[0]
                if len(parts) > 1 and parts[1]:
                    found[f"price[{n}].graph"] = parts[1]
        elif key in ("template", "head_template", "replay_target"):
            found[key] = value
    # Not an oracle option: the profile is a server flag, read by the replay
    # runner and not by the price composition. It is required all the same --
    # it is what the pool is sized from at every width -- so it is reported
    # here with the files that are.
    found["memory_model"] = memory_model(tp, root)
    return found


#: Kept under the old name: `required_artifacts` is what the plan's tests and
#: the first readers of this module called it.
required_artifacts = option_paths


#: Roles whose single file answers for the whole group rather than for a rank.
#: A target and a profile are per *width* and build the same Config on every
#: rank of it, and a collective's price list is a measurement of the group,
#: not of a member: nothing writes `ar_capture.tp2.json`, so asking for one
#: and then reporting the unsuffixed file as a fallback would file a claim
#: against a file that is correct.
_GROUP_STEMS = ("ar_capture.json", "ar_plain.json", "ag_prices.json")


def _group_level(role: str, path: str) -> bool:
    return role in ("replay_target", "memory_model") or path.endswith(_GROUP_STEMS)


def resolution(tp: int, root) -> dict:
    """What each rank of this width actually opens, and whether it is its own.

    An option names `b27dec32.json`; the rank appends its own coordinates and
    reads `b27dec32.tp2.json`, falling back to the unsuffixed file when it has
    none of its own. The fallback is legitimate -- a symmetric group's ranks
    time within a fraction of a percent -- but it is a claim, and the whole
    reason to report it here is that a run in which every rank silently read
    rank 0's artifacts looks identical to one in which each read its own.

    Group-level roles and TP=1 are excluded from that: neither can confuse one
    rank's measurement for another's, so reporting them would bury the case
    that can under dozens of lines that cannot.
    """
    out = {}
    for rank in range(tp):
        per = {}
        for role, path in option_paths(tp, root).items():
            if _group_level(role, path):
                per[role] = {"path": path, "own": True,
                             "exists": Path(path).exists()}
                continue
            if tp == 1:
                # The source-width files carry `.tp1.r0` in the written name,
                # and a group of one has no other rank to be confused with.
                per[role] = {"path": path, "own": True,
                             "exists": Path(path).exists()}
                continue
            resolved, own = resolve_rank_path(path, {"tp": rank})
            per[role] = {"path": resolved, "own": own,
                         "exists": Path(resolved).exists()}
        out[rank] = per
    return out


#: What a cc-traces run would still refuse with these artifacts in place, from
#: `SOURCE_ONLY_SERVING.md`'s own list. Every artifact resolving is necessary
#: and not sufficient: a cell with nothing absent is *configured*, not ready,
#: and these are the reasons. `closes` names what would retire each one, so a
#: cell's readiness is a question with an answer rather than a run that fails
#: at the first decode step with a running-request count of 31.
OPEN_REFUSALS = {
    "head_rows": {
        # Closed at TP=1 and only there. The row ladder in `_tp1_prices`
        # measures the source width at 1..32 across both dispatch regimes,
        # which is every running-request count `max_num_seqs=32` can reach.
        # TP=2 and TP=4 still stage the single decode-32 head price out of
        # `_WIDE`, so the original refusal stands at those widths unchanged.
        "what": "at TP>1 the head is priced at 32 rows and nowhere else; any "
                "other running-request count is refused, and one measured "
                "point fits nothing",
        "cells": tuple(c for c in CELLS if not c.startswith("tp1-")),
        "closes": "the source-width row ladder, re-measured or transferred at "
                  "each width, over the counts the workloads actually reach",
        "first_met": True,
    },
    "seeded_structure": {
        "what": "only the decode-32 structure is seeded; every prefill, "
                "chunked step and other bucket needs derivation",
        "cells": CELLS,
        "closes": "derivation of the remaining structures, ~10.5 s each, "
                  "device-free",
        "first_met": False,
    },
    "derivation_is_rank0_s": {
        "what": "derivation produces rank 0's shard whatever rank asks, so "
                "at TP>1 the other ranks are served the representative",
        "cells": tuple(c for c in CELLS if not c.startswith("tp1-")),
        "closes": "per-rank derivation, or a measured bound on the "
                  "rank-to-rank spread at each width",
        "first_met": False,
    },
    "region_model_held_out": {
        "what": "the region model is held out at TP2 and TP4: it was fitted "
                "at the source width and its transfer is what G4 tests",
        "cells": tuple(c for c in CELLS if not c.startswith("tp1-")),
        "closes": "the G4 held-out transfer result, or a width-local fit "
                  "declared as such",
        "first_met": False,
    },
    # The padded tail below the capture bucket used to be listed here as a
    # third permanent refusal. It is not one any more: `derive and bind an
    # ordinary padded decode` (70306a99) traces the padded body and keeps the
    # binding, and `pack M-RoPE decode positions at the width the graph reads`
    # (6d01323b) lays the pad inside each M-RoPE section rather than once at
    # the end, so a three-request step replaying a bucket of four is priced
    # rather than named. What remains below are the two that are refusals on
    # purpose.
    "unallocatable_steps": {
        "what": "a step nobody offers an allocation for is refused by name: "
                "a mixed prefill/decode batch, and a row with no state slot",
        "cells": CELLS,
        "closes": "nothing, for cc-traces: with TBO off the scheduler emits "
                  "no mixed batch, and a row with no state slot is a refusal "
                  "on purpose",
        "first_met": False,
    },
}


COST_TERMS = {
    "capture": {
        "what": "the TP=1 tracing pass that produced the graphs",
        "state": "owed",
        "from": "the source-width capture run's own log",
        "in_gate": False,
    },
    "calibration": {
        "what": "the source measurement pass (sweep / primitive pricing)",
        "state": "owed",
        "from": "the pricing acquisition's log, summed over its passes",
        "in_gate": False,
    },
    "derivation": {
        "what": "CPU derivation of this width's graphs from the TP=1 capture",
        "state": "owed",
        "from": "the device-free derivation run, per width",
        # In the denominator: deriving this candidate is work that asking the
        # question costs, unlike capture and calibration which are paid once.
        "in_gate": True,
    },
    "load": {
        "what": "weight load and graph capture inside startup",
        "state": "owed",
        "from": "the real side's startup log",
        "in_gate": False,
    },
    "startup_real": {
        "what": "process start to healthy, real side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": False,
    },
    "startup_modelled": {
        "what": "process start to healthy, modelled side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": False,
    },
    "execution_real": {
        "what": "the measured window itself, real side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": True,
    },
    "execution_modelled": {
        "what": "the measured window itself, modelled side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": True,
    },
}

OWED_TERMS = tuple(k for k, v in COST_TERMS.items() if v["state"] == "owed")


def verify(tp: int, root) -> dict:
    """Build this width's oracle from its own options and report what loaded.

    `check` resolves paths; this resolves *configuration*. The two answer
    different questions and the difference is not academic: a file that exists
    and a file the composition read are the same thing only when nothing is
    quietly declining it. Run 5's whole family-price path was inert with every
    artifact present, because one option was absent -- `check` would have
    called that configuration complete.

    So this calls `build_source_group` with the registry's own options, builds
    the price library's identities, and reports the object that resulted: the
    provider class actually selected, the interpolation gap it will honour,
    which price files contributed a family and which are exact-signature only,
    and the region model taken. No shape is priced and nothing is derived.

    `build_source_group` rather than `source_cost_oracle`: the served entry
    point discards everything but the oracle, and the composition is where the
    gap ratio and the rank artifacts are. One width per process -- ATOM
    registers attention layers in a global table at construction and refuses a
    second build -- so `main` forks rather than looping.
    """
    from atom.compass.runtime.source_oracle import build_source_group

    kwargs = {}
    for item in options(tp, root):
        key, _, value = item.partition("=")
        kwargs[key] = value
    built = build_source_group(**kwargs)

    library = getattr(built.oracle, "library", None)
    # The library takes its identities lazily, so a report written before this
    # would show an empty read list for a composition that reads forty files.
    build = getattr(library, "_build", None)
    if callable(build):
        build()
    unbuildable = dict(getattr(library, "unbuildable", {}) or {})
    requested = [spec.split(":")[0]
                 for item in options(tp, root) if item.startswith("price=")
                 for spec in item.partition("=")[2].split(",")]
    return {
        "tp": tp,
        "root": str(Path(root)),
        "provider": type(library).__name__ if library is not None else None,
        # Off the library, not off the group: `build_source_group` returns
        # `SourceGroup`, which has no such field, and reading it there would
        # report `null` for a composition that is interpolating.
        "max_gap_ratio": getattr(library, "max_gap_ratio", None),
        "interpolating": getattr(library, "max_gap_ratio", None) is not None,
        "regions": next((o.partition("=")[2] for o in options(tp, root)
                         if o.startswith("regions=")), None),
        "region_snapshot": bool(
            getattr(built.oracle, "compass_region_snapshot", None)),
        "price_files_requested": len(requested),
        "price_files_exact_only": {Path(p).name: why
                                   for p, why in sorted(unbuildable.items())},
        "loaded_inputs": len(getattr(built, "loaded_inputs", ()) or ()),
        "build_seconds": getattr(built, "build_seconds", None),
        "means": (
            "the composition this width's options actually build; a price "
            "file listed under price_files_exact_only was read but "
            "contributes no family, and provider names the path that will "
            "answer a lookup. At TP>1 the group holds one library per rank "
            "and this describes the one the group oracle exposes, so it "
            "speaks for that rank's load and not for the spread across them"
        ),
    }


def cell_config(tp: int, klass: str, root) -> dict:
    """One cell, whole: what it is served with and what it still owes."""
    return {
        "cell": f"tp{tp}-{klass}",
        "tp": tp,
        "class": klass,
        "oracle": ORACLE,
        "oracle_options": options(tp, root),
        "rank_aggregation": RANK_AGGREGATION,
        "allocation": "native",
        "artifacts": option_paths(tp, root),
        "resolution": resolution(tp, root),
        "owed_costs": list(OWED_TERMS),
        "open_refusals": sorted(k for k, v in OPEN_REFUSALS.items()
                                if f"tp{tp}-{klass}" in v["cells"]),
    }


def check(root) -> dict:
    """Resolve every cell and say what is missing, without running anything."""
    cells = []
    for tp in TPS:
        for klass in CLASSES:
            config = cell_config(tp, klass, root)
            absent, shared = set(), set()
            for rank, roles in config["resolution"].items():
                for role, found in roles.items():
                    if not found["exists"]:
                        absent.add(f"{role}@tp{rank}")
                    elif not found["own"]:
                        # Exists, and is not this rank's: one rank's file
                        # answering for another is admissible and is a claim,
                        # so it is named rather than counted as present.
                        shared.add(f"{role}@tp{rank}")
            bad = sorted(key for key in INADMISSIBLE
                         if any(o.startswith(f"{key}=")
                                for o in config["oracle_options"]))
            config["absent_artifacts"] = sorted(absent)
            config["shared_artifacts"] = sorted(shared)
            config["inadmissible_options"] = bad
            config["runnable"] = not absent and not bad
            # Configured is not ready. A cell with every artifact resolved
            # still refuses the steps below, and the difference between the
            # two is the whole content of this report.
            config["ready"] = config["runnable"] and not config["open_refusals"]
            cells.append(config)
    return {
        "root": str(Path(root)),
        "cells": cells,
        "owed_costs": {k: COST_TERMS[k] for k in OWED_TERMS},
        "means": (
            "the configuration each cell would be served with, what is "
            "absent, and which ranks would read another rank's file; nothing "
            "was run and no cell is claimed ready"
        ),
    }


def _resolved(cell, key: str) -> str:
    role, _, rank = key.partition("@tp")
    return cell["resolution"][int(rank)][role]["path"]


def render(report: dict) -> str:
    out = [f"# six cc-traces cells under {report['root']}", ""]
    for cell in report["cells"]:
        state = ("ready" if cell["ready"] else
                 "configured, NOT ready" if cell["runnable"] else
                 "NOT runnable")
        out.append(f"## {cell['cell']} -- {state}")
        out.append(f"  oracle            {cell['oracle']}")
        out.append(f"  rank_aggregation  {cell['rank_aggregation']}")
        out.append(f"  allocation        {cell['allocation']}")
        for key in cell["absent_artifacts"]:
            out.append(f"  ABSENT  {key}: {_resolved(cell, key)}")
        for key in cell["shared_artifacts"]:
            out.append(f"  NOT THIS RANK'S  {key}: {_resolved(cell, key)}")
        for key in cell["inadmissible_options"]:
            out.append(f"  INADMISSIBLE  {key}: {INADMISSIBLE[key]}")
        for key in cell["open_refusals"]:
            term = OPEN_REFUSALS[key]
            first = " (met first)" if term["first_met"] else ""
            out.append(f"  REFUSES{first}  {key}: {term['what']}")
            out.append(f"    closed by: {term['closes']}")
        out.append("")
    out.append("## costs no step of this harness measures")
    for name, term in report["owed_costs"].items():
        where = ("in the gate denominator" if term["in_gate"]
                 else "reported beside it")
        out.append(f"  {name}: {term['what']} -- from {term['from']}, {where}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="directory the artifact paths resolve against")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="also build each width's oracle and report the "
                         "configuration that actually loaded, not only the "
                         "files that resolve")
    ap.add_argument("--verify-tp", type=int, choices=TPS,
                    help="verify one width in this process; what --verify "
                         "forks per width")
    args = ap.parse_args(argv)

    if args.verify_tp:
        try:
            report = verify(args.verify_tp, args.root)
        except Exception as error:  # noqa: BLE001 - a refusal is the result
            report = {"tp": args.verify_tp,
                      "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(report, indent=2))
        return 0 if "error" not in report else 1

    if args.verify:
        # A child per width, because a second build in one process is refused:
        # ATOM registers attention layers in a global table at construction.
        # Only the child's last JSON object is read -- the model build writes
        # to stdout too, and a report that swallowed that would be unreadable.
        built = []
        for tp in TPS:
            done = subprocess.run(
                [sys.executable, __file__, "--root", str(args.root),
                 "--verify-tp", str(tp)],
                capture_output=True, text=True)
            tail = done.stdout[done.stdout.rfind("\n{"):] or done.stdout
            try:
                built.append(json.loads(tail))
            except ValueError:
                built.append({"tp": tp, "error": "child produced no report",
                              "stderr": done.stderr[-400:]})
        print(json.dumps(built, indent=2))
        return 0 if all("error" not in b for b in built) else 1

    report = check(args.root)
    print(json.dumps(report, indent=2) if args.json else render(report))
    # Non-zero while anything is absent, so this can gate a lease rather than
    # only describe one.
    return 0 if all(c["runnable"] for c in report["cells"]) else 1


if __name__ == "__main__":
    sys.exit(main())
