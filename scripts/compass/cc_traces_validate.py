"""Decide whether a cc-traces acceptance cell counts, and what the matrix says.

`compare.py` already refuses a run that did not complete as asked, and this does
not restate it -- it loads that checker and runs it first, so the two cannot
drift. What it adds is everything that is specific to *this* acceptance, and
each addition exists because a run could pass every check that came before it
and still not be the experiment `CC_TRACES_PROTOCOL.md` registers:

  * a run that sent 20 of the registered 64 requests, or sent them rescaled, is
    a valid run of a workload nobody registered;
  * a clock can be consistent and still be infinite -- `inf - inf` is `nan`, and
    a `nan` comparison is false, so a derived-versus-computed check passes;
  * a modelled side that read a calibration measured at the width it is
    predicting has predicted nothing, and nothing in the artifact says so
    unless something reads the digests against a declaration of where they
    came from.

It runs on CPU, reads files only, and fails closed: an artifact it cannot
attribute is a refusal, not a warning.

    python scripts/compass/cc_traces_validate.py cell <dir> --class long \\
        --calibration-registry registry.json
    python scripts/compass/cc_traces_validate.py matrix <dir> <dir> ... --out verdict.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """A sibling script as a module, the way the compass tests load them."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("compare")

# --------------------------------------------------------------------------
# what the protocol registers

WORKLOADS = {
    "long": ROOT / "atom" / "compass" / "cc_traces_long.jsonl",
    "short": ROOT / "atom" / "compass" / "cc_traces_short.jsonl",
}

#: The engine every cell is served by. Read back from the server's own
#: provenance rather than from the command line that was meant to produce it.
ENGINE = {
    "max_model_len": 262144,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 32,
    "enable_prefix_caching": False,
}

#: The width the cost model is calibrated from. Every other width is a
#: prediction, which is the whole bet -- see the protocol's section 2.
SOURCE_TP = 1

#: Kinds of calibration artifact, and the width each may have been measured at.
#: `standalone_primitive` is the exception and the only one: a collective's
#: price is a property of the rank count and a single rank cannot produce it,
#: so it is measured on its own, outside the target engine, at that width.
CALIBRATION_KINDS = {
    "source_calibration": SOURCE_TP,
    "region_model": SOURCE_TP,
    "overhead_constant": SOURCE_TP,
    "derived_graph": SOURCE_TP,
    "standalone_primitive": None,
}

COST_TERMS = (
    "capture",
    "calibration",
    "derivation",
    "startup_real",
    "startup_modelled",
    "load",
    "execution_real",
    "execution_modelled",
)

#: The terms `cc_traces_run.py` measures from the cell's own repeats: plain
#: seconds. The rest are supplied at merge time and carry their origin and
#: their container, because `load` happens inside `startup_real` and adding it
#: beside that startup charges the same work twice.
MEASURED_COST_TERMS = (
    "startup_real",
    "startup_modelled",
    "execution_real",
    "execution_modelled",
)
SUPPLIED_COST_TERMS = ("capture", "calibration", "derivation", "load")

#: The record shape whose supplied terms were bare numbers. Refused rather
#: than upgraded: it never stated containment, so there is nothing to read.
COSTS_SCHEMA_V2 = "compass.costs/2"
#: The record shape whose supplied terms carried no repeat. Its execution
#: terms are medians over the cell's repeats and its journalled terms are sums
#: over all of them, so a ratio of the two is wrong by the repeat count and
#: nothing in the record says by how much. Refused rather than upgraded.
COSTS_SCHEMA_V3 = "compass.costs/3"
#: What `cc_traces_run.py costs` writes now.
COSTS_SCHEMA = "compass.costs/4"

#: What a supplied part writes in its `repeat` field when the cost is spent
#: once in every repeat rather than in one of them. It is a claim the record
#: makes, not an allocation a reader performs over a total.
EVERY_REPEAT = "each"

TOLERANCE_PCT = {"throughput_tok_s": 10.0, "tpot": 10.0, "ttft": 15.0}
RHO_MIN = 0.90
SPEEDUP_MIN = 5.0

#: The only clock a runtime cost may be measured on, and the value
#: `cc_traces_run.py` writes into a cost record's `execution_clocks`. A
#: predicting engine serves on a virtual clock, so its reported window is a
#: prediction about duration rather than time anything spent.
WALL_CLOCK = "wall"

#: How far a paced arrival may land from where the workload declared it. The
#: real side sleeps until each moment and the engine stamps on receipt, so some
#: drift is the client's scheduler and the socket; a large one means the
#: arrival process that ran was not the registered one.
ARRIVAL_DRIFT_S = 2.0


def registered_rows(klass: str) -> list[dict]:
    rows = []
    for line in WORKLOADS[klass].read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# --------------------------------------------------------------------------
# per-run checks


def check_clocks_finite(run) -> list[str]:
    """Every stamp is a real number.

    `compare.check_run` orders the three stamps and cross-checks the derived
    `ttft`, and both survive infinities: `inf <= inf <= inf` holds, and
    `abs(inf - inf) > 1e-3` is false because the difference is `nan`. So a run
    whose clock ran away would report finite-looking percentage errors computed
    from infinite windows.
    """
    bad = []
    for i, record in sorted(run.joined.items()):
        for field in (
            "arrive_time",
            "first_token_time",
            "finish_time",
            "ttft",
            "latency",
        ):
            value = record.get(field)
            if value is None:
                continue  # absence is compare.py's to report
            if not _finite(value):
                bad.append(
                    f"request {i}: {field} is {value!r}, not a finite " f"number"
                )
            elif field == "arrive_time" and value < 0:
                bad.append(f"request {i}: arrive_time {value} is negative")
    return bad


def check_workload_is_registered(run, rows, label: str) -> list[str]:
    """The requests that ran are the registered ones, in order, unrescaled."""
    bad = []
    scale = run.manifest.get("time_scale")
    if scale is not None and abs(float(scale) - 1.0) > 1e-9:
        bad.append(
            f"{label}: time_scale {scale} -- the arrival process was "
            f"compressed, so this is not the registered workload"
        )
    if len(run.workload) != len(rows):
        bad.append(
            f"{label}: {len(run.workload)} requests were sent against "
            f"the registered {len(rows)}"
        )
        return bad
    for i, (sent, want) in enumerate(zip(run.workload, rows)):
        for field in ("input_tokens", "output_tokens"):
            if int(sent.get(field, -1)) != int(want[field]):
                bad.append(
                    f"{label}: request {i} {field} {sent.get(field)} "
                    f"against the registered {want[field]}"
                )
        if abs(float(sent.get("arrival_s", -1)) - float(want["arrival_s"])) > 1e-6:
            bad.append(
                f"{label}: request {i} declares arrival "
                f"{sent.get('arrival_s')} against the registered "
                f"{want['arrival_s']}"
            )
        if len(bad) > 8:
            bad.append(f"{label}: ... further differences not listed")
            break
    return bad


def check_usage_against_workload(run, rows, label: str) -> list[str]:
    """The server's own `usage`, against what the workload asked for.

    `--check-lengths` is the client's claim about the same thing; this reads the
    server's counts out of the saved responses, so a run that reported
    `prompt_lengths: passed` and served something else is still caught.
    """
    bad = []
    for i, want in enumerate(rows):
        usage = run.usage.get(i)
        if usage is None:
            continue  # no response at all is compare.py's to report
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is not None and int(prompt) != int(want["input_tokens"]):
            bad.append(
                f"{label}: request {i} was served {prompt} prompt "
                f"tokens against the registered {want['input_tokens']}"
            )
        if completion is None:
            continue
        if int(completion) > int(want["output_tokens"]):
            bad.append(
                f"{label}: request {i} produced {completion} tokens, "
                f"more than the {want['output_tokens']} it asked for"
            )
        elif int(completion) < int(want["output_tokens"]):
            # Not a failure on its own -- a stop token is legitimate -- but it
            # must be visible, because throughput and TPOT are computed from
            # what was produced and the workload says something else.
            bad.append(
                f"NOTE {label}: request {i} produced {completion} of "
                f"{want['output_tokens']} requested output tokens"
            )
        if len(bad) > 8:
            bad.append(f"{label}: ... further differences not listed")
            break
    return bad


def check_arrival_process(
    run, rows, label: str, paced: bool, drift_s: float = ARRIVAL_DRIFT_S
) -> list[str]:
    """What arrived, when, against what the workload declared.

    Two different questions on the two sides. The modelled side is *declared*:
    the engine honours `compass_arrival`, so its stamps must reproduce the
    workload exactly. The real side is *paced*: the client sleeps until each
    moment, so its stamps drift, and what matters is that the drift is small
    enough that the arrival process which ran is the one registered.
    """
    bad = []
    if not run.joined:
        return bad
    stamps = {i: run.joined[i].get("arrive_time") for i in run.joined}
    if any(not _finite(v) for v in stamps.values()):
        return bad  # already reported by check_clocks_finite
    origin = min(stamps.values())
    worst, worst_at = 0.0, None
    for i, stamp in stamps.items():
        if i >= len(rows):
            continue
        drift = abs((stamp - origin) - float(rows[i]["arrival_s"]))
        if drift > worst:
            worst, worst_at = drift, i
    limit = drift_s if paced else 1e-3
    if worst > limit:
        bad.append(
            f"{label}: request {worst_at} arrived {worst:.3f}s from "
            f"where the workload declares it (limit {limit}s"
            + (", paced" if paced else ", declared")
            + "): the arrival "
            "process that ran is not the registered one"
        )
    order = [stamps[i] for i in sorted(stamps)]
    inversions = sum(1 for a, b in itertools.pairwise(order) if b < a - limit)
    if inversions:
        bad.append(
            f"{label}: {inversions} requests arrived out of the "
            f"registered order by more than {limit}s"
        )
    return bad


def check_side_roles(real, modelled) -> list[str]:
    """Each side did its own job: one measured warm, one predicted cold."""
    bad = []
    rm, mm = real.manifest, modelled.manifest
    if not rm.get("paced"):
        bad.append(
            "the real side was not paced: its arrivals were declared "
            "to a server on a wall clock, which ignores them"
        )
    if mm.get("paced"):
        bad.append(
            "the modelled side was paced: a predictor advances virtual "
            "time, so pacing it against the wall clock races the two"
        )
    if not (rm.get("prepare") or {}).get("drained"):
        bad.append(
            "the real side has no drained preparation: PROTOCOL.md "
            "section 4 makes the drain the boundary of the measurement"
        )
    if mm.get("prepare"):
        bad.append(
            "the modelled side was prepared: a declared arrival is an "
            "offset from the engine's epoch, so preparation lands "
            "inside every measured request's TTFT"
        )
    # The modelled side has to be *known* not to have timed out. `compare.py`
    # refuses a True reading, which leaves unknown passing -- and unknown is
    # what a run gets when the engine could not be asked, or when the artifact
    # predates the field. Reported as a pass, that is the same claim as "the
    # barrier held", from evidence that says nothing. For a predictor it is the
    # whole question: every latency it produces depends on virtual time never
    # having advanced past an arrival still in flight, and only the engine can
    # say. So acceptance requires the engine's own False here. A real server
    # runs on a wall clock and has no barrier to wait on, so an unread barrier
    # there stays not applicable.
    if mm.get("arrival_barrier_timed_out") is None:
        why = (mm.get("arrival_barrier") or {}).get("why") or (
            "the manifest carries no reading"
        )
        bad.append(
            f"the modelled side's arrival barrier was never read ({why}): "
            f"an unread barrier is not a barrier that held, so this run's "
            f"latencies are unverified rather than verified"
        )
    for label, run, mode, clock in (
        ("real", real, "measure", "wall"),
        ("modelled", modelled, "predict", "virtual"),
    ):
        compass = (run.manifest.get("server") or {}).get("compass") or {}
        if compass.get("mode") != mode:
            bad.append(
                f"the {label} side ran in compass mode "
                f"{compass.get('mode')!r}, not {mode!r}"
            )
        if label == "modelled" and not compass.get("virtual_clock"):
            bad.append(
                "the modelled side did not hold a virtual clock, so "
                "its timings are a wall-clock measurement of a simulator"
            )
        if label == "modelled":
            warmup = (compass.get("oracle_options") or {}).get("warmup_seconds")
            if warmup not in (None, 0, 0.0, "0", "0.0"):
                bad.append(
                    f"the modelled side was given warmup_seconds="
                    f"{warmup!r}: a warmed cell charges no first-use "
                    f"constant (PROTOCOL.md section 6)"
                )
        if run.clock not in ("?", clock) and label == "real" and run.clock == "virtual":
            bad.append("the real side reports a virtual clock")
    return bad


def check_engine(run, tp: int, label: str) -> list[str]:
    """The configuration the server reports, against the registered one."""
    bad = []
    server = run.manifest.get("server") or {}
    if not server:
        bad.append(
            f"{label}: no server provenance, so the configuration that "
            f"produced this cannot be read from it"
        )
        return bad
    if server.get("tensor_parallel_size") not in (None, tp):
        bad.append(
            f"{label}: served at TP={server.get('tensor_parallel_size')}"
            f", not the cell's TP={tp}"
        )
    for key, want in ENGINE.items():
        have = server.get(key)
        if have is None:
            bad.append(f"{label}: the server does not report {key}")
        elif isinstance(want, float):
            if abs(float(have) - want) > 1e-9:
                bad.append(f"{label}: {key} is {have}, not the registered {want}")
        elif bool(have) != want if isinstance(want, bool) else have != want:
            bad.append(f"{label}: {key} is {have!r}, not the registered {want!r}")
    return bad


def _rolled_digest(contents: dict) -> str:
    """The server's own digest over a set of files, recomputed here.

    `_artifact_digests` in the server hashes `name:sha\\n` per file in sorted
    order when an option stands for more than one file. Recomputing it is how a
    declared enumeration is checked against the digest the server reported,
    rather than being taken on the registry's word.
    """
    rolled = hashlib.sha256()
    for name in sorted(contents):
        rolled.update(f"{name}:{contents[name]}\n".encode())
    return rolled.hexdigest()


def _hexish(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


def check_artifact_provenance(role, entry, observed_files, workload_sha, forbidden):
    """What a calibration artifact must say about where it came from.

    A digest identifies a file; it does not say what produced it, and two of
    the PoC's option values are directories whose contents a single digest
    hides. So an entry must enumerate what the server actually loaded, name the
    measurements it was derived from, and pin the code that derived it -- each
    with a digest, so a later edit is visible. Every one of those digests is
    then checked for the same leakage the top-level digest is checked for: a
    source that is this cell's own step table is a source whether it is reached
    in one hop or two.
    """
    bad = []
    sha = entry.get("sha256") or ""
    tag = f"{role}={sha[:16]}"

    contents = entry.get("contents")
    if observed_files:
        if not isinstance(contents, dict) or not contents:
            bad.append(
                f"{tag}: the server loaded {len(observed_files)} file(s) for "
                f"this option and the registry enumerates none of them, so "
                f"what was read is declared only as one digest"
            )
        else:
            missing = sorted(set(observed_files) - set(contents))
            extra = sorted(set(contents) - set(observed_files))
            differing = sorted(
                name
                for name in set(contents) & set(observed_files)
                if contents[name] != observed_files[name]
            )
            if missing:
                bad.append(
                    f"{tag}: the server read {', '.join(missing)} which the "
                    f"registry does not declare"
                )
            if extra:
                bad.append(
                    f"{tag}: the registry declares {', '.join(extra)}, which "
                    f"the server did not read"
                )
            if differing:
                bad.append(
                    f"{tag}: {', '.join(differing)} differs between what the "
                    f"server read and what the registry declares"
                )
            if (
                not (missing or extra or differing)
                and len(contents) > 1
                and _rolled_digest(contents) != sha
            ):
                bad.append(
                    f"{tag}: the declared contents do not roll up to the "
                    f"digest the server reported"
                )
    elif isinstance(contents, dict) and contents:
        bad.append(
            f"{tag}: the registry enumerates contents for an option the "
            f"server did not report reading any file for"
        )

    sources = entry.get("sources")
    if not isinstance(sources, list) or not sources:
        bad.append(
            f"{tag}: declares no sources, so the measurement behind it is "
            f"named nowhere and the declaration is its own evidence"
        )
        sources = []
    code = entry.get("code")
    if not isinstance(code, dict) or not code:
        bad.append(
            f"{tag}: declares no code digests, so the module that produced "
            f"it can change without anything here changing"
        )
        code = {}
    for name, digest in sorted(code.items()):
        if not _hexish(digest):
            bad.append(f"{tag}: the code digest for {name} is not a sha256")

    if entry.get("kind") == "overhead_constant" and not _finite(entry.get("value")):
        bad.append(
            f"{tag}: an overhead constant with no value written down cannot "
            f"be checked against anything"
        )

    reachable = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not source.get("path"):
            bad.append(f"{tag}: source {index} carries no path")
            continue
        digest = source.get("sha256")
        if not _hexish(digest):
            bad.append(
                f"{tag}: source {source['path']} carries no sha256, so it "
                f"names a file rather than a file's contents"
            )
            continue
        reachable.append((f"source {source['path']}", digest))
    if isinstance(contents, dict):
        reachable += [(f"loaded file {n}", d) for n, d in sorted(contents.items())]

    for what, digest in reachable:
        if digest == workload_sha:
            bad.append(
                f"{tag}: its {what} is the acceptance workload itself, so "
                f"the predictor reaches the run it is predicting"
            )
        for name, forbidden_sha in forbidden.items():
            if forbidden_sha and digest == forbidden_sha:
                bad.append(
                    f"{tag}: its {what} is this cell's own {name}, reached "
                    f"through the artifact rather than directly"
                )
    return bad


def check_calibration(
    modelled, registry: dict, tp: int, workload_sha: str, forbidden: dict
) -> list[str]:
    """Where the predictor's numbers came from, against a declaration of it.

    Fail-closed in both directions: an artifact the server read and nobody
    declared is a refusal, and a declared artifact measured at the width being
    predicted is a refusal even if it is the only thing the oracle read.

    The declaration itself is then checked against what the server reported
    loading, file by file, and against the sources and code the registry says
    produced it -- see `check_artifact_provenance`. A registry that names an
    artifact but not what is inside it, or what made it, is metadata rather
    than provenance.
    """
    bad = []
    compass = (modelled.manifest.get("server") or {}).get("compass") or {}
    digests = compass.get("oracle_option_sha256") or {}
    option_files = compass.get("oracle_option_files") or {}
    named = {
        k: v
        for k, v in (compass.get("oracle_options") or {}).items()
        if isinstance(v, str) and v
    }
    if not digests and named:
        bad.append(
            f"the modelled server names {sorted(named)} but reported no "
            f"digest for any of them: what it was fitted to is unrecorded"
        )
    by_sha = {a.get("sha256"): a for a in (registry.get("artifacts") or [])}
    for role, sha in sorted(digests.items()):
        entry = by_sha.get(sha)
        if entry is None:
            bad.append(
                f"the modelled server read {role}={sha[:16]}, which the "
                f"calibration registry does not declare: it cannot be "
                f"attributed to a measurement"
            )
            continue
        kind = entry.get("kind")
        if kind not in CALIBRATION_KINDS:
            bad.append(
                f"{role}={sha[:16]} is declared kind {kind!r}, which is "
                f"not one the protocol recognises"
            )
            continue
        required = CALIBRATION_KINDS[kind]
        at = entry.get("measured_at_tp")
        if required is not None and at not in (None, required):
            bad.append(
                f"{role}={sha[:16]} is a {kind} measured at TP={at}, "
                f"but only TP={required} may be a source for it; at "
                f"TP={tp} this is the width being predicted"
            )
        if kind == "standalone_primitive" and at not in (None, 1, tp):
            bad.append(
                f"{role}={sha[:16]} is a standalone primitive measured "
                f"at TP={at}, which is neither the source width nor "
                f"this cell's TP={tp}"
            )
        if entry.get("workload_sha256") == workload_sha:
            bad.append(
                f"{role}={sha[:16]} was produced from the acceptance "
                f"workload itself, so the predictor was fitted to the "
                f"run it is predicting"
            )
        if entry.get("from_target_engine"):
            bad.append(
                f"{role}={sha[:16]} declares it came from the target "
                f"engine at the evaluated width"
            )
        for what, forbidden_sha in forbidden.items():
            if forbidden_sha and sha == forbidden_sha:
                bad.append(
                    f"{role}={sha[:16]} is this cell's own {what}: the "
                    f"predictor read the measurement it is predicting"
                )
        observed = option_files.get(role) or {}
        bad += check_artifact_provenance(role, entry, observed, workload_sha, forbidden)
    for role, found in sorted(option_files.items()):
        if role not in digests:
            bad.append(
                f"the modelled server read {len(found)} file(s) for {role} "
                f"and reported no digest for it"
            )
    return bad


# --------------------------------------------------------------------------
# the source-derived predictor, as it was actually invoked


#: The factory the integrated source oracle is reached through, exactly as it
#: must appear after `--compass-oracle`. Compared whole: a module path that
#: resolves to something else, or an older name, is a different predictor with
#: the same story told about it.
SOURCE_FACTORY = "atom.compass.runtime.source_oracle.source_cost_oracle"

#: Every keyword the factory accepts. `source_cost_oracle` takes `**kwargs` and
#: forwards, so a misspelling does raise -- but it raises inside the factory,
#: and only inside the factory this list was taken from. Keeping the list here
#: means a record naming an option nobody implements is refused rather than
#: read as a setting that took effect.
SOURCE_FACTORY_OPTIONS = (
    "model",
    "tp",
    "device",
    "replay_target",
    "block_size",
    "max_model_len",
    "position_rows",
    "block_policy",
    "cudagraph_mode",
    "price",
    "template",
    "head_template",
    "head",
    "regions",
    "seconds_per_launch",
    "require_complete",
    "carry_allocation",
    "derive",
    "interpolate",
)

#: Options an acceptance run must state rather than default. Each has a working
#: default, which is the problem: tp, require_complete, head and regions all
#: decide how much of the model is actually priced, and a value nobody chose is
#: not coverage.
SOURCE_FACTORY_STATED = ("tp", "require_complete", "head", "regions")


def _factory_flag(value):
    """What the factory would make of this, or None if it would refuse it.

    `arg_utils` converts a value that parses as a number, so `head=1` is
    recorded as `1` and `head=true` as `"true"`. Both reach the factory; both
    are read the same way here.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("1", "true", "yes", "on"):
            return True
        if word in ("0", "false", "no", "off"):
            return False
    return None


def _given(options: dict, name: str) -> bool:
    """Whether the run stated this option, as opposed to taking the default."""
    return name in options and options[name] is not None and options[name] != ""


def check_source_factory(modelled, tp: int, label: str) -> list[str]:
    """The predictor an acceptance cell ran, and how it was configured.

    The binding is decided: a cc-traces acceptance cell is a cell predicted by
    the declared complete transfer factory. Another oracle is not a weaker
    acceptance run, it is a different experiment -- so it is refused here
    rather than skipping this check and quietly counting as one. The
    explicitly diagnostic route exists for that case, and `check_not_diagnostic`
    is what keeps the two apart.

    Naming the factory is not enough on its own. The run must also have
    configured it so that the whole model was priced: an unsupported or
    half-stated configuration must not reach a verdict looking like a complete
    one. Every coupling below is read off the factory rather than chosen here,
    except the four the protocol adds -- completeness is required, the head
    region is priced, the region profile is a real one rather than `none`, and
    the carried-allocation approximation, which the factory itself documents as
    unmeasured, is not available to a run being graded.
    """
    compass = (modelled.manifest.get("server") or {}).get("compass") or {}
    named = compass.get("oracle")
    if named != SOURCE_FACTORY:
        return [
            (
                f"{label}: the modelled side was predicted by {named!r} rather "
                f"than the declared acceptance factory {SOURCE_FACTORY}; "
                f"whatever it measured, it is not a cc-traces acceptance cell"
            )
        ]
    bad = []
    options = dict(compass.get("oracle_options") or {})

    unknown = sorted(set(options) - set(SOURCE_FACTORY_OPTIONS))
    if unknown:
        bad.append(
            f"{label}: the modelled server passed {unknown} to "
            f"{SOURCE_FACTORY}, which takes no such option: whatever those "
            f"were meant to set, they did not set it"
        )

    flags = {}
    for name in ("head", "require_complete", "carry_allocation", "derive"):
        if name not in options:
            continue
        flags[name] = _factory_flag(options[name])
        if flags[name] is None:
            bad.append(
                f"{label}: {name}={options[name]!r} is not a boolean the "
                f"factory reads, so what it was set to is unknown"
            )
    # An absent flag is its documented default, which is what the factory used.
    head = flags.get("head", False)
    require_complete = flags.get("require_complete", True)
    carry_allocation = flags.get("carry_allocation", False)
    derive = flags.get("derive", True)

    for name in SOURCE_FACTORY_STATED:
        if not _given(options, name):
            bad.append(
                f"{label}: the modelled server left {name} to the factory "
                f"default rather than stating it, so the record does not say "
                f"what was predicted"
            )

    if require_complete is False:
        bad.append(
            f"{label}: require_complete is off, so a shape the predictor "
            f"could not price was answered with an approximation instead of "
            f"refused: the prediction is not complete-only"
        )
    if head is False:
        bad.append(
            f"{label}: head is off, so the head region was not priced at all "
            f"and nothing in the numbers says it is missing"
        )
    if carry_allocation is True:
        bad.append(
            f"{label}: carry_allocation is on, which the factory documents as "
            f"an explicitly unmeasured assumption: it may be reported, not "
            f"graded"
        )
    # Interpolation is allowed in an acceptance cell -- a fitted price inside
    # the support its own measurements declare is a prediction, and the
    # coverage record says which operators were fitted. What is not allowed is
    # a density nobody can read: `interpolate=maybe` reaches the factory, which
    # refuses it, so a record carrying one describes a run that never started.
    if _given(options, "interpolate"):
        from atom.compass.runtime.source_oracle import gap_ratio

        try:
            limit = gap_ratio(options["interpolate"])
        except ValueError as exc:
            bad.append(
                f"{label}: interpolate={options['interpolate']!r} is not a "
                f"sampling density the factory reads ({exc}), so what fitted "
                f"prices were allowed to span is unknown"
            )
        else:
            # `true` builds a working predictor at the provider's own default,
            # which is the right default and the wrong record: the manifest
            # keeps the option as written, so a cell graded under `true` has
            # nothing in it saying how wide a gap was crossed, and the answer
            # moves if the provider's default ever does.
            if not isinstance(limit, float):
                bad.append(
                    f"{label}: interpolate={options['interpolate']!r} takes "
                    f"whatever density the provider currently defaults to, so "
                    f"the record does not state the support fitted prices were "
                    f"allowed over; state the ratio"
                )

    if _factory_flag(options.get("regions")) is False or str(
        options.get("regions", "")
    ).strip().lower() in ("none", "null"):
        bad.append(
            f"{label}: regions={options.get('regions')!r} is not a source "
            f"region profile, so the prediction is not attributed to any "
            f"region model at all"
        )

    stated_tp = options.get("tp")
    served_tp = (modelled.manifest.get("server") or {}).get("tensor_parallel_size")
    if stated_tp is not None:
        if str(stated_tp) != str(tp):
            bad.append(
                f"{label}: the factory was given tp={stated_tp!r} in a TP={tp} "
                f"cell, so the predictor was built for another width"
            )
        if served_tp is not None and str(stated_tp) != str(served_tp):
            bad.append(
                f"{label}: the factory was given tp={stated_tp!r} but the "
                f"server reports tensor_parallel_size={served_tp!r}"
            )

    # What the factory itself refuses, or silently has nothing to answer with.
    if derive:
        for name in ("model", "block_size", "max_model_len"):
            if not _given(options, name):
                bad.append(
                    f"{label}: derive is on and {name} was not given; the "
                    f"factory needs it to trace the shapes no template covers"
                )
    else:
        if not _given(options, "template"):
            bad.append(
                f"{label}: derive is off and no template was given, so every "
                f"shape would be refused for want of a graph"
            )
        if head and not _given(options, "head_template"):
            bad.append(
                f"{label}: head is on with derive off and no head_template, "
                f"so the head region has neither a graph nor a deriver"
            )
    return bad


# --------------------------------------------------------------------------
# the device-free proof


#: Device nodes whose presence means this container can reach an accelerator.
#: `/dev/dri` is a directory; the rest are character devices. Presence, not
#: usability, is the test: a fallback that is reachable is a fallback that can
#: be taken, and the claim under test is that the prediction needed no device
#: at all.
DEVICE_NODES = (
    "/dev/kfd",
    "/dev/dri",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia0",
)

#: Variables that hide a device from a library without removing it from the
#: container. They are recorded and never accepted as the proof: an empty
#: `HIP_VISIBLE_DEVICES` is a statement about what torch will enumerate, and
#: the process can still open `/dev/kfd` itself.
MASKING_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

GPU_FREE_EVIDENCE = "gpu_free.json"

#: Bumped when the probe learns to look somewhere new. Evidence from an older
#: probe is refused rather than read: it looked at less.
GPU_FREE_PROBE = 1


def _driver_handles() -> list:
    """Open file descriptors onto a driver, held by anything in this container.

    A container with no device nodes cannot have these. A container that has
    them, masked, usually does -- which is why this is asked separately from
    the node list.
    """
    handles = []
    for proc in sorted(Path("/proc").glob("[0-9]*")):
        fd_dir = proc / "fd"
        try:
            entries = sorted(fd_dir.iterdir())
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith(("/dev/kfd", "/dev/dri", "/dev/nvidia")):
                handles.append({"pid": proc.name, "fd": entry.name, "target": target})
    return handles


def _torch_view() -> dict:
    """What the runtime itself says, asked in this container."""
    try:
        import torch
    except Exception as bad:  # noqa: BLE001 - the reason is the evidence
        return {"imported": False, "error": f"{type(bad).__name__}: {bad}"}
    view = {"imported": True, "version": getattr(torch, "__version__", None)}
    try:
        view["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as bad:  # noqa: BLE001
        view["cuda_available"] = None
        view["cuda_error"] = f"{type(bad).__name__}: {bad}"
    try:
        view["device_count"] = int(torch.cuda.device_count())
    except Exception as bad:  # noqa: BLE001
        view["device_count"] = None
        view["device_count_error"] = f"{type(bad).__name__}: {bad}"
    return view


def observe_device_freedom(covers: dict) -> dict:
    """The observation itself, taken from inside the container under test."""
    return {
        "probe_version": GPU_FREE_PROBE,
        "taken_in": {
            "hostname": platform.node(),
            "cwd": os.getcwd(),
        },
        "device_nodes": {node: os.path.exists(node) for node in DEVICE_NODES},
        "driver_handles": _driver_handles(),
        "masking_vars": {
            var: os.environ.get(var) for var in MASKING_VARS if var in os.environ
        },
        "torch": _torch_view(),
        "covers": covers,
        "means": (
            "the container this ran in holds no device node and no open driver "
            "handle; masking variables are recorded for the reader and are not "
            "what makes this pass"
        ),
        "does_not_mean": (
            "that the artifacts named in `covers` were produced by this exact "
            "process -- it means they existed, with these digests, in this "
            "device-free container when it was asked"
        ),
    }


def gpu_free(args) -> int:
    """Record the device-free observation into a cell. Run it in the cell's own
    container, after the modelled repeats, so what it covers already exists."""
    cell_dir = Path(args.dir)
    if not cell_dir.is_dir():
        print(f"  FAIL: {cell_dir} is not a directory, so there is no cell here")
        return 1
    covers = {p.name: _digest(p) for p in _runs(cell_dir, "modelled")}
    if not covers:
        print(
            f"  FAIL: no modelled runs in {cell_dir}: a device-free probe "
            f"that covers nothing proves nothing"
        )
        return 1
    evidence = observe_device_freedom(covers)
    (cell_dir / GPU_FREE_EVIDENCE).write_text(json.dumps(evidence, indent=1) + "\n")
    present = [n for n, there in evidence["device_nodes"].items() if there]
    print(
        f"{cell_dir.name}: {len(covers)} modelled run(s) covered, "
        f"{len(present)} device node(s) present, "
        f"{len(evidence['driver_handles'])} driver handle(s)"
    )
    return 0 if not present and not evidence["driver_handles"] else 1


def check_gpu_free(cell_dir: Path, modelled_paths: list) -> list[str]:
    """Was the prediction produced somewhere a device could not have helped?

    The gate is a GPU-free replay, so the absence has to be a property of the
    container rather than of an environment variable. This reads an observation
    taken inside that container and refuses anything weaker.
    """
    path = cell_dir / GPU_FREE_EVIDENCE
    if not path.exists():
        return [
            (
                f"no {GPU_FREE_EVIDENCE}: nothing observed that the modelled "
                f"side ran where no device was reachable, so the GPU-free "
                f"claim rests on how the run was described"
            )
        ]
    try:
        evidence = json.loads(path.read_text())
    except json.JSONDecodeError as bad:
        return [f"{GPU_FREE_EVIDENCE} is not readable JSON: {bad}"]
    bad = []
    if evidence.get("probe_version") != GPU_FREE_PROBE:
        bad.append(
            f"{GPU_FREE_EVIDENCE} was written by probe version "
            f"{evidence.get('probe_version')!r}, not {GPU_FREE_PROBE}: it "
            f"looked at a different set of places"
        )
    nodes = evidence.get("device_nodes")
    masked = evidence.get("masking_vars") or {}
    if not isinstance(nodes, dict) or set(nodes) != set(DEVICE_NODES):
        bad.append(
            f"{GPU_FREE_EVIDENCE} does not report every device node the "
            f"protocol asks about ({', '.join(DEVICE_NODES)})"
        )
    else:
        present = sorted(n for n, there in nodes.items() if there)
        if present:
            bad.append(
                f"the modelled side ran in a container that can reach "
                f"{', '.join(present)}"
                + (
                    f"; {sorted(masked)} being set is masking, not device " f"freedom"
                    if masked
                    else ""
                )
            )
    handles = evidence.get("driver_handles")
    if handles is None:
        bad.append(
            f"{GPU_FREE_EVIDENCE} does not say whether any driver " f"handle was open"
        )
    elif handles:
        targets = sorted({h.get("target") for h in handles if isinstance(h, dict)})
        bad.append(
            f"{len(handles)} open driver handle(s) in the modelled "
            f"container ({', '.join(t for t in targets if t)})"
        )
    torch_view = evidence.get("torch") or {}
    if not torch_view.get("imported"):
        bad.append(
            f"the probe could not import torch ({torch_view.get('error')}), "
            f"so the runtime was never asked what it could see"
        )
    else:
        if torch_view.get("cuda_available"):
            bad.append(
                "torch reports an accelerator available in the modelled container"
            )
        count = torch_view.get("device_count")
        if count is None:
            bad.append("the probe could not read a device count from torch")
        elif int(count) != 0:
            bad.append(f"torch reports {count} device(s) in the modelled container")
    covers = evidence.get("covers")
    if not isinstance(covers, dict) or not covers:
        bad.append(
            f"{GPU_FREE_EVIDENCE} covers no artifact, so it is an "
            f"observation about a container and not about this cell's runs"
        )
    else:
        for run_path in modelled_paths:
            have = covers.get(run_path.name)
            if have is None:
                bad.append(
                    f"the device-free probe does not cover {run_path.name}: "
                    f"that modelled run is not shown to be device-free"
                )
            elif have != _digest(run_path):
                bad.append(
                    f"{run_path.name} has changed since the device-free "
                    f"probe saw it ({_digest(run_path)[:16]} against "
                    f"{have[:16]})"
                )
    return bad


# --------------------------------------------------------------------------
#: A cell directory carrying this file is a diagnostic, whatever it is called.
#: The name is not the evidence -- a diagnostic can be renamed, and an
#: acceptance cell can be given a frightening name and still be one.
DIAGNOSTIC_MARKER = "DIAGNOSTIC.json"

#: The only purpose an acceptance verdict may be computed from. New execution
#: evidence has to say this word: `cc_traces_run.py` writes the purpose into
#: every execution record, every journal and the stamp inside every artifact,
#: so a run that says nothing was either produced by something else or edited,
#: and neither is a run we can grade. Stored verdicts are the exception --
#: those predate the field, and a verdict with no purpose keeps the protocol
#: it was computed under rather than being retroactively refused.
ACCEPTANCE_PURPOSE = "acceptance"


def _purpose_of(blob) -> str:
    """The purpose of a stored verdict, with silence meaning acceptance."""
    return (blob or {}).get("purpose") or ACCEPTANCE_PURPOSE


def _stated_purpose(blob):
    """The purpose an execution record actually states, or None if it does not.

    Deliberately not `_purpose_of`: for evidence being graded now, the
    difference between "this says acceptance" and "this says nothing" is the
    whole point. Silence is not a weak yes.
    """
    said = (blob or {}).get("purpose")
    if said is None or (isinstance(said, str) and not said.strip()):
        return None
    return said


def check_not_diagnostic(cell_dir: Path, journals: dict, manifests: dict) -> list[str]:
    """Whether anything here says this was run for something else.

    A diagnostic exercises the same harness, writes the same file names and
    produces the same shapes. What separates it from a cell is what it was
    *for*, and that is not a property of the directory: copying the artifacts
    somewhere better-named changes the path and nothing else. So the purpose
    travels in the execution evidence -- the journal, each execution record,
    and the stamp inside each artifact -- and any one of them saying
    `diagnostic` refuses the cell.

    Evidence that says nothing is refused too. A missing purpose used to read
    as acceptance, which made the weakest possible artifact -- one that never
    declared what it was for -- the one that passed unquestioned. The harness
    has written the field for as long as the acceptance protocol has existed,
    so anything reaching this validator without it was produced by some other
    tool or edited afterwards.

    Refusing rather than noting, and refusing on the marker file as well: a
    diagnostic that reaches the validator at all is a mistake somewhere
    upstream, and the cheap failure is the one that happens here.
    """
    problems = []
    if (cell_dir / DIAGNOSTIC_MARKER).exists():
        problems.append(
            f"{DIAGNOSTIC_MARKER} is present: this directory was written as a "
            f"diagnostic, and a diagnostic has no acceptance verdict to give "
            f"however it is named"
        )
    for side, journal in journals.items():
        said = _stated_purpose(journal)
        if said is None:
            problems.append(
                f"{side}: run.{side}.json does not say what it was run for, "
                f"and an acceptance cell has to say so"
            )
        elif said != ACCEPTANCE_PURPOSE:
            problems.append(
                f"{side}: run.{side}.json was written for {said!r}, not " f"acceptance"
            )
        for index, execution in enumerate((journal or {}).get("executions") or []):
            said = _stated_purpose(execution)
            if said is None:
                problems.append(
                    f"{side}[{index}]: this execution does not say what it "
                    f"was run for, so it cannot be counted as acceptance"
                )
            elif said != ACCEPTANCE_PURPOSE:
                problems.append(
                    f"{side}[{index}]: this execution was run for {said!r}, "
                    f"not acceptance"
                )
    for side, blobs in manifests.items():
        for index, blob in enumerate(blobs):
            said = _stated_purpose((blob or {}).get("execution"))
            if said is None:
                problems.append(
                    f"{side}[{index}]: the artifact's stamp does not say what "
                    f"the run was for, so it is not acceptance evidence"
                )
            elif said != ACCEPTANCE_PURPOSE:
                problems.append(
                    f"{side}[{index}]: the artifact carries the stamp of a "
                    f"{said!r} run, so it was copied here rather than "
                    f"produced here"
                )
    return problems


# the cell


def _runs(cell: Path, side: str) -> list[Path]:
    """Repeat artifacts, in a stable order, excluding preparation records."""
    found = sorted(
        p
        for p in cell.glob(f"{side}*.json")
        if not p.name.endswith(".prepare.json") and not p.name.endswith("_steps.json")
    )
    return found


def _digest(path: Path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _same_host(left, right) -> bool:
    """Short names: one side may carry a domain the other does not."""
    return str(left or "").split(".")[0] == str(right or "").split(".")[0]


def _who_answered(execution, where: str) -> list[str]:
    """Re-derive, from the raw fields, that the launched process answered.

    The harness makes this check while it runs and writes down a verdict. A
    verdict is not evidence. What is recomputed here is the comparison itself,
    from `said` (the server's account of which process it is) and `observed`
    (what the harness read out of `/proc`), so a run whose recorded conclusion
    does not follow from its own recorded facts is refused rather than
    believed.
    """
    seen = execution.get("server_process")
    if not isinstance(seen, dict) or not seen:
        return [
            (
                f"{where}: nothing recorded about which process answered, so this "
                f"repeat cannot be attributed to the server it launched -- a "
                f"matching code digest says only that some server was built from "
                f"the same tree"
            )
        ]
    said, observed = seen.get("said"), seen.get("observed")
    if not isinstance(said, dict) or not isinstance(observed, dict):
        return [
            (
                f"{where}: the record of who answered is incomplete, so there is "
                f"nothing to recompute it from"
            )
        ]
    wrong = []
    pid, launched = said.get("pid"), observed.get("launched_pid")
    ancestry = observed.get("ancestry") or []
    if pid is None or launched is None:
        wrong.append("neither account names a process")
    elif launched not in ancestry:
        wrong.append(
            f"pid {pid} answered but is neither the launched process "
            f"{launched} nor a descendant of it"
        )
    if said.get("start_ticks") != observed.get("start_ticks"):
        wrong.append(
            f"pid {pid} reports start tick {said.get('start_ticks')!r} where "
            f"the harness read {observed.get('start_ticks')!r}, so that pid "
            f"named a different process than the one that answered"
        )
    if not _same_host(said.get("host"), observed.get("host")):
        wrong.append(
            f"the server answered from {said.get('host')!r} but the repeat "
            f"was launched on {observed.get('host')!r}"
        )
    if (
        said.get("boot_id")
        and observed.get("boot_id")
        and said["boot_id"] != observed["boot_id"]
    ):
        wrong.append("the server reports a different boot than the harness")
    if observed.get("alive_at_provenance") is not True:
        wrong.append(
            "the launched process was not alive when the server answered, so "
            "whatever answered was not it"
        )
    problems = [f"{where}: {reason}" for reason in wrong]
    if seen.get("verified") is not True:
        problems.append(
            f"{where}: the run does not claim the server was verified, so "
            f"nothing attributes these numbers to a known process"
        )
    elif wrong:
        problems.append(
            f"{where}: the run records the server as verified, but its own "
            f"evidence does not support that"
        )
    return problems


def _who_served(execution, manifest, where: str) -> list[str]:
    """And that the process checked at startup is the one that served.

    The harness reads `/compass/provenance` when the server comes up. The
    replay client reads it again, from a different process, during the run it
    is recording. Startup and service are different moments, and only the
    second one is the one the numbers come from, so the two accounts have to
    name the same process.
    """
    seen = execution.get("server_process") or {}
    said = seen.get("said") if isinstance(seen, dict) else None
    if not isinstance(said, dict):
        return []  # already refused by the check above
    theirs = (manifest.get("server") or {}).get("server_process")
    if not isinstance(theirs, dict) or not theirs:
        return [
            (
                f"{where}: the replay artifact records no account of which "
                f"process served the requests, so the check made at startup "
                f"covers only startup"
            )
        ]
    for field in ("pid", "start_ticks", "boot_id"):
        if theirs.get(field) != said.get(field):
            return [
                (
                    f"{where}: the server checked at startup reports "
                    f"{field}={said.get(field)!r} but the server that served the "
                    f"replay reports {theirs.get(field)!r}, so the requests were "
                    f"answered by a different process than the one verified"
                )
            ]
    return []


def check_who_served(journal, manifests: list, label: str) -> list[str]:
    """Whether this side's numbers can be attributed to processes it launched.

    Offline, from what the run wrote down. The harness refuses at the time,
    but a refusal that only ever happens live is a refusal nobody can audit
    afterwards, and the artifacts outlive the process that made them.
    """
    if not isinstance(journal, dict) or not journal:
        return [
            (
                f"{label}: no run.{label}.json, so nothing records which process "
                f"served these repeats"
            )
        ]
    executions = journal.get("executions") or []
    if not executions:
        return [f"{label}: run.{label}.json records no executions"]
    problems = []
    if len(executions) != len(manifests):
        problems.append(
            f"{label}: {len(executions)} executions recorded against "
            f"{len(manifests)} saved repeats, so they cannot be matched up"
        )
    for index, execution in enumerate(executions):
        where = f"{label}[{index}]"
        problems += _who_answered(execution, where)
        if index < len(manifests):
            problems += _who_served(execution, manifests[index], where)
    return problems


def cell(args) -> int:
    cell_dir = Path(args.dir)
    if not cell_dir.is_dir():
        # Refused rather than created: a verdict written into a directory this
        # tool had to make is a verdict about nothing.
        print(f"  FAIL: {cell_dir} is not a directory, so there is no cell here")
        return 1
    klass = getattr(args, "class")
    rows = registered_rows(klass)
    workload_sha = _digest(WORKLOADS[klass])
    failures, notes = [], []

    stamp_path = cell_dir / "cc_traces_protocol.json"
    if not stamp_path.exists():
        failures.append(
            "no cc_traces_protocol.json: this cell does not say "
            "which protocol it ran under"
        )
    else:
        stamp = json.loads(stamp_path.read_text())
        if not stamp.get("matches_registration"):
            failures.append(
                "the cell's protocol stamp does not match the "
                "registration in force when it ran"
            )
        stamped = (stamp.get("workloads") or {}).get(f"cc_traces_{klass}")
        if stamped and stamped != workload_sha:
            failures.append(
                f"the cell ran against workload {stamped[:16]}, "
                f"not the registered {workload_sha[:16]}"
            )

    costs_path = cell_dir / "costs.json"
    costs = json.loads(costs_path.read_text()) if costs_path.exists() else {}
    said = costs.get("cost_schema")
    if costs and said != COSTS_SCHEMA:
        failures.append(
            f"costs.json is a {said!r} record, not {COSTS_SCHEMA!r}: its "
            f"execution terms are medians over this cell's repeats while its "
            f"journalled terms are totals across all of them, and nothing in "
            f"it says which repeat any second was spent in. A ratio of the "
            f"two understates the replay speedup by the repeat count. "
            f"Re-merge the cell rather than reinterpreting the old record"
        )
    _, timing_reasons = _timing(costs)
    failures += timing_reasons
    missing = [t for t in MEASURED_COST_TERMS if not _finite(costs.get(t))]
    for term in SUPPLIED_COST_TERMS:
        value = costs.get(term)
        if isinstance(value, (int, float)):
            # A version 2 bare float. It never said whether it happens inside
            # a measured window, so it cannot be placed in a total now.
            failures.append(
                f"costs.json records {term} as a bare number: that is a "
                f"{COSTS_SCHEMA_V2} record, whose supplied terms never stated "
                f"what contains them. `load` sits inside `startup_real` and "
                f"was charged twice by every total that read one. Re-merge "
                f"the cell rather than reinterpreting the old number"
            )
            continue
        parts = _parts(costs, term)
        if not parts or not all(_finite(p.get("seconds")) for p in parts):
            missing.append(term)
            continue
        for index, part in enumerate(parts):
            where = f"{term}[{index}]" if len(parts) > 1 else term
            if not part.get("source"):
                failures.append(
                    f"costs.json does not say where {where} came from: a "
                    f"supplied second without its artifact cannot be told "
                    f"from a measured one"
                )
            elif "within" not in part:
                failures.append(
                    f"costs.json does not say what contains {where}: unstated "
                    f"is not the same as contained by nothing, and a term "
                    f"whose containment is unknown cannot be summed either way"
                )
            elif part.get("within") not in (None,) + MEASURED_COST_TERMS:
                failures.append(
                    f"costs.json says {where} happens inside "
                    f"{part.get('within')!r}, which is not a window this cell "
                    f"measures ({', '.join(MEASURED_COST_TERMS)})"
                )
    if missing:
        failures.append(
            f"costs.json does not record {', '.join(missing)}: the "
            f"speedup claim cannot be separated from the capture "
            f"and calibration it rests on"
        )
    off_wall = _off_wall_clocks(costs)
    if off_wall:
        failures.append(
            f"costs.json does not record both execution terms as wall-clock "
            f"seconds ({', '.join(off_wall)}): a predicting engine serves on "
            f"a virtual clock, so the window it reports is what the "
            f"prediction says the workload would take and not what producing "
            f"the prediction cost"
        )

    isolation_path = cell_dir / "isolation.json"
    isolation = (
        json.loads(isolation_path.read_text()) if isolation_path.exists() else {}
    )
    if not isolation:
        failures.append(
            "no isolation.json: nothing says this cell had the " "node to itself"
        )
    elif isolation.get("verdict") == "own_contaminated":
        failures.append(
            "the isolation audit found this cell's own devices "
            "occupied by another tenant"
        )
    elif not isolation.get("isolated"):
        notes.append(
            f"isolation verdict {isolation.get('verdict')!r}: timings " f"are advisory"
        )

    registry = (
        json.loads(Path(args.calibration_registry).read_text())
        if args.calibration_registry
        else None
    )
    if registry is None:
        failures.append(
            "no calibration registry: where the predictor's "
            "numbers came from is undeclared, and the protocol's "
            "section 4 rule cannot be applied"
        )

    real_paths, modelled_paths = _runs(cell_dir, "real"), _runs(cell_dir, "modelled")
    if not real_paths or not modelled_paths:
        failures.append(
            f"{len(real_paths)} real and {len(modelled_paths)} "
            f"modelled runs in {cell_dir}"
        )
    if len(real_paths) < args.repeats or len(modelled_paths) < args.repeats:
        failures.append(
            f"{len(real_paths)} real and {len(modelled_paths)} "
            f"modelled repeats, fewer than the {args.repeats} the "
            f"protocol registers"
        )

    # The record has to be about these runs, not about a run of this cell that
    # is no longer here.
    for side, paths in (("real", real_paths), ("modelled", modelled_paths)):
        failures += check_costs_cover_runs(costs, side, paths, args.repeats)

    failures += check_gpu_free(cell_dir, modelled_paths)

    journals, blobs = {}, {}
    for side, paths in (("real", real_paths), ("modelled", modelled_paths)):
        journal_path = cell_dir / f"run.{side}.json"
        journals[side] = (
            json.loads(journal_path.read_text()) if journal_path.exists() else None
        )
        blobs[side] = [json.loads(p.read_text()) for p in paths]

    # Before anything is measured: a diagnostic run exercises this same harness
    # and writes these same file names, so what separates it from a cell is
    # what it was for. That is carried in the evidence rather than in the path,
    # and it is refused here whatever the directory is called.
    failures += check_not_diagnostic(cell_dir, journals, blobs)

    # A matching code digest says the server was built from this tree; it does
    # not say the replies came from the process this cell launched. The harness
    # checks that live and writes down what it saw. Recheck it here from the
    # artifacts, because a check that only ever runs live cannot be audited
    # after the process is gone.
    for side in ("real", "modelled"):
        failures += check_who_served(
            journals[side], [(b.get("run") or {}) for b in blobs[side]], side
        )

    # Every step table this cell wrote, whichever repeat wrote it and whichever
    # rank suffix the engine appended: a predictor calibrated on any of them is
    # calibrated on the run it is being compared against.
    forbidden = {
        f"step table {p.name}": _digest(p)
        for p in sorted(cell_dir.glob("*_steps*.jsonl"))
    }

    reports = []
    for index, (rp, mp) in enumerate(zip(real_paths, modelled_paths)):
        real = compare.load_run(str(rp), f"real[{index}]")
        modelled = compare.load_run(str(mp), f"modelled[{index}]")
        for run, label, paced in (
            (real, f"real[{index}]", True),
            (modelled, f"modelled[{index}]", False),
        ):
            failures += [
                f"{label}: {reason}"
                for reason in compare.check_run(run, expect_requests=len(rows))
            ]
            failures += [f"{label}: {reason}" for reason in check_clocks_finite(run)]
            failures += check_workload_is_registered(run, rows, label)
            failures += check_arrival_process(run, rows, label, paced)
            failures += check_engine(run, args.tp, label)
            for reason in check_usage_against_workload(run, rows, label):
                (notes if reason.startswith("NOTE") else failures).append(reason)
        failures += [
            f"repeat {index}: {reason}" for reason in compare.check_pair(real, modelled)
        ]
        failures += [
            f"repeat {index}: {reason}" for reason in check_side_roles(real, modelled)
        ]
        failures += check_source_factory(modelled, args.tp, f"repeat {index}")
        if registry is not None:
            failures += [
                f"repeat {index}: {reason}"
                for reason in check_calibration(
                    modelled, registry, args.tp, workload_sha, forbidden
                )
            ]
        reports.append(compare.compare(real, modelled))

    verdict = {
        "cell": str(cell_dir),
        "class": klass,
        "tp": args.tp,
        "workload_sha256": workload_sha,
        "purpose": ACCEPTANCE_PURPOSE,
        "repeats": len(reports),
        "costs": costs,
        "isolation": isolation.get("verdict"),
        "failures": failures,
        "notes": notes,
        "passed": not failures and bool(reports),
        "metrics": _across_repeats(reports),
        "speedup": _speedup(costs, args.reuse_cells),
    }
    (cell_dir / "cc_traces_cell.json").write_text(json.dumps(verdict, indent=1) + "\n")
    for reason in failures:
        print(f"  FAIL: {reason}")
    for reason in notes:
        print(f"  note: {reason}")
    print(
        f"{cell_dir.name}: class={klass} tp={args.tp} repeats={len(reports)} "
        f"{'PASS' if verdict['passed'] else 'REFUSED'}"
    )
    return 0 if verdict["passed"] else 1


def _stat(report: dict, metric: str, side: str):
    block = (report.get("metrics") or {}).get(metric) or {}
    if metric == "throughput_tok_s":
        return block.get(side)
    return (block.get(side) or {}).get("median")


def _across_repeats(reports: list[dict]) -> dict:
    """Per metric: each side's repeats, their spread, and the error between.

    The spread is what makes a tie checkable later. Two configurations whose
    real repeats overlap are not separated by the hardware, and a model that
    orders them either way has not disagreed with it.
    """
    out = {}
    for metric in ("throughput_tok_s", "ttft", "tpot", "latency"):
        real = [v for v in (_stat(r, metric, "real") for r in reports) if _finite(v)]
        modelled = [
            v for v in (_stat(r, metric, "modelled") for r in reports) if _finite(v)
        ]
        if not real or not modelled:
            continue
        centre_r = sorted(real)[len(real) // 2]
        centre_m = sorted(modelled)[len(modelled) // 2]
        out[metric] = {
            "real": real,
            "modelled": modelled,
            "real_centre": centre_r,
            "modelled_centre": centre_m,
            "real_range": [min(real), max(real)],
            # The model's own spread, on the same footing as the hardware's.
            # A separation is a claim either side can make, and §6 gates both
            # directions of disagreement about one, so the modelled repeats
            # have to be readable as a range and not only as a centre.
            "modelled_range": [min(modelled), max(modelled)],
            "error_pct": (
                ((centre_m - centre_r) / centre_r * 100.0) if centre_r else None
            ),
            "tolerance_pct": TOLERANCE_PCT.get(metric),
            "within_tolerance": (
                None
                if not centre_r or metric not in TOLERANCE_PCT
                else abs((centre_m - centre_r) / centre_r * 100.0)
                <= TOLERANCE_PCT[metric]
            ),
        }
    return out


def _off_wall_clocks(costs: dict) -> list:
    """Which execution terms this record does not say are wall-clock seconds.

    Silence counts as off-wall. The record this replaced wrote the engine's
    served window into `execution_modelled` and said nothing about it, so a
    missing declaration is exactly the case that has to be refused rather than
    read as a wall.
    """
    clocks = costs.get("execution_clocks")
    return [
        f"{side}={(clocks or {}).get(side)!r}"
        for side in ("real", "modelled")
        if not isinstance(clocks, dict) or clocks.get(side) != WALL_CLOCK
    ]


def _parts(costs: dict, term: str) -> list:
    """A supplied term's parts, whether it was written as one or as several.

    A structure the oracle derives on demand is derived when the schedule
    first shows it, which may be while the server is coming up or may be in
    the middle of serving -- `TemplateGraphs.graph_for` calls its deriver on a
    miss, and a miss has no phase. So one derivation total can straddle two
    measured windows, and a single container for the whole term cannot say
    that. A term is therefore a list of parts, each with its own container,
    and a lone object is the one-part case.

    A bare number is a version 2 record, which `_costs` refuses before
    reaching here. Returning nothing for a shape this does not recognise keeps
    the arithmetic from inventing seconds.
    """
    value = costs.get(term)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [p for p in value if isinstance(p, dict)]
    return []


def _supplied_seconds(costs: dict, term: str) -> float:
    """The term's whole duration, however deep each part sits inside a window."""
    total = 0.0
    for part in _parts(costs, term):
        seconds = part.get("seconds")
        if _finite(seconds):
            total += float(seconds)
    return total


def _outside(costs: dict, term: str, *windows: str) -> float:
    """The part of the term that none of `windows` already contains.

    `load` is inside `startup_real`, so a real-side total that adds it beside
    that startup counts the same work twice -- which is what version 2 of this
    record did, asymmetrically, inflating the real side by `load` and the
    modelled side by `derivation`. The seconds a window already holds are
    dropped here; the seconds outside every named window are real and are
    added. Passing no window asks for the parts contained by nothing at all.
    """
    total = 0.0
    for part in _parts(costs, term):
        seconds = part.get("seconds")
        if not _finite(seconds):
            continue
        within = part.get("within")
        if within and (not windows or within in windows):
            continue
        total += float(seconds)
    return total


def _uncontained(costs: dict, term: str) -> float:
    """The part no measured window holds, which an end-to-end total may add."""
    return _outside(costs, term)


def _repeat_of(part) -> object:
    """Which repeat a supplied part's seconds were spent in.

    `EVERY_REPEAT` is a part that happens once per repeat -- a cost the record
    states rather than one a reader allocates. `None` is a part nobody placed,
    and a part nobody placed cannot be matched to a per-repeat denominator.
    """
    value = part.get("repeat")
    if value == EVERY_REPEAT:
        return EVERY_REPEAT
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _by_repeat(costs: dict, key: str, side: str) -> dict:
    """`{repeat: seconds}` for one side's measured window, or `{}`."""
    block = ((costs.get(key) or {}).get(side)) or {}
    if not isinstance(block, dict):
        return {}
    out = {}
    for repeat, seconds in block.items():
        try:
            index = int(repeat)
        except (TypeError, ValueError):
            continue
        if _finite(seconds):
            out[index] = float(seconds)
    return out


def _repeat_coverage(costs: dict, side: str):
    """One side's per-repeat execution seconds, or a reason it cannot be used.

    The map is not the authority on which repeats exist -- the record's own
    `repeats` count is, and the map has to account for every one of them. A
    side that ran three times and lists two has not said what the third cost,
    and a median over the two that were written down is not the median the
    cost term carries. So the keys must be repeat numbers, each named once,
    each with a duration, and together exactly the repeats the side reports.
    """
    block = ((costs.get("execution_by_repeat") or {}).get(side)) or {}
    stated = (costs.get("repeats") or {}).get(side)
    count = int(stated) if _finite(stated) else None
    if not isinstance(block, dict) or not block:
        if count is not None and count > 1:
            return {}, (
                f"costs.json reports {count} {side} repeats but no "
                f"execution_by_repeat.{side}: its execution term is a median "
                f"over them and its supplied terms are totals across them, so "
                f"the two cannot be put in one ratio. Re-merge the cell"
            )
        return {}, None
    out = {}
    for key, seconds in block.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            return {}, (
                f"execution_by_repeat.{side} is keyed by {key!r}, which is not "
                f"a repeat number, so its seconds belong to no repeat"
            )
        if index in out:
            return {}, (
                f"execution_by_repeat.{side} names repeat {index} twice, so "
                f"one repeat's duration was replaced by another's and the "
                f"side covers fewer repeats than it lists"
            )
        if not _finite(seconds):
            return {}, (
                f"execution_by_repeat.{side} records no duration for repeat "
                f"{index} ({seconds!r}); a repeat nobody timed is not a repeat "
                f"that cost nothing"
            )
        out[index] = float(seconds)
    expected = set(range(1, (count if count is not None else len(out)) + 1))
    if set(out) != expected:
        return {}, (
            f"execution_by_repeat.{side} covers repeats "
            f"{sorted(out)}, not the {len(expected)} this record reports "
            f"({sorted(expected)}): a denominator built from part of the "
            f"repeats is not one repeat's share of the whole"
        )
    return out, None


def _foreign_repeats(costs: dict, term: str, known: set) -> list:
    """Nonzero parts of `term` charged to a repeat this record does not have.

    A part naming repeat 4 of a three-repeat cell is matched to nothing: it is
    not added to any repeat's denominator and it is not reported as belonging
    to none, so it leaves the arithmetic silently and the replay reads faster
    than it was measured to be.
    """
    out = []
    for part in _parts(costs, term):
        seconds = part.get("seconds")
        if not _finite(seconds) or float(seconds) == 0.0:
            continue
        where = _repeat_of(part)
        if where is None or where == EVERY_REPEAT or where in known:
            continue
        out.append((where, float(seconds)))
    return out


def _unplaced_repeats(costs: dict, term: str) -> float:
    """Nonzero seconds of `term` that name no repeat, wherever they sit.

    Once a cell says what each repeat cost, every second of a per-repeat term
    has a repeat: the journal places it by interval, and a part the placement
    missed is a part outside every window this cell measured. Its container
    does not rescue it -- a part inside the execution window that belongs to
    no repeat is a contradiction in the record, not a free second.
    """
    return sum(
        float(part.get("seconds"))
        for part in _parts(costs, term)
        if _finite(part.get("seconds"))
        and float(part.get("seconds")) != 0.0
        and _repeat_of(part) is None
    )


def _for_repeat(costs: dict, term: str, repeat: int, *windows: str) -> float:
    """This repeat's share of `term`, minus whatever `windows` already hold.

    A part belongs to this repeat when it says so, or when it says it happens
    in every repeat. Parts belonging to another repeat are another repeat's
    cost and are not divided into this one.
    """
    total = 0.0
    for part in _parts(costs, term):
        seconds = part.get("seconds")
        if not _finite(seconds):
            continue
        where = _repeat_of(part)
        if where not in (repeat, EVERY_REPEAT):
            continue
        within = part.get("within")
        if within and (not windows or within in windows):
            continue
        total += float(seconds)
    return total


def _timing(costs: dict):
    """What this cell's executions cost, read from the records of them.

    The per-execution records are the measurement: one row per run, each with
    its own duration. `execution_<side>` is a summary of those rows under the
    registered quantile convention, and a summary cannot be more authoritative
    than what it summarises -- a record carrying three twenty-second repeats
    and a 120-second total is not a 120-second cell, it is a cell beside a
    number from another run. So the term is derived from the rows wherever
    there are rows, and a stated total that disagrees with them is reported as
    the contradiction it is rather than quietly preferred.

    A cell that ran once has no rows to summarise and states its seconds
    directly. That is the one-shot diagnostic shape, and `cell` is what keeps
    it out of multi-repeat acceptance.

    Returns `({side: {"by_repeat": ..., "execution": ...}}, [reasons])`.
    """
    out, reasons = {}, []
    for side in ("real", "modelled"):
        coverage, why = _repeat_coverage(costs, side)
        stated = costs.get(f"execution_{side}")
        seconds = float(stated) if _finite(stated) else None
        if why:
            reasons.append(why)
        elif coverage:
            derived = _quantile(list(coverage.values()))
            if seconds is not None and abs(seconds - derived) > 1e-6 * max(
                1.0, derived
            ):
                reasons.append(
                    f"costs.json reports execution_{side}={seconds:.3f}s while "
                    f"its per-execution records say {derived:.3f}s: an "
                    f"aggregate that is not the median of the repeats it was "
                    f"taken over belongs to another run of this cell"
                )
            seconds = derived
        out[side] = {"by_repeat": coverage, "execution": seconds}
    return out, reasons


def _saved_repeats(paths) -> set:
    """The repeat numbers this cell actually saved a run artifact for."""
    found = set()
    for path in paths:
        match = re.search(r"\.r(\d+)\.json$", path.name)
        if not match:
            return None
        found.add(int(match.group(1)))
    return found


def check_costs_cover_runs(costs: dict, side: str, paths: list, required: int) -> list:
    """The cost record has to be about the executions this cell saved.

    `repeats` and `execution_by_repeat` agreeing with each other says the
    record is self-consistent, not that it is this run's. A cell that saved
    three runs and carries a one-repeat record is priced by an execution that
    is not in front of us, and every per-repeat quantity read off it is a
    quantity of something else.
    """
    coverage, why = _repeat_coverage(costs, side)
    if why:
        # Already reported once, by `_timing`. Saying it twice is not a
        # second finding.
        return []
    if not coverage:
        if required > 1:
            return [
                (
                    f"costs.json prices the {side} side in aggregate only and "
                    f"this cell ran {required} repeats: the gate divides one "
                    f"repeat's seconds, so every run this cell saved needs a "
                    f"per-execution record. Aggregate alone is the one-shot "
                    f"diagnostic shape"
                )
            ]
        return []
    saved = _saved_repeats(paths)
    if saved is None:
        return [
            (
                f"the {side} run artifacts are not named per repeat, so the "
                f"per-execution records in costs.json cannot be matched to "
                f"the runs this cell saved"
            )
        ]
    if set(coverage) != saved:
        return [
            (
                f"costs.json prices {side} repeats {sorted(coverage)} while "
                f"this cell saved runs for {sorted(saved)}: the record is "
                f"about a different run of this cell"
            )
        ]
    return []


def _quantile(values: list) -> float:
    """The one quantile convention, applied to a list of per-repeat totals."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.5 * len(ordered)))]


def _speedup(costs: dict, reuse_cells: int) -> dict:
    """The gate ratio, and every cost the gate does not include.

    The registered gate is the PoC's original one and has not been moved: a
    GPU-free replay of this workload, after the capture exists, against serving
    it for real, at least `SPEEDUP_MIN` times faster. `derivation` is inside the
    denominator because deriving this candidate's graph is work the prediction
    needs and a per-candidate cost; `capture` and `calibration` are not, because
    they happen once and the gate is about what asking one more question costs.

    **Both halves of the ratio are one repeat's seconds.** `execution_*` is the
    median over the cell's repeats -- what one replay cost -- so a derivation
    journal covering three repeats cannot be added to it whole. Doing that
    divided a per-repeat numerator by an all-repeats denominator and understated
    the ratio by the repeat count: three repeats of a ten-second replay each
    deriving for ten seconds, against two minutes of real serving, read 3x where
    asking one question costs twenty seconds and the answer is 6x. So the
    denominator is built per repeat and reduced under the same quantile
    convention as the terms it is built from.

    Everything the gate excludes is reported beside it rather than folded in:
    the acquisition seconds, the startup-inclusive ratio for a reader who wants
    end-to-end wall clock, the amortised ratio over a stated number of reusing
    cells, and the break-even count at which the modelled path has repaid its
    acquisition. An amortised ratio is a different and weaker claim than the
    gate, so it is never what `meets_gate` reads.
    """
    # One reading of what this cell's executions cost, shared with `cell`:
    # the per-execution records where there are any, the stated aggregate only
    # for a cell that ran once.
    timing, reasons = _timing(costs)
    if reasons:
        return {
            "replay_ratio": None,
            "amortised_ratio": None,
            "meets_gate": None,
            "reason": reasons[0],
        }
    execution_real = timing["real"]["execution"]
    execution_modelled = timing["modelled"]["execution"]
    if not (
        _finite(execution_real)
        and _finite(execution_modelled)
        and execution_modelled > 0
    ):
        return {
            "replay_ratio": None,
            "amortised_ratio": None,
            "meets_gate": None,
            "reason": "costs.json does not carry both execution terms",
        }
    # A speedup is a ratio of machine time. A predicting engine serves on a
    # virtual clock, so the window it reports is how long the prediction says
    # the workload would take -- a statement about accuracy, and off the wall
    # clock by more than an order of magnitude. Both sides have to say which
    # clock their execution term was taken on, and both have to say wall.
    off_wall = _off_wall_clocks(costs)
    if off_wall:
        return {
            "replay_ratio": None,
            "amortised_ratio": None,
            "meets_gate": None,
            "reason": (
                "costs.json does not record both execution terms as wall-clock "
                "seconds (" + ", ".join(off_wall) + "): a predicted duration "
                "is not what running the predictor cost, and a ratio of the "
                "two is not a speedup"
            ),
        }
    # Which repeats there are to divide by. A record that states more than one
    # repeat and does not say what each cost has nothing to match a journalled
    # term against, and the median it does carry is not one repeat's total.
    modelled_repeats = timing["modelled"]["by_repeat"]
    if modelled_repeats:
        # Every second of the term the gate divides by has to land on one of
        # the repeats this cell actually ran -- not on a repeat number the
        # record does not have, and not on no repeat at all.
        foreign = _foreign_repeats(costs, "derivation", set(modelled_repeats))
        if timing["real"]["by_repeat"]:
            foreign += _foreign_repeats(costs, "load", set(timing["real"]["by_repeat"]))
        if foreign:
            listed = ", ".join(f"repeat {r} ({s:.3f}s)" for r, s in sorted(foreign))
            return {
                "replay_ratio": None,
                "amortised_ratio": None,
                "meets_gate": None,
                "reason": (
                    f"costs.json charges {listed} to repeats this cell does "
                    f"not have (it ran {sorted(modelled_repeats)}), so those "
                    f"seconds are in no denominator and in no report"
                ),
            }
        stray = _unplaced_repeats(costs, "derivation")
        if stray > 0:
            return {
                "replay_ratio": None,
                "amortised_ratio": None,
                "meets_gate": None,
                "reason": (
                    f"{stray:.3f}s of derivation belongs to no repeat: it "
                    f"cannot be charged to one question without deciding how "
                    f"many questions it was for"
                ),
            }
    # Deriving this candidate's graphs is work the prediction needs, so the
    # gate's denominator has to include all of it -- and exactly once. A
    # structure first seen mid-schedule is derived inside the served window, so
    # those seconds are already in `execution_modelled`; adding the whole term
    # beside it charges them twice and makes the replay look slower than it
    # was. A structure derived while the server came up is not in that window
    # and is added. Nothing here decides which is which: a part that does not
    # say what contains it is refused by `_costs` before this runs.
    derivation = _supplied_seconds(costs, "derivation")
    derivation_added = _outside(costs, "derivation", "execution_modelled")
    derivation_in_execution = derivation - derivation_added
    acquisition = sum(_supplied_seconds(costs, t) for t in ("capture", "calibration"))
    startup_real = float(costs.get("startup_real") or 0.0)
    startup_modelled = float(costs.get("startup_modelled") or 0.0)
    if modelled_repeats:
        added_per_repeat = {
            index: _for_repeat(costs, "derivation", index, "execution_modelled")
            for index in modelled_repeats
        }
        predict_once = _quantile(
            [modelled_repeats[i] + added_per_repeat[i] for i in modelled_repeats]
        )
        derivation_added_reported = _quantile(list(added_per_repeat.values()))
        startups = _by_repeat(costs, "startup_by_repeat", "modelled")
        modelled_total = _quantile(
            [
                modelled_repeats[i]
                + startups.get(i, startup_modelled)
                + _for_repeat(costs, "derivation", i)
                for i in modelled_repeats
            ]
        )
        real_repeats = timing["real"]["by_repeat"]
        real_startups = _by_repeat(costs, "startup_by_repeat", "real")
        real_total = (
            _quantile(
                [
                    real_repeats[i]
                    + real_startups.get(i, startup_real)
                    + _for_repeat(costs, "load", i)
                    for i in real_repeats
                ]
            )
            if real_repeats
            else execution_real + startup_real + _uncontained(costs, "load")
        )
    else:
        predict_once = execution_modelled + derivation_added
        derivation_added_reported = derivation_added
        # A supplied term that happens inside a measured window is already in
        # that window's seconds. `load` is inside `startup_real` by the
        # protocol's own definition (§5, "weight load and graph capture inside
        # that startup"), so adding it here charged the real side twice; the
        # same holds for any term that declares a container.
        real_total = execution_real + startup_real + _uncontained(costs, "load")
        modelled_total = (
            execution_modelled + _uncontained(costs, "derivation") + startup_modelled
        )
    replay_ratio = execution_real / predict_once if predict_once > 0 else None
    per_cell = acquisition / max(1, reuse_cells)
    amortised_total = modelled_total + per_cell
    saved_per_cell = real_total - modelled_total
    return {
        "gate": (
            f"execution_real / (execution_modelled + the derivation that "
            f"window does not already contain), both one repeat's seconds, "
            f">= {SPEEDUP_MIN}"
        ),
        "replay_ratio": replay_ratio,
        "repeats": sorted(modelled_repeats) or None,
        "derivation_included_s": derivation,
        # The same seconds, split by who already counted them, so a reader can
        # check that the denominator holds every derivation exactly once.
        "derivation_added_to_gate_s": derivation_added,
        "derivation_inside_execution_s": derivation_in_execution,
        # What one question's derivation cost, which is what the gate divides
        # by -- the whole term above is every question this cell asked.
        "derivation_per_repeat_s": derivation_added_reported,
        "predict_once_s": predict_once,
        "acquisition_s": acquisition,
        "acquisition_terms": {
            "capture": _supplied_seconds(costs, "capture"),
            "calibration": _supplied_seconds(costs, "calibration"),
        },
        "startup_inclusive_ratio": (
            real_total / modelled_total if modelled_total > 0 else None
        ),
        "amortised_ratio": (
            real_total / amortised_total if amortised_total > 0 else None
        ),
        "amortised_over_cells": reuse_cells,
        "break_even_cells": (
            math.ceil(acquisition / saved_per_cell)
            if saved_per_cell > 0 and acquisition > 0
            else (0 if acquisition == 0 else None)
        ),
        "meets_gate": (
            replay_ratio >= SPEEDUP_MIN if replay_ratio is not None else None
        ),
    }


# --------------------------------------------------------------------------
# the matrix


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so tied values do not get an arbitrary order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]):
    """Rank correlation, tie-aware, with no dependency to install."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = math.sqrt(
        sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)
    )
    return (num / den) if den else None


#: Objective, and whether more of it is better. Configuration choice is what
#: the tool is for, so each is ranked on its own.
OBJECTIVES = {"throughput_tok_s": "max", "ttft": "min", "tpot": "min"}


def _spread(block: dict, side: str):
    """One side's range on a metric, from the range or from the repeats.

    Returns None when the verdict carries neither, which is not the same as a
    zero-width range: a side whose spread was never recorded cannot be said to
    have separated two configurations or to have tied them.
    """
    stated = block.get(f"{side}_range")
    if (
        isinstance(stated, (list, tuple))
        and len(stated) == 2
        and all(_finite(v) for v in stated)
    ):
        return (float(min(stated)), float(max(stated)))
    repeats = [v for v in (block.get(side) or []) if _finite(v)]
    if repeats:
        return (float(min(repeats)), float(max(repeats)))
    return None


def _overlap(left, right) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _fidelity(named: list) -> list:
    """Where the model disagrees with the hardware about a tie, pair by pair.

    `CC_TRACES_PROTOCOL.md` §6: two configurations are tied on a metric when
    the repeats overlap on it, and a model that *invents* a separation the
    hardware does not show fails, as does one that *flattens* a separation the
    hardware does show. Both are read the same way on both sides -- overlap of
    the measured spreads -- because the question is whether the model claims a
    difference the hardware has, not whether its centres happen to sort the
    same way. Top-1 agreement does not cover this: a model can pick the same
    winner while claiming a gap between two configurations that are the same
    machine within its own noise.
    """
    reasons = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            (left, lm), (right, rm) = named[i], named[j]
            pair = f"{left['cell']} vs {right['cell']}"
            spreads = {
                side: (_spread(lm, side), _spread(rm, side))
                for side in ("real", "modelled")
            }
            missing = [
                side for side, (a, b) in spreads.items() if a is None or b is None
            ]
            if missing:
                reasons.append(
                    f"{pair}: no {', '.join(missing)} spread is recorded, so "
                    f"whether these two configurations differ is unmeasured"
                )
                continue
            real_tied = _overlap(*spreads["real"])
            modelled_tied = _overlap(*spreads["modelled"])
            if real_tied and not modelled_tied:
                reasons.append(
                    f"{pair}: the model invents a separation the hardware does "
                    f"not show -- the real repeats overlap and the modelled "
                    f"ones do not"
                )
            elif modelled_tied and not real_tied:
                reasons.append(
                    f"{pair}: the model flattens a separation the hardware does "
                    f"show -- the real repeats are apart and the modelled ones "
                    f"overlap"
                )
    return reasons


def _decide(cells: list[dict], metric: str, direction: str) -> dict:
    """Top-1 agreement, ties as the hardware shows them, and regret.

    **Regret is a loss and is positive when the model chose badly**, whichever
    way the metric points: the real measurement of the configuration the model
    picked, against the real measurement of the best one, as a percentage of
    the best. Picking a configuration that turns out 13 % slower and one whose
    TTFT turns out 13 % higher both read `+13.3`. Zero is agreement. One
    convention for every objective, because a signed-by-direction regret is
    read wrongly the first time it is read.
    """
    named = [
        (c, c["metrics"][metric]) for c in cells if metric in (c.get("metrics") or {})
    ]
    if len(named) < 2:
        return {
            "comparable_cells": len(named),
            "reason": "fewer than two cells carry this metric",
        }
    real = [(c, m["real_centre"], m["real_range"]) for c, m in named]
    best = (
        max(real, key=lambda t: t[1])
        if direction == "max"
        else min(real, key=lambda t: t[1])
    )
    # Tied with the best when the repeats overlap: the spread across a cell's
    # own repeats bounds what a difference between cells can mean.
    tied = [t for t in real if t[2][0] <= best[2][1] and best[2][0] <= t[2][1]]
    chosen = (
        max(named, key=lambda t: t[1]["modelled_centre"])
        if direction == "max"
        else min(named, key=lambda t: t[1]["modelled_centre"])
    )
    chosen_real = next(t for t in real if t[0] is chosen[0])
    regret = ((chosen_real[1] - best[1]) / best[1] * 100.0) if best[1] else None
    if direction == "max" and regret is not None:
        regret = -regret
    rho = spearman(
        [m["real_centre"] for _, m in named], [m["modelled_centre"] for _, m in named]
    )
    separation_failures = _fidelity(named)
    return {
        "comparable_cells": len(named),
        "real_best": best[0]["cell"],
        "real_best_value": best[1],
        "real_tied_with_best": [t[0]["cell"] for t in tied],
        "modelled_top1": chosen[0]["cell"],
        "top1_agrees": chosen[0]["cell"] in [t[0]["cell"] for t in tied],
        "regret_pct": regret,
        "spearman_rho": rho,
        "rho_meets_gate": (rho is not None and rho >= RHO_MIN),
        "separation_failures": separation_failures,
        "separation_faithful": not separation_failures,
        # Three readings, one pass. `_across_repeats` writes None when the
        # error could not be computed -- a zero real centre, or a metric with
        # no registered tolerance -- and a cell nobody could grade is not a
        # cell inside tolerance. The ungraded ones are named separately so a
        # failure says which of the two it is.
        "within_tolerance": all(m.get("within_tolerance") is True for _, m in named),
        "tolerance_undecided": [
            c["cell"] for c, m in named if m.get("within_tolerance") is None
        ],
    }


#: The cells `CC_TRACES_PROTOCOL.md` §3 registers. The decision the protocol
#: is for is which width to deploy at, per workload class, so the matrix is
#: the unit of acceptance and a subset of it is not a smaller version of the
#: same claim -- it is a different one.
REGISTERED_CELLS = tuple((tp, klass) for tp in (1, 2, 4) for klass in sorted(WORKLOADS))


def _cell_key(verdict: dict, where: str, refused: list):
    """This verdict's place in the registered matrix, or None with a reason."""
    klass, tp = verdict.get("class"), verdict.get("tp")
    try:
        tp = int(tp)
    except (TypeError, ValueError):
        tp = None
    if (tp, klass) not in REGISTERED_CELLS:
        refused.append(
            f"{where}: this verdict is for tp={verdict.get('tp')!r} "
            f"class={klass!r}, which is not one of the six cells the protocol "
            f"registers ({', '.join(f'tp{t} {k}' for t, k in REGISTERED_CELLS)})"
        )
        return None
    return (tp, klass)


def matrix(args) -> int:
    cells, refused = [], []
    seen = {}
    for where in args.dirs:
        path = Path(where) / "cc_traces_cell.json"
        if not path.exists():
            refused.append(f"{where}: no cc_traces_cell.json; run `cell` first")
            continue
        verdict = json.loads(path.read_text())
        # A verdict is a file, and a file can be written by hand or copied out
        # of a diagnostic. `cell` refuses a diagnostic outright, so a passing
        # verdict beside a marker did not come from this validator.
        if (Path(where) / DIAGNOSTIC_MARKER).exists():
            refused.append(
                f"{where}: {DIAGNOSTIC_MARKER} is present, so this directory "
                f"is a diagnostic and cannot contribute to a decision"
            )
            continue
        if _purpose_of(verdict) != ACCEPTANCE_PURPOSE:
            refused.append(
                f"{where}: this verdict was written for "
                f"{_purpose_of(verdict)!r}, not acceptance"
            )
            continue
        if not verdict.get("passed"):
            refused.append(
                f"{where}: the cell did not pass "
                f"({len(verdict.get('failures') or [])} failures)"
            )
            continue
        key = _cell_key(verdict, where, refused)
        if key is None:
            continue
        # Two directories for one configuration is not two measurements of the
        # matrix; it is one cell counted twice while another is absent, and the
        # count alone cannot tell the difference.
        if key in seen:
            refused.append(
                f"{where}: tp{key[0]} {key[1]} is already contributed by "
                f"{seen[key]}, so this cell would be counted twice"
            )
            continue
        seen[key] = where
        cells.append(verdict)
    # Every objective, for every cell. A cell that carries no `ttft` is a cell
    # nobody ranked on `ttft`, and skipping the comparison reports a gate that
    # was never evaluated as one that passed.
    for verdict in cells:
        absent = [m for m in OBJECTIVES if m not in (verdict.get("metrics") or {})]
        if absent:
            refused.append(
                f"{verdict['cell']}: no {', '.join(sorted(absent))} in the "
                f"verdict, so this cell cannot be ranked on "
                f"{'them' if len(absent) > 1 else 'it'}"
            )
    for tp, klass in REGISTERED_CELLS:
        if (tp, klass) not in seen:
            refused.append(
                f"tp{tp} {klass}: no cell was contributed for it. A ranking "
                f"over part of the matrix is not the ranking the protocol "
                f"registers"
            )
    report = {
        "cells_used": [c["cell"] for c in cells],
        "refused": refused,
        "by_class": {},
        "pooled": {},
    }
    if refused:
        for reason in refused:
            print(f"  REFUSED: {reason}")
        print(
            "a ranking over an incomplete matrix is not a ranking; "
            "no decision gate is reported"
        )
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        return 1
    for klass in sorted({c["class"] for c in cells}):
        group = [c for c in cells if c["class"] == klass]
        report["by_class"][klass] = {
            metric: _decide(group, metric, direction)
            for metric, direction in OBJECTIVES.items()
        }
    report["pooled"] = {
        metric: _decide(cells, metric, direction)
        for metric, direction in OBJECTIVES.items()
    }
    report["speedup"] = {c["cell"]: c.get("speedup") for c in cells}
    for klass, block in report["by_class"].items():
        for metric, result in block.items():
            print(
                f"{klass:5s} {metric:18s} top1="
                f"{result.get('top1_agrees')} rho={result.get('spearman_rho')} "
                f"regret={result.get('regret_pct')} "
                f"separation={result.get('separation_faithful')}"
            )
            for reason in result.get("separation_failures") or ():
                print(f"  SEPARATION: {klass} {metric}: {reason}")
            for cell_name in result.get("tolerance_undecided") or ():
                print(
                    f"  UNGRADED: {klass} {metric}: {cell_name} carries no "
                    f"within_tolerance reading, so it is not inside tolerance"
                )
    # Validity and acceptance are different questions and the report answers
    # both. Every cell above is a measurement this validator stands behind;
    # whether the measurements together are the result the protocol registers
    # is these gates, each reported by name so a failure says which claim
    # failed rather than only that one did.
    results = [r for block in report["by_class"].values() for r in block.values()]
    gates = {
        "ranking": all(
            r.get("top1_agrees") and r.get("rho_meets_gate") for r in results
        ),
        "separation": all(r.get("separation_faithful") for r in results),
        "tolerance": all(r.get("within_tolerance") for r in results),
        # §6 registers the replay speedup as a gate. A model that ranks
        # perfectly and answers no faster than the hardware has measured
        # something valid and is not the thing being accepted.
        "speedup": all(
            ((c.get("speedup") or {}).get("meets_gate") is True) for c in cells
        ),
        "decided": all(not r.get("reason") for r in results),
    }
    report["gates"] = gates
    report["accepted"] = all(gates.values())
    for name, held in gates.items():
        if not held:
            print(f"  GATE FAILED: {name}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("MATRIX " + ("PASS" if report["accepted"] else "FAIL"))
    return 0 if report["accepted"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("dir")
    c.add_argument("--class", required=True, choices=sorted(WORKLOADS))
    c.add_argument("--tp", type=int, required=True)
    c.add_argument("--calibration-registry", default=None)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument(
        "--reuse-cells",
        type=int,
        default=2,
        help="how many cells share this cell's capture and "
        "calibration, for the amortised speedup",
    )
    g = sub.add_parser(
        "gpu-free",
        help="record, from inside the modelled side's own container, that "
        "no device was reachable there",
    )
    g.add_argument("dir")
    m = sub.add_parser("matrix")
    m.add_argument("dirs", nargs="+")
    m.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return {"cell": cell, "gpu-free": gpu_free, "matrix": matrix}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
