"""What the collective layer costs a rank, measured standalone and composed.

Two terms above TP=1 -- `non_torch` and `load_residue` -- had no admissible
basis: their built-in table entries were fitted on a 0.6B engine, and
`_at_width` would otherwise answer a width it has no entry for with the widest
one below it. This module replaces both with a *delta* that was measured with
no model in the picture, on the engine's own initialization path.

The composition is a delta, not an absolute, and the TP=1 standalone control is
what makes that honest:

    term(width) = engine_term_at_TP1 + (standalone(width) - standalone(TP=1))

The standalone control measured **zero** for both terms -- at `world_size == 1`
`GroupCoordinator` builds no device communicator at all, so there is nothing to
subtract and nothing to double-count. Composing at width 1 therefore returns
the TP=1 engine calibration unchanged, by construction rather than by luck. The
engine term keeps everything the engine pays for that the probe does not run
(context, driver, the model path); the delta adds only what appeared between a
bare primed context and the end of `init_dist_env`. Neither side was chosen by
looking at a target residual -- the arithmetic is fixed by which phase each
measurement brackets.

**Two CustomAllreduce instances per rank, measured.** Seven `GroupCoordinator`s
are constructed; two of them (`tp` and `ep`) have `world_size > 1` and each
builds a `CudaCommunicator` with an active `ca_comm`, a `pynccl_comm` and a
`qr_comm`. The earlier source read assumed one and came out ~52% low. The count
here is from the probe's constructor hooks and its census of the live objects,
not from doubling a number that looked half-sized.

**Point and range mean different things.** `non_torch` is a per-rank quantity
and the ranks do not agree: at TP=4 they span 128 MiB (RCCL opens a different
number of channel buffers per rank). The point value is the **rank maximum**,
because the failure mode is asymmetric -- under-reserving `non_torch` inflates
the KV cache and kills the run at steady state, while over-reserving costs
blocks. The spread is published alongside it rather than averaged away.

Every reading is conditional on the environment it was taken in (below). Under
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` or
`AITER_CUSTOM_AR_RAW_INPUT_POOL`, each instance's 1 GiB input pool moves out of
the torch allocator: 2 GiB per rank shifts from `load_residue` to `non_torch`
and both numbers below are wrong. That is why the conditions travel with the
values and are checked rather than documented.
"""

from typing import Mapping, Optional

__all__ = [
    "TOPOLOGY_CONDITIONS", "TOPOLOGY_DELTAS", "TOPOLOGY_PROVENANCE",
    "UnmeasuredWidth", "topology_delta", "compose_calibration",
]


class UnmeasuredWidth(LookupError):
    """A width with no standalone measurement behind it.

    Raised rather than carried forward from a narrower width: the whole reason
    this module exists is that a table silently answering width 4 with the
    width-1 entry is indistinguishable from a calibration.
    """


#: The environment the deltas were measured under. A reading is only comparable
#: to another under the same one, so a mismatch is refused, not warned about.
#: `None` means "must be unset".
TOPOLOGY_CONDITIONS = {
    "PYTORCH_HIP_ALLOC_CONF": None,
    "PYTORCH_CUDA_ALLOC_CONF": None,
    "AITER_CUSTOM_AR_RAW_INPUT_POOL": None,
    "AITER_CUSTOM_AR_MAX_SIZE": None,
    "AITER_CUSTOM_AR_MIN_SIZE": None,
    "AITER_QUICK_REDUCE_QUANTIZATION": None,
}

TOPOLOGY_PROVENANCE = (
    "T27/1: standalone topology probe, MI308X gfx942 node18 GPU0-3, "
    "2026-09-12. No model, no checkpoint, no KV cache, no target-engine "
    "reading. Engine path: aiter init_dist_env(..., backend='nccl', "
    "pp=1, dp=1, pcp=1, dcp=1) as model_runner._setup_device_and_distributed "
    "calls it. Bracketed phases: context_primed -> after_init_dist_env. "
    "non_torch per model_runner.py:1648 = max(used - reserved, 0)."
)

#: Per width: what `init_dist_env` added on top of a primed, empty context.
#:
#: `non_torch_by_rank` is indexed by rank within the width. `load_residue` is
#: the torch-allocator delta, identical on every rank at both measured widths
#: and equal to `ca_instances * (8 MiB rank_data + 1 GiB input pool)` -- flat in
#: width, as the source read said, but twice what it said.
TOPOLOGY_DELTAS = {
    1: {
        "non_torch_by_rank": {0: 0},
        "load_residue": 0,
        "reserved": 2097152,
        "ca_instances": 0,
        "note": "control: world_size == 1 builds no device communicator "
                "(parallel_state.py:570-573), so both deltas are zero and the "
                "composition returns the TP=1 engine calibration unchanged",
    },
    2: {
        "non_torch_by_rank": {0: 5676990464, 1: 5676990464},
        "load_residue": 2164260864,
        "reserved": 2170552320,
        "ca_instances": 2,
        "note": "both ranks agree exactly",
    },
    4: {
        "non_torch_by_rank": {0: 5928648704, 1: 5962203136,
                              2: 5928648704, 3: 5827985408},
        "load_residue": 2164260864,
        "reserved": 2170552320,
        "ca_instances": 2,
        "note": "the ranks span 134217728 B (128 MiB); the point value is the "
                "rank maximum and the spread is published, not averaged",
    },
}


def _delta(width: int):
    if width not in TOPOLOGY_DELTAS:
        raise UnmeasuredWidth(
            "no standalone topology measurement at TP=%d (measured: %s). The "
            "narrower widths are not evidence about this one -- run the probe "
            "at this width rather than carrying a value forward."
            % (width, ", ".join(str(w) for w in sorted(TOPOLOGY_DELTAS))))
    return TOPOLOGY_DELTAS[width]


def _check_conditions(env: Optional[Mapping]):
    if env is None:
        return
    bad = []
    for key, want in TOPOLOGY_CONDITIONS.items():
        got = env.get(key)
        got = None if got in (None, "") else str(got)
        if got != want:
            bad.append("%s=%r (measured under %r)" % (key, got, want))
    if bad:
        raise UnmeasuredWidth(
            "the topology deltas were measured under a different allocator/pool "
            "environment: %s. Under expandable segments or a raw input pool, "
            "2 GiB per rank moves from load_residue to non_torch and both "
            "terms below are wrong." % "; ".join(bad))


def topology_delta(width: int, *, rank: Optional[int] = None,
                   side: str = "point", env: Optional[Mapping] = None):
    """The measured collective-layer delta at this width.

    `side` is "point" (the rank maximum -- see the module docstring on why the
    maximum and not the mean), "min" or "max". `rank` overrides all of that
    with that rank's own reading, which is the right call when the caller is
    sizing one specific rank.
    """
    entry = _delta(int(width))
    _check_conditions(env)
    by_rank = entry["non_torch_by_rank"]
    if rank is not None:
        if int(rank) not in by_rank:
            raise UnmeasuredWidth(
                "no reading for rank %d at TP=%d (ranks measured: %s)"
                % (rank, width, sorted(by_rank)))
        non_torch = by_rank[int(rank)]
    elif side == "point" or side == "max":
        non_torch = max(by_rank.values())
    elif side == "min":
        non_torch = min(by_rank.values())
    else:
        raise ValueError("side must be 'point', 'min' or 'max', not %r" % side)
    return {
        "width": int(width),
        "rank": None if rank is None else int(rank),
        "side": "rank" if rank is not None else side,
        "non_torch": int(non_torch),
        "non_torch_range": [int(min(by_rank.values())),
                            int(max(by_rank.values()))],
        "load_residue": int(entry["load_residue"]),
        "ca_instances": int(entry["ca_instances"]),
        "provenance": TOPOLOGY_PROVENANCE,
        "conditions": dict(TOPOLOGY_CONDITIONS),
    }


def compose_calibration(base: Mapping, width: int, *,
                        rank: Optional[int] = None, side: str = "point",
                        env: Optional[Mapping] = None) -> dict:
    """A calibration for `derived_readings`, composed from TP=1 plus the delta.

    `base` is the TP=1 full-engine calibration (class S27): the source config's
    own `persistent`, `non_torch` and `load_residue`. This returns the same
    mapping with `non_torch` and `load_residue` re-keyed at `width`, each the
    TP=1 value plus the standalone delta, and with a provenance string per term
    that names both halves.

    The result is what `derived_readings` already consumes -- width-keyed
    tables and a provenance block -- so nothing downstream needs to learn about
    this module. `persistent` is passed through untouched: it is flat in width
    and this probe says nothing about it.
    """
    delta = topology_delta(width, rank=rank, side=side, env=env)
    width = int(width)

    def at_one(term):
        table = base.get(term)
        if isinstance(table, Mapping):
            keys = {int(k): v for k, v in table.items()}
            if 1 not in keys:
                raise UnmeasuredWidth(
                    "the base calibration has no TP=1 entry for %r, so there "
                    "is nothing to compose the delta onto" % term)
            return int(keys[1])
        if table is None:
            raise UnmeasuredWidth(
                "the base calibration carries no %r term" % term)
        return int(table)

    base_provenance = dict(base.get("provenance") or {})
    composed = dict(base)
    composed["non_torch"] = {width: at_one("non_torch") + delta["non_torch"]}
    composed["load_residue"] = {
        width: at_one("load_residue") + delta["load_residue"]}
    for term in ("non_torch", "load_residue"):
        composed_from = base_provenance.get(term) or "(unstated TP=1 term)"
        base_provenance[term] = (
            "%s at TP=1, plus the standalone topology delta at TP=%d "
            "(%s). %s" % (composed_from, width, delta["side"],
                          TOPOLOGY_PROVENANCE))
    composed["provenance"] = base_provenance
    composed["topology_delta"] = delta
    return composed
