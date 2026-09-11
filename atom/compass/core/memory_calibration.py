"""Per-model memory constants calibrated at a declared source configuration.

Some memory terms have no derivation worth the name. `persistent` is whatever
forward buffers the model's own modules allocate and hold; `load_residue` is
what the collective libraries register through the allocator. Neither falls out
of `config.json`, and the campaign's defaults for both were fitted on the 0.6B
and do not transfer -- against the 27B they read 51% and 93% low.

The alternative to guessing is to measure once, at a configuration declared as
the *source*, and carry the number with enough provenance that a reader can
tell what it is. That is what this module holds. The rules it enforces:

* **A calibrated constant names its origin.** Model, tensor-parallel width, the
  full config it was measured under, the record it came from, and which term it
  supplies. A number without that is not source-calibrated, it is just a number
  that happens to be right about something; `MODEL_HEADROOM` is the cautionary
  case, a 27B fit whose configuration nobody wrote down.
* **The source is a configuration, not a model.** Calibrating on the model under
  evaluation is fine. Calibrating at the configuration under evaluation is not:
  a constant fitted at TP=4 and then used to predict TP=4 predicts nothing.
* **A fit checked against its own run is a residual, not a validation.** The
  two are different claims and `classify` refuses to conflate them -- comparing
  a calibration against the run it came from reports `residual`, and only a run
  the fit never saw reports `validation`.
* **Identity is not integrity.** Which *execution* wrote a record is what
  separates a residual from an independent repeat, and a content hash cannot
  answer it: the three source runs here wrote byte-identical records, and
  re-serialising any one of them changes its hash without a second execution
  happening. `classify` takes a producer, `integrity` takes a hash, and a
  record whose producer is unidentified is classified as the weaker claim.

`non_torch` is here too, but on stricter terms than the others. It is
`(total - free) - reserved`, which is device-wide: it charges this
configuration for whatever the neighbours hold. The historical TP=1 record
reads 32 MiB above what the same configuration reads on an exclusively-held
device, so that record is not usable for this term and is not used. The value
here comes instead from three runs of the source configuration on a device
nobody else was on, with ownership sampled throughout. It is still the weakest
constant in the module: the term threw one unexplained 486 MiB excursion in ten
exclusive-device runs, and a single value cannot bound that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "CalibratedTerm",
    "SourceCalibration",
    "SourceRun",
    "for_model",
    "producer_key",
]

#: The fields that name one execution. Taken from the vocabulary the campaign
#: harness already writes -- `compare.py` reads a run manifest out of a record's
#: `run` block, `merge_sweep.py` labels fresh-process shards by `started_at`,
#: and `residual.py`/`step_accounting.py` attribute steps by `pid`. A second
#: run-ID scheme would have to be reconciled with those; this one is them.
PRODUCER_FIELDS = ("host", "pid", "started_at")


def producer_key(run: Mapping | None) -> str | None:
    """Which execution wrote a record, or None when the record cannot say.

    **A content hash is integrity, not identity.** It answers "are these the
    same bytes", and the two questions come apart in both directions. Three
    phase C repeats of the 27B's source configuration -- three processes, three
    device allocations, three teardowns -- wrote byte-identical records, so
    equal hashes there mean three executions and not one. Re-serialising one
    record with a different indent changes its hash without there having been a
    second execution at all. Neither case is exotic; the first is this module's
    own evidence.

    So identity comes from fields that name the execution, and all of them are
    required: a partial identity would collide across runs exactly where it
    matters, and a collision here silently upgrades a residual into a repeat.
    None means unidentified, and callers are expected to treat that as "cannot
    tell" rather than as "no match".
    """
    if not run:
        return None
    values = []
    for field_name in PRODUCER_FIELDS:
        value = run.get(field_name)
        if value in (None, ""):
            return None
        values.append(str(value))
    return "|".join(values)


@dataclass(frozen=True)
class SourceRun:
    """The configuration a constant was measured at, in full.

    Everything a reader needs to decide whether a later run is independent of
    this one. Stored rather than summarised: "TP=1" alone would not say whether
    a run at another utilization or another `max_num_seqs` counts as separate.
    """

    model: str
    tensor_parallel: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_model_len: int
    capture_sizes: tuple
    enable_prefix_caching: bool
    record: str
    record_sha256: str
    role: str = "source"
    #: Which *executions* the fit was made from, as `producer_key` spells them.
    #: Empty means the executions are unidentified, which is the common case and
    #: is why `classify` has to be conservative rather than clever.
    producers: tuple = ()
    #: Where those identities came from. A record that does not carry its own
    #: producer leaves only external witnesses, and a reader has to be told
    #: which it is looking at.
    producer_basis: str = ""

    def matches(self, config: Mapping) -> bool:
        """Whether `config` is this same configuration, not merely this model."""
        topology = config.get("topology") or {}
        return (
            config.get("model") == self.model
            and int(topology.get("tp", 1)) == self.tensor_parallel
            and float(config.get("gpu_memory_utilization", -1))
            == self.gpu_memory_utilization
            and int(config.get("max_num_seqs", -1)) == self.max_num_seqs
        )


@dataclass(frozen=True)
class CalibratedTerm:
    """One constant, its value, where it came from, and why it is not derived."""

    term: str
    value: int
    run: SourceRun
    basis: str
    validated_against: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceCalibration:
    """Every calibrated constant for one model, keyed by term."""

    model: str
    terms: Mapping[str, CalibratedTerm]

    def mapping(self, world_size: int) -> dict:
        """The calibration mapping `memory_model` already accepts.

        Only terms calibrated at this width are emitted. `load_residue` is a
        collective-library cost and is emphatically not flat in width -- 14 MiB
        at TP=1 against 2.1 GB at TP=2 -- so a TP=1 calibration says nothing
        about TP=2 and is not offered there.
        """
        out: dict = {}
        for name, term in self.terms.items():
            if name == "persistent":
                out["persistent"] = term.value
            elif term.run.tensor_parallel == world_size:
                out[name] = {world_size: term.value}
        return out

    def classify(
        self,
        term: str,
        config: Mapping,
        *,
        record_sha256: str | None = None,
        producer: Mapping | None = None,
    ) -> str:
        """What a comparison against this record is worth: three answers.

        The distinction the whole module exists for: a term that reproduces the
        run it was fitted on has demonstrated arithmetic, and reporting that as
        agreement is reporting the fit back to itself.

        * `residual` -- this is the record the term was fitted on.
        * `repeat` -- a different run of the same configuration. Reproducibility
          evidence, and worth having, but it holds fixed everything the
          constant was fitted against, so it cannot show the constant transfers.
        * `validation` -- a run at a configuration the fit never saw.

        Residual and repeat differ by *execution*, so only `producer` can
        separate them. The first cut used the record's hash, which cannot: this
        calibration's own source record is byte-identical across three separate
        runs, so a hash match there is three executions, and a re-serialised
        copy of one run mismatches without a second execution existing. See
        `producer_key`.

        Unidentified producer is answered `residual` -- the weaker claim. An
        unknown run reported as a repeat would credit the constant with
        reproducibility nobody observed; reported as a residual it credits the
        constant with nothing at all, which is what is actually known. Pass
        `record_sha256` to `integrity` instead, where bytes are the question.
        """
        entry = self.terms.get(term)
        if entry is None:
            return "underived"
        if not entry.run.matches(config):
            return "validation"
        key = producer_key(producer)
        if key and entry.run.producers:
            return "residual" if key in entry.run.producers else "repeat"
        return "residual"

    def identifies(self, term: str, producer: Mapping | None) -> bool:
        """Whether this producer names the execution the term was read off.

        The precondition for `integrity` being worth *reporting*. A hash
        mismatch on a record nobody claimed was the fitted artifact only means
        "a different record", which a row saying `validation` already says;
        it is when the producer says *this is that run* that a mismatch means
        the bytes moved under it.
        """
        entry = self.terms.get(term)
        if entry is None:
            return False
        key = producer_key(producer)
        return bool(key) and key in entry.run.producers

    def integrity(self, term: str, record_sha256: str | None) -> str:
        """Whether a record is the *bytes* the term was fitted on.

        Separate from `classify` because it answers a separate question, and
        conflating the two is what made a repeat look like a residual. `altered`
        does not mean a different run -- re-serialising a record changes its
        hash and nothing else -- it means the comparison is no longer against
        the artifact the constant was read off.
        """
        entry = self.terms.get(term)
        if entry is None or not record_sha256:
            return "unknown"
        return "intact" if record_sha256 == entry.run.record_sha256 else "altered"


#: The 27B's TP=1 source run: `max_num_seqs 32`, the cc-traces capture ladder,
#: prefix caching off, utilization 0.90. Utilization is part of the identity
#: even though both terms below are invariant in it -- which is itself a
#: measurement, not an assumption: `current_torch - weights_torch` is
#: 252 339 712 B at utilizations 0.33, 0.40, 0.41 and 0.90 alike, to the byte.
_QWEN27B_TP1 = SourceRun(
    model="Qwen/Qwen3.8-27B",
    tensor_parallel=1,
    gpu_memory_utilization=0.9,
    max_num_seqs=32,
    max_model_len=262144,
    capture_sizes=(1, 2, 4, 8, 16, 32),
    enable_prefix_caching=False,
    record="tests/compass/memory_records/27b.tp1.memory.json",
    record_sha256="6233290081fde7197b221f489145dc74997e2788e19bc8f9c1cada7212d2e17e",
    producer_basis=(
        "Unidentified. The record predates any ownership sampling and carries "
        "no run block, so which execution wrote it is not recoverable -- and "
        "its `non_torch` is 32 MiB above the exclusive run's, which is the one "
        "term that would most want to name its neighbours."
    ),
)

#: The same configuration again, on a device nobody else was on: three runs,
#: byte-identical records, with ownership sampled every 30 s throughout. This
#: is the run `non_torch` may be taken from, and the historical record is not.
_QWEN27B_TP1_EXCLUSIVE = SourceRun(
    model="Qwen/Qwen3.8-27B",
    tensor_parallel=1,
    gpu_memory_utilization=0.9,
    max_num_seqs=32,
    max_model_len=262144,
    capture_sizes=(1, 2, 4, 8, 16, 32),
    enable_prefix_caching=False,
    record="tests/compass/memory_records/27b.tp1.exclusive.memory.json",
    record_sha256="4ff602796143df38166131f12ed4c1b728cc051fbb9c78be3bfd4540f417258e",
    producer_basis=(
        "Three executions, witnessed from outside the record: the ownership "
        "sampler saw three disjoint process cohorts on the device -- engine "
        "pids 695009, 751439 and 775417, each with its own launcher pair -- "
        "across 08:55:09Z-09:03:34Z, and all three wrote byte-identical "
        "records. `producers` is left empty all the same: the records "
        "themselves carry no run block, so nothing a reader is handed can be "
        "matched against those identities, and `classify` answers `residual` "
        "for every one of them. Making that answer better needs the writer to "
        "record its own producer -- see MEMORY_EVIDENCE.md, O12."
    ),
)

_QWEN27B = SourceCalibration(
    model="Qwen/Qwen3.8-27B",
    terms={
        "persistent": CalibratedTerm(
            term="persistent",
            value=252339712,
            run=_QWEN27B_TP1,
            basis=(
                "Forward buffers the modules hold between weight load and the "
                "profile peak. No source expression for them exists yet; the "
                "0.6B default of 118 MiB is 51% low against this model."
            ),
            validated_against=(
                "TP=1 util 0.33, 0.40, 0.41 (exact, 0 B)",
                "TP=2 rank 0 (252 332 544 B, -0.003%)",
                "TP=4 ranks 0 and 1 (252 328 960 B, -0.004%)",
            ),
        ),
        "non_torch": CalibratedTerm(
            term="non_torch",
            value=1157627904,
            run=_QWEN27B_TP1_EXCLUSIVE,
            basis=(
                "(total - free) - reserved, so it is device-wide and only "
                "means anything measured on a device nobody else is on. Three "
                "runs of the source configuration produced byte-identical "
                "records; six further exclusive runs at other utilizations "
                "read the same value. The historical TP=1 record reads "
                "1 191 182 336 B, 32 MiB higher, and is not used."
            ),
            validated_against=(
                "TP=1 util 0.33 settled, 0.40, 0.41 (exact, 0 B)",
                (
                    "KNOWN RISK: one run in ten read 1 677 721 600 B, +42%, "
                    "cause unknown and not reproduced by three separate "
                    "controls. This constant does not bound that excursion."
                ),
            ),
        ),
        "load_residue": CalibratedTerm(
            term="load_residue",
            value=14924832,
            run=_QWEN27B_TP1,
            basis=(
                "What the loader leaves in the allocator above the parameters. "
                "At TP=1 there is no collective pool, so this is small and the "
                "0.6B default of 1 MiB is 93% low. Width-specific: the TP>=2 "
                "entries stay on the 0.6B table and are not calibrated here."
            ),
            validated_against=("TP=1 util 0.33, 0.40, 0.41 (exact, 0 B)",),
        ),
    },
)

_BY_MODEL = {_QWEN27B.model: _QWEN27B}


def for_model(model: str) -> SourceCalibration | None:
    """The source calibration for this model, or None if it has none.

    None is the honest answer for every model nobody has measured, and callers
    are expected to fall back to the campaign defaults rather than to a
    neighbouring model's constants.
    """
    return _BY_MODEL.get(model)
