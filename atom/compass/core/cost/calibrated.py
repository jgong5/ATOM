"""**empirical/fitted** -- a model-step oracle fitted to steps that were timed.

The shape of the cost function is chosen by hand and its coefficients come from
measurement. It is deliberately the simplest thing
that could reproduce a serving run, because the purpose of the first one is to
produce an error number — until something predicts time, every claim about
Compass is structural, and structural claims cannot be ranked by how much they
matter.

Prefill and decode are fitted separately. They are not two regimes of one
function: prefill is compute-bound in the number of new tokens, decode is
bandwidth-bound in the KV history it must read. Fitting them together produces a
model that is wrong about both.

Features, per step:

* prefill — new tokens, plus attention within each chunk and over each chunk's
  own history. Both attention terms are summed per request: a step prefilling
  two requests at once must not charge one's tokens against the other's history
* decode — batch size (one row of GEMM work each) and total context across the
  batch (the KV bytes that must be read)

Total context is summed rather than averaged deliberately. A decode batch mixing
short and long histories does not cost what its mean history suggests, and the
sum is the quantity the hardware actually moves.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from atom.compass.core.artifacts import resolve_rank_path
from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["CalibratedCostOracle"]


def _prefill_features(shape: StepShape) -> list[float]:
    """What a prefill step costs: its own tokens, and the history it reads.

    `tokens` and `tokens**2` alone describe a prefill that starts from nothing --
    linear GEMM work plus self-attention within the chunk. Chunked prefill does
    not start from nothing. A 262144-token prompt arrives as sixteen chunks of
    16384, and every one of them has the same `tokens`, so a model without a
    context term predicts the same cost for the first chunk and the sixteenth.

    Measured, they are not the same: a full 16384-token chunk costs 1922 ms at a
    context under 50k, 2387 ms between 50k and 100k, and 2992 ms between 100k
    and 200k. The model was fitted on samples spanning all of that and predicted
    one number for them, 23% under the mean, because the feature it needed was
    not there to fit.

    Attention is the chunk reading over what came before -- each of its queries
    reads its own request's history. Keeping that separate from attention
    *within* the chunk, because the two grow differently: the second is fixed
    once the chunk size is, and the first climbs with every chunk.

    **Both are summed per request, not taken on the batch totals.** A step can
    prefill two requests at once, and collapsing it into `tokens * history` with
    both sides summed cross-multiplies them -- request A's new tokens get
    charged against request B's history, which A never reads. The error is not
    small. On one real step, ``[1856 new on 67968 ctx] + [14528 new on 14528]``:

        charged   16384 * 66112 = 1.08e9
        attends    1856 * 66112 = 1.23e8      8.8x too much

    Across a 27B serving run the fourteen steps that prefilled two requests came
    out +27.28% while the ninety-two single-request ones came out -4.01%, and
    the two nearly cancelled into a +0.04% headline. Per request they are -0.81%
    and -3.35%. The step's error, with the unmodelled cold start held out, goes
    from +2.96% to -0.20% -- and that 3% was what closed two decode windows and
    put TTFT 90% high.

    This is the shortcut ``StepShape`` was written to prevent, taken anyway:
    "sequence lengths are kept per request rather than reduced to a scalar.
    Collapsing them is the standard shortcut in this field and it is the
    standard source of error." The lengths were there; this function summed them.

    Reduces to the previous features exactly for a single-request step, which is
    all a calibration sweep mostly contains and is why a sweep-fitted model
    looked healthy while serving runs did not.
    """
    tokens = float(shape.num_prefill_tokens)
    within = 0.0    # attention inside each chunk, quadratic in that chunk
    across = 0.0    # attention over each chunk's own history
    for scheduled, context in zip(shape.num_scheduled_tokens,
                                  shape.context_lens or ()):
        if scheduled <= 1:
            # A decode row riding along in a prefill step. Its attention is what
            # the decode model is for; charging it here would double-count.
            continue
        new = float(scheduled)
        within += new * new
        # The history is what this request's context holds *before* its own new
        # tokens, which the reading already counts.
        across += new * max(0.0, float(context) - new)
    return [1.0, tokens, within, across]


def _decode_features(shape: StepShape) -> list[float]:
    batch = float(shape.batch_size)
    context = float(sum(shape.context_lens)) if shape.context_lens else 0.0
    return [1.0, batch, context]


def _decode_bucket_features(shape: StepShape) -> list[float]:
    """Decode features within one CUDA-graph bucket: the KV read, and the
    padding read with it.

    Batch size is dropped because the bucket already carries it -- the replay
    runs the padded rung whatever the batch, so twelve sequences and sixteen
    perform the same work.

    Total context alone treats a batch as the sum of its histories, which makes
    ``[10k, 10k, 10k]`` and ``[1k, 1k, 28k]`` the same step. They are not, if
    attention reads a rectangle sized by the longest sequence: the second reads
    three times 28k where the first reads three times 10k. The difference is
    the padding, and it is the second feature here.

    Measured on a 0.6B serving run, this is most of a large error. Real decode
    batches run 2.8 to 3.9 times ragged -- longest history over mean -- and
    predictions from total context alone came out 21 to 23% low at rungs 8 and
    16. With the padding term they come out 7.5% and 13.3% low.

    It could not be fitted until the sweep had ragged batches to fit it from.
    Every calibration round ran sequences of one length, so the padding was zero
    in every sample and the coefficient had no variance to be identified by --
    a rank deficiency rather than a coverage gap, which is why widening the
    context range did nothing for it.
    """
    lengths = [float(v) for v in (shape.context_lens or ())]
    if not lengths:
        return [1.0, 0.0, 0.0]
    context = sum(lengths)
    # What a padded read would touch beyond the real histories. Zero for a
    # uniform batch, which is what makes it droppable where nothing varies.
    padding = len(lengths) * max(lengths) - context
    return [1.0, context, padding]


def _truthy(value) -> bool:
    """Options arrive from the command line as strings, so "off" must mean off."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "off", "false", "no", ""}


def _leading_warmup(rows: list[list[float]], targets: list[float],
                    coeffs: list[float], ratio: float) -> list[float]:
    """Seconds the sweep's own first steps cost above what their shape says.

    Triton autotunes per shape and the first forwards of a process pay for it.
    On the 27B sweep the first three prefill steps took 47.5 s, 19.5 s and 6.1 s
    at 8, 32 and 96 tokens; the same shapes later took about 0.11 s. That is not
    a function of the step, so ``_least_squares`` is right to reject all three --
    their relative residuals are 0.99.

    **This is measured and reported, not charged.** A real serving run pays a
    first-step cost too -- 6.87 s on the 27B, and a remarkably steady one, 6.75,
    6.66, 6.69 and 6.68 s across four repeats -- and it is tempting to treat the
    sweep's leading warmth as a prediction of it. It is not. The two are
    different quantities:

        sweep, first three prefill steps      72.80 s, at 8/32/96 tokens
        serving run, first prefill step        6.87 s, at 640 tokens

    Charging the first to the second was tried and it moves the run's prefill
    total from +0.10% to +31.03%, with the first step priced at 47.56 s against
    6.87 s measured. The reason is that they are different programs: a server
    does a profile run and captures graphs before its first *measured* step, so
    most of the process-level warmth is already spent by then, while the sweep
    meets it head-on. How much is left over depends on what the server did at
    startup, which is not in this table.

    So what is returned here exists to be said out loud in ``describe`` -- the
    model has no term for the first step of a run, and a reader comparing a
    simulated run against a real one should know that the simulated one starts
    ahead. Learning the residual warmth properly needs the first use of each
    shape class to survive the fit, which is a different piece of work.

    Only a *leading* run of steps is taken, and only while each costs at least
    ``ratio`` times what its shape predicts.
    """
    out: list[float] = []
    for features, measured in zip(rows, targets):
        predicted = sum(c * f for c, f in zip(coeffs, features))
        if predicted <= 0.0 or measured < ratio * predicted:
            break
        out.append(measured - predicted)
    return out


def _drop_constant_columns(rows: list[list[float]]) -> tuple[list[list[float]], list[int]]:
    """Remove features that never vary, and say which were kept.

    A column of identical values carries no information about its coefficient
    and makes the normal equations singular. That is not hypothetical here: the
    padding term is exactly zero for every uniform batch, so a rung calibrated
    without ragged samples has a dead column. Dropping it fits the model that
    the data can identify rather than refusing, or worse, returning a
    coefficient the samples never constrained.

    The intercept is kept whatever it looks like; it is meant to be constant.
    """
    keep = [0]
    for i in range(1, len(rows[0]) if rows else 0):
        values = {round(r[i], 12) for r in rows}
        if len(values) > 1:
            keep.append(i)
    if len(keep) == len(rows[0] if rows else []):
        return rows, keep
    return [[r[i] for i in keep] for r in rows], keep


def _least_squares(
    rows: list[list[float]], targets: list[float], outlier_sigmas: float = 4.0,
) -> tuple[Optional[list[float]], int]:
    """Least squares, resistant to one-off contamination.

    Refuses rather than extrapolates when there are fewer samples than
    coefficients. An underdetermined fit returns numbers that look like a model
    and predict nothing, which is the failure mode this project keeps meeting.

    Fits, then discards points whose residual is far outside the spread of the
    rest, then refits. This is not statistical hygiene for its own sake — the
    contamination is specific and expected. **Triton autotunes per shape, not
    once per process**, so a calibration sweep deliberately made of varied
    shapes pays a one-off benchmarking cost on many of its own samples. One such
    row sat at 0.13 s where its neighbour at a larger size took 0.036 s.

    Spread is measured by median absolute deviation rather than standard
    deviation, since the contaminating points would otherwise inflate the very
    quantity used to detect them.

    **Fitted on relative error, not absolute.** Ordinary least squares minimises
    seconds-squared, so a 250 ms sample counts sixty times a 32 ms one, and a
    prefill sweep spanning 64 to 16 000 tokens is decided almost entirely by its
    largest steps. What anyone reads off this model is a percentage, and the
    small end is where serving actually lives: widening a sweep for decode
    coverage dropped the share of prefill steps under 1024 tokens from 17% to 9%
    and moved the prediction at 512 tokens from -13% to -17% against an
    unchanged measurement, purely by adding large samples elsewhere. Dividing
    each row and its target by that target makes every sample worth the same
    fraction of itself, which is the quantity being reported.

    Returns the coefficients and how many points were dropped. The count is
    returned rather than logged and forgotten: a fit that quietly discarded half
    its evidence should not describe itself the same way as one that kept it.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep of torch
        return None, 0
    full_width = len(rows[0])
    # A feature with no variance in these samples cannot have a coefficient
    # identified for it, and leaves the normal equations singular. Dropped
    # here and returned as zero, so a rung calibrated without ragged batches
    # fits the model its evidence supports instead of failing or inventing one.
    rows, kept = _drop_constant_columns(rows)
    width = len(rows[0])
    if len(rows) < width:
        return None, 0

    def _expand(values):
        """Back to the caller's feature width, zero where nothing varied."""
        out = [0.0] * full_width
        for slot, column in enumerate(kept):
            out[column] = float(values[slot])
        return out

    a = np.asarray(rows, dtype=float)
    b = np.asarray(targets, dtype=float)

    # Scale each equation by 1/target so the residual being minimised is the
    # fractional one. A non-positive target carries no scale, so it keeps its
    # own: durations are positive and one that is not is not evidence.
    scale = np.where(b > 0.0, 1.0 / np.where(b > 0.0, b, 1.0), 1.0)
    aw = a * scale[:, None]
    bw = b * scale
    coeffs, *_ = np.linalg.lstsq(aw, bw, rcond=None)

    # Enough points left to still determine the fit after dropping some.
    if len(rows) < width + 2:
        return _expand(coeffs), 0

    # Residuals in the same relative units, so the outlier test is not itself
    # dominated by the largest samples.
    residuals = np.abs(bw - aw @ coeffs)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    if mad <= 0.0:
        return _expand(coeffs), 0

    keep = residuals <= np.median(residuals) + outlier_sigmas * mad
    dropped = int((~keep).sum())
    if not dropped or int(keep.sum()) < width + 1:
        return _expand(coeffs), 0

    refit, *_ = np.linalg.lstsq(aw[keep], bw[keep], rcond=None)
    return _expand(refit), dropped


class CalibratedCostOracle:
    """Predicts step duration from coefficients fitted to measured steps."""

    def __init__(self, table: str, floor_seconds: float = 1e-6,
                 rank_coords: Optional[dict] = None, warmup=True,
                 warmup_ratio: float = 3.0) -> None:
        """
        Args:
            table: Path to a JSONL file written by ``--compass-mode=measure``.
                Under parallelism each rank wrote its own, and ``rank_coords``
                selects between them.
            floor_seconds: Smallest duration ever returned. A fitted model can
                produce a negative prediction outside the range it saw, and a
                negative step duration would run the virtual clock backwards.
            rank_coords: This rank's coordinates, supplied by the runner when
                the run is parallel. Absent for a single-rank run, where no
                per-rank table was written in the first place.
            warmup: Measure the autotuning the sweep paid on its own first
                steps and report it in ``describe``. It is not charged to any
                prediction -- see ``_leading_warmup`` for why the sweep's
                warmth does not transfer to a serving run's. Pass
                ``warmup=off`` to skip the detection.
            warmup_ratio: How much more than its shape predicts a leading step
                must cost to be called warmth rather than noise. Three is well
                clear of both sides on the evidence here: the steps this is
                meant to catch run 50 to 400 times their prediction, and the
                first ordinary step after them runs about 1.3 times.
        """
        self.table, own = resolve_rank_path(table, rank_coords)
        if rank_coords and not own:
            logger.warning(
                "ATOMCompass WARNING: rank %s has no calibration table of its "
                "own; fitting to the shared %s. Ranks of a symmetric group time "
                "within a fraction of a percent of each other, so this is "
                "usually fine — but it is an assumption, not a measurement.",
                rank_coords, self.table,
            )

        if not os.path.exists(self.table):
            # Named here rather than left to `open`, because this raises inside
            # a worker process: what reaches the terminal is the engine
            # manager's summary of a worker that vanished, which names neither
            # Compass nor the file. The message has to carry its own context.
            tried = f" (also tried {resolve_rank_path(table, rank_coords)[0]})" \
                if rank_coords else ""
            raise FileNotFoundError(
                f"ATOMCompass: no calibration table at {table}{tried}. "
                f"Record one with --compass-mode=measure --compass-measure-out "
                f"before predicting from it."
            )
        self.floor_seconds = floor_seconds
        self._prefill: Optional[list[float]] = None
        self._decode: Optional[list[float]] = None
        # One small fit per CUDA-graph rung, which is what a replay actually is:
        # a fixed shape, plus the KV history it reads. Held out on 499 unseen
        # rows, this halves-and-halves-again the error of a single fit carrying
        # batch size as a linear term -- median 5.04% -> 0.93%, RMSE 0.42ms ->
        # 0.13ms. A shared slope across rungs does not work (8.09%): a replay at
        # rung 16 reads sixteen padded rows and one at rung 1 reads one, so the
        # cost per unit of history is not the same number.
        self._decode_by_bucket: dict[int, list[float]] = {}
        self._decode_bucket_n: dict[int, int] = {}
        self._decode_bucket_hull: dict[int, tuple[float, float]] = {}
        self._n_prefill = 0
        self._n_decode = 0
        self._fallback_prefill = 0.0
        self._fallback_decode = 0.0
        self._dropped_prefill = 0
        self._dropped_decode = 0
        # The range each feature was calibrated over, so a prediction can say
        # whether it is interpolating or extrapolating.
        self._hull: dict = {}
        self._warned: set = set()
        # Reported, never charged. The model has no term for the first step of
        # a run, and this is how it admits that rather than letting a reader
        # infer from a total that it does.
        self._warmup_enabled = _truthy(warmup)
        self._warmup_ratio = float(warmup_ratio)
        self._warmup: list[float] = []
        self._fit()

    def _fit(self) -> None:
        prefill_rows, prefill_targets = [], []
        decode_rows, decode_targets = [], []
        by_bucket: dict[int, tuple[list, list]] = {}
        with open(self.table, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                shape = StepShape(
                    num_scheduled_tokens=tuple(row["num_scheduled_tokens"]),
                    context_lens=tuple(row["context_lens"]),
                    num_prefill_tokens=row["num_prefill_tokens"],
                )
                if shape.is_prefill:
                    prefill_rows.append(_prefill_features(shape))
                    prefill_targets.append(row["seconds"])
                else:
                    decode_rows.append(_decode_features(shape))
                    decode_targets.append(row["seconds"])
                    bucket = row.get("capture_bucket")
                    if bucket:
                        by_bucket.setdefault(int(bucket), ([], []))
                        rows_, targets_ = by_bucket[int(bucket)]
                        rows_.append(_decode_bucket_features(shape))
                        targets_.append(row["seconds"])

        self._n_prefill, self._n_decode = len(prefill_targets), len(decode_targets)
        for name, rows in (("prefill", prefill_rows), ("decode", decode_rows)):
            if rows:
                # Column 0 is the intercept and carries no range.
                self._hull[name] = [
                    (min(r[i] for r in rows), max(r[i] for r in rows))
                    for i in range(1, len(rows[0]))
                ]
        # The mean is the fallback when a fit is refused: a poor predictor, but
        # one whose error is bounded by the spread of what was measured, rather
        # than an extrapolation that can be arbitrarily wrong.
        if prefill_targets:
            self._fallback_prefill = sum(prefill_targets) / len(prefill_targets)
            self._prefill, self._dropped_prefill = _least_squares(
                prefill_rows, prefill_targets)
            if self._prefill is not None and self._warmup_enabled:
                # After the fit, so "what its shape predicts" means the model
                # this table produced, not one contaminated by the very steps
                # being measured against it.
                self._warmup = _leading_warmup(
                    prefill_rows, prefill_targets, self._prefill,
                    self._warmup_ratio)
        if decode_targets:
            self._fallback_decode = sum(decode_targets) / len(decode_targets)
            self._decode, self._dropped_decode = _least_squares(
                decode_rows, decode_targets)

        # Per rung. Three samples is the floor for two coefficients plus a
        # residual worth calling one; below that the rung falls back rather than
        # fitting a line through almost nothing.
        for bucket, (rows_, targets_) in by_bucket.items():
            if len(targets_) < 3:
                continue
            coeffs, _dropped = _least_squares(rows_, targets_)
            if coeffs is None:
                continue
            self._decode_by_bucket[bucket] = coeffs
            self._decode_bucket_n[bucket] = len(targets_)
            self._decode_bucket_hull[bucket] = (
                min(r[1] for r in rows_), max(r[1] for r in rows_))

        if self._prefill is None and self._n_prefill:
            logger.warning(
                "ATOMCompass WARNING: %d prefill steps is too few to fit; using their "
                "mean. Predictions will not vary with prompt length.",
                self._n_prefill,
            )
        if self._decode is None and self._n_decode:
            logger.warning(
                "ATOMCompass WARNING: %d decode steps is too few to fit; using their "
                "mean. Predictions will not vary with batch size or context.",
                self._n_decode,
            )
        if not self._n_prefill and not self._n_decode:
            raise ValueError(f"no usable measurements in {self.table}")

    def estimate(self, shape: StepShape) -> StepCost:
        if shape.is_prefill:
            coeffs, features = self._prefill, _prefill_features(shape)
            fallback = self._fallback_prefill
        else:
            bucketed = self._decode_for_bucket(shape)
            if bucketed is not None:
                return bucketed
            coeffs, features = self._decode, _decode_features(shape)
            fallback = self._fallback_decode
        if coeffs is None and not (self._n_prefill if shape.is_prefill else self._n_decode):
            # Never invent a number for a kind of step that was never measured.
            # Returning zero here is what made TTFT come back as 0 ms against a
            # real 7.6 s -- a confident, precise, entirely fictional answer. An
            # oracle asked something outside its evidence should say so.
            kind = "prefill" if shape.is_prefill else "decode"
            raise ValueError(
                f"{type(self).__name__}: asked to cost a {kind} step, but the "
                f"table {self.table} contains no {kind} measurements. Measure a "
                "workload that exercises it rather than extrapolating into it."
            )
        if coeffs is None:
            return StepCost(seconds=max(fallback, self.floor_seconds))
        self._warn_if_extrapolating(shape, features)
        predicted = sum(c * f for c, f in zip(coeffs, features))
        return StepCost(seconds=max(predicted, self.floor_seconds))

    def _decode_for_bucket(self, shape: StepShape) -> Optional[StepCost]:
        """Cost this decode step from the fit for its own CUDA-graph rung.

        Returns None when there is no rung to use -- an eager step carries no
        bucket, and an older table recorded none -- leaving the caller on the
        single batch-linear fit, which is what this replaces but not what it
        removes: a run without CUDA graphs is still a run worth costing.

        A rung that was never measured is a different matter. It is not an
        extrapolation along an axis, it is a gap in the evidence, and the nearest
        rung is the least-bad answer available; the alternative is refusing
        mid-run and killing a serving process over one step. Said out loud, once
        per rung, because the number that comes back is not supported by
        anything measured at that width.
        """
        if not self._decode_by_bucket:
            return None
        bucket = shape.capture_bucket
        if bucket is None:
            return None

        coeffs = self._decode_by_bucket.get(bucket)
        if coeffs is None:
            nearest = min(self._decode_by_bucket, key=lambda b: abs(b - bucket))
            key = ("bucket", bucket)
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "ATOMCompass WARNING: costing a decode step at CUDA graph "
                    "bucket %d, which was never measured; using bucket %d "
                    "instead. Rungs are measured, not interpolated -- calibrate "
                    "over a workload that reaches this concurrency.",
                    bucket, nearest,
                )
            coeffs = self._decode_by_bucket[nearest]
            bucket = nearest

        features = _decode_bucket_features(shape)
        low, high = self._decode_bucket_hull.get(bucket, (None, None))
        if low is not None and not (low <= features[1] <= high):
            key = ("bucket-context", bucket, features[1] < low)
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "ATOMCompass WARNING: costing a decode step at bucket %d "
                    "whose total context is %.0f, outside the calibrated range "
                    "[%.0f, %.0f]. The prediction is an extrapolation.",
                    bucket, features[1], low, high,
                )
        predicted = sum(c * f for c, f in zip(coeffs, features))
        return StepCost(seconds=max(predicted, self.floor_seconds))

    def _warn_if_extrapolating(self, shape: StepShape, features: list[float]) -> None:
        """Say so when asked about a shape outside what was calibrated.

        A fitted model answers anything, confidently, including questions its
        evidence does not cover — and a linear extrapolation is at its worst
        exactly where the intercept starts to dominate. This has now caused the
        same error twice, in two different dimensions: a prefill model fitted to
        1753-16370 tokens asked about 520, and a decode model fitted to batch
        sizes 1-4 asked about 8. Neither said anything; both were simply wrong.

        Warned once per kind and direction, because a serving run asks this
        thousands of times and a warning repeated per step is a warning nobody
        reads.
        """
        kind = "prefill" if shape.is_prefill else "decode"
        bounds = self._hull.get(kind)
        if not bounds:
            return
        for index, (low, high) in enumerate(bounds):
            value = features[index + 1]
            if low <= value <= high:
                continue
            key = (kind, index, value < low)
            if key in self._warned:
                continue
            self._warned.add(key)
            logger.warning(
                "ATOMCompass WARNING: costing a %s step whose feature %d is %.0f, "
                "outside the calibrated range [%.0f, %.0f]. The prediction is "
                "an extrapolation; calibrate over a workload that brackets "
                "this one.",
                kind, index, value, low, high,
            )

    def describe(self) -> str:
        def part(name, coeffs, n, dropped):
            if coeffs is None:
                return f"{name}=mean of {n}"
            outliers = f", {dropped} dropped" if dropped else ""
            return f"{name}=fitted on {n} steps{outliers}"

        rungs = ""
        if self._decode_by_bucket:
            # Which rungs exist, and how thinly each is supported: a rung fitted
            # on four samples predicts with the same confidence as one fitted on
            # four hundred, and only this says which is which.
            rungs = ", decode buckets={" + ", ".join(
                f"{b}:{self._decode_bucket_n[b]}"
                for b in sorted(self._decode_by_bucket)
            ) + "}"

        warmup = ""
        if self._warmup:
            # Said out loud because nothing else will say it: the table's first
            # steps were autotuning, the fit dropped them, and no coefficient
            # replaces them. A simulated run therefore starts ahead of a real
            # one by however much that run's own first step costs, which is a
            # different number from this one and is not in this table.
            warmup = (f", leading warmup in table={len(self._warmup)} steps "
                      f"totalling {sum(self._warmup):.2f}s, not charged")
        elif not self._warmup_enabled:
            warmup = ", warmup detection off"

        return (
            "CalibratedCostOracle("
            + part("prefill", self._prefill, self._n_prefill, self._dropped_prefill)
            + ", "
            + part("decode", self._decode, self._n_decode, self._dropped_decode)
            + rungs
            + warmup
            + ")"
        )
