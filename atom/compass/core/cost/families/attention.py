"""A price for an attention call at a (query, history) structure nobody ran.

The row families move in one component -- token rows -- so one curve and an
interpolation between measured row counts answers them. The attention families
do not. `unified_attention_with_output_base` at fixed operand shapes differs
13x on history alone, so its price is set by the joint structure of the batch:
which rows are new queries, how much history each of them reads, and which
native branch that combination takes. There is no single axis to interpolate
along, which is why `FamilyPriceLibrary` refuses these outright today.

This module is the model that replaces that refusal -- and only where the
refusal was a gap rather than a finding. Exact lookup still comes first; a
modelled price is what an *honest miss* falls through to.

What it is not
--------------
It does not fit whole-step timings, and it does not reuse the calibrated
oracle's paired coefficients: those were fitted against the target engine's
own step durations, which is the thing a source-only prediction may not touch.
Every number here comes from a source primitive measurement of one operator.

It also does not interpolate blindly. A structure whose regime has no
identifiable support is refused by name, and the refusal says which points
would close it.

Regimes
-------
A regime is a native branch, not a convenience grouping. `attention_mha.py`
dispatches prefill and decode through different kernels, and the cached-prefix
path reads KV that the cold path does not, so their costs are different
functions of the same structure rather than one function with a parameter.

  ``unified.prefill.cold``    new queries only, no cached prefix to read
  ``unified.prefill.cached``  queries over a prefix already in the KV cache
  ``unified.decode``          one query row per sequence, history in cache
  ``gdn.prefill``             chunked scan over fresh sequences
  ``gdn.decode``              conv update plus recurrence over fixed state

Observations from two regimes are never pooled, and a structure is priced only
from its own regime's fit.

Scope
-----
A per-layer wrapper is not one kernel. Which kernel runs depends on the KV
cache dtype and layout, on `sliding_window`, and on the backend the deployment
selected -- and none of those are recoverable from the operand dtypes, because
a BF16 query says nothing about whether the cached KV it reads is FP8.

So :data:`REQUIRED_SCOPE` has to be declared by the observations themselves.
Under `strict` -- the default, and what an acceptance run gets -- a fit whose
observations do not declare them is refused rather than assumed. `strict=False`
permits an undeclared fit for diagnostic use and stamps `scope_undeclared` on
the result, so a number produced that way can never be mistaken for one that
was scoped.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["REQUIRED_SCOPE", "UNIFIED", "GDN", "Structure", "Regime",
           "Refusal", "Fit", "Model", "structure_of", "regime_of",
           "features_for", "fit_regime", "CHUNK_SIZE"]

UNIFIED = "aiter::unified_attention_with_output_base"
GDN = "aiter::linear_attention_with_output_base"

#: The native chunk width the GDN scan is written against. Declared here so a
#: chunk-count feature can be stated; it is not a fitted quantity, and a
#: deployment that changes it invalidates the chunk term rather than rescaling
#: it.
CHUNK_SIZE = 64

#: Static deployment facts that decide *which kernel* an attention call takes.
#: Two observations that disagree on any of these are measurements of
#: different work, and one that declares none of them cannot be shown to be in
#: any regime at all.
#:
#: `kv_cache_dtype` is resolvable today from the collector's own record of the
#: pool it stood up. The rest are not: an allocation geometry proves the
#: storage, not the view the backend takes over it, so layout, sliding window
#: and backend selection still have to be declared by whoever resolves them.
REQUIRED_SCOPE = ("kv_cache_dtype", "kv_cache_layout", "sliding_window",
                  "attention_backend")

#: Bytes of KV one history row costs, per MHA layer, at the measured
#: deployment: heads 4 x head_dim 256 x 2 bytes (BF16) x K and V = 4 KiB.
#:
#: Stated because it is easy to get wrong by an order of magnitude and the
#: wrong number has been written down before. The pool record is a *block* of
#: 16 rows -- 16 x 4 x 256 x 2 x 2 = 64 KiB per MHA layer -- and the model has
#: 16 MHA layers, not 64: the other 48 bound modules are GDN and hold no KV.
#: Dividing the pool across all 64 gives 16 KiB and is the error to avoid.
#:
#: It is deliberately NOT a separate feature. Per-layer bytes are this
#: constant times `history_rows`, so a byte column would be exactly collinear
#: with the row column and the fit would refuse it. `history_rows` *is* the
#: gather term; this is the scale a reader needs to interpret its coefficient.
KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER = 4 * 256 * 2 * 2
MHA_LAYERS = 16


class Refusal:
    """Why no price was produced. Carries the reason, never a number."""

    __slots__ = ("reason", "missing")

    def __init__(self, reason: str, missing: tuple = ()) -> None:
        self.reason = reason
        self.missing = tuple(missing)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Refusal({self.reason!r})"


class Structure:
    """The ragged shape of one attention call, as the key records it.

    ``queries`` and ``histories`` are per request and stay per request. The
    product of their sums is a different quantity from the sum of their
    products, and only the second is work: one long query over a short history
    and one short query over a long history do not cost what their totals
    suggest.
    """

    __slots__ = ("queries", "histories", "is_prefill", "has_cached", "state",
                 "bucket", "num_prefills", "num_decodes", "num_actual_tokens")

    def __init__(self, queries=(), histories=(), *, is_prefill=None,
                 has_cached=None, state=None, bucket=None,
                 num_prefills=None, num_decodes=None, num_actual_tokens=None):
        self.queries = tuple(int(q) for q in queries)
        self.histories = tuple(int(h) for h in histories)
        self.is_prefill = is_prefill
        self.has_cached = has_cached
        self.state = state
        self.bucket = bucket
        self.num_prefills = num_prefills
        self.num_decodes = num_decodes
        self.num_actual_tokens = num_actual_tokens

    @property
    def sequences(self) -> int:
        return len(self.queries) or int(self.num_decodes or 0)

    @property
    def query_total(self) -> int:
        return sum(self.queries) if self.queries else int(
            self.num_actual_tokens or 0)

    @property
    def history_total(self) -> int:
        return sum(self.histories)

    def paired_work(self) -> int:
        """``sum_i [ q_i*h_i + q_i(q_i+1)/2 ]`` -- the attended pairs.

        Per request, deliberately. The first term is every new query row
        reading every cached history row; the second is the causal triangle
        among the new rows themselves.
        """
        total = 0
        for q, h in zip(self.queries, self.histories):
            total += q * h + q * (q + 1) // 2
        return total

    def chunks(self) -> int:
        """Per-sequence ceil(q/CHUNK_SIZE), summed. Not ceil of the total."""
        return sum(-(-q // CHUNK_SIZE) for q in self.queries)


def _context(op: dict) -> dict:
    return {k: v for k, v in (tuple(x) for x in op.get("context") or ())}


def structure_of(op: dict) -> Optional[Structure]:
    """The ragged structure an operator records, or None if it records none."""
    ctx = _context(op)
    cu = ctx.get("cu_seqlens_q")
    context = ctx.get("context_lens")
    queries: tuple = ()
    if isinstance(cu, (list, tuple)) and len(cu) > 1:
        queries = tuple(int(cu[i + 1]) - int(cu[i]) for i in range(len(cu) - 1))
    histories: tuple = ()
    if isinstance(context, (list, tuple)) and queries:
        histories = tuple(int(c) - q for c, q in zip(context, queries))
    elif isinstance(context, (list, tuple)):
        histories = tuple(int(c) for c in context)
    execution = ctx.get("capture_bucket")
    return Structure(
        queries, histories,
        is_prefill=ctx.get("is_prefill"),
        has_cached=ctx.get("has_cached"),
        state=ctx.get("state"),
        bucket=execution,
        num_prefills=ctx.get("num_prefills"),
        num_decodes=ctx.get("num_decodes"),
        num_actual_tokens=ctx.get("num_actual_tokens"),
    )


class Regime:
    """One native branch, with the features its cost is a function of."""

    __slots__ = ("name", "features")

    def __init__(self, name: str, features: tuple) -> None:
        self.name = name
        self.features = tuple(features)

    def __eq__(self, other):
        return isinstance(other, Regime) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Regime({self.name!r})"


#: Features per regime. Kept minimal and checked for collinearity at fit time
#: rather than assumed independent: `tokens` and `chunks` coincide exactly when
#: every query is a multiple of CHUNK_SIZE, which is true of every GDN prefill
#: measured so far.
REGIMES = {
    "unified.prefill.cold": Regime("unified.prefill.cold",
                                   ("paired_work", "query_rows")),
    "unified.prefill.cached": Regime("unified.prefill.cached",
                                     ("paired_work", "query_rows",
                                      "history_rows")),
    "unified.decode": Regime("unified.decode",
                             ("context_rows", "active", "bucket_pad")),
    # No history term. The native conv update and recurrence consume a fixed
    # state, so a GDN decode does not read the full history and a term for it
    # would be a coefficient fitted to noise.
    "gdn.decode": Regime("gdn.decode", ("active", "bucket_pad")),
    "gdn.prefill": Regime("gdn.prefill", ("query_rows", "chunks", "sequences")),
}


def regime_of(op: dict, structure: Optional[Structure] = None):
    """Which native branch this call takes, or a `Refusal` naming the gap."""
    name = op.get("name", "")
    structure = structure_of(op) if structure is None else structure
    if structure is None:
        return Refusal("the operator records no ragged structure")
    if name == UNIFIED:
        if structure.is_prefill is None:
            return Refusal(
                "the key does not say whether this is a prefill, and the "
                "prefill and decode kernels are different work")
        if not structure.is_prefill:
            return REGIMES["unified.decode"]
        if structure.has_cached is None:
            return Refusal(
                "the key does not say whether a cached prefix was read, and "
                "the cold and cached prefill paths read different KV")
        return REGIMES["unified.prefill.cached" if structure.has_cached
                       else "unified.prefill.cold"]
    if name == GDN:
        prefills = structure.num_prefills
        decodes = structure.num_decodes
        if prefills is None or decodes is None:
            return Refusal(
                "the key does not carry the prefill/decode split, which is "
                "the branch the linear-attention kernel takes")
        if int(prefills) and int(decodes):
            return Refusal(
                "this call mixes fresh prefill sequences with continued "
                "decode ones; the two run different kernels in one call and "
                "no measurement separates their share")
        return REGIMES["gdn.prefill" if int(prefills) else "gdn.decode"]
    return Refusal(f"{name} is not an attention family this models")


def features_for(regime: Regime, structure: Structure):
    """The feature vector for one call, or a `Refusal` for what it lacks."""
    values = []
    for feature in regime.features:
        if feature == "paired_work":
            if len(structure.queries) != len(structure.histories):
                return Refusal("queries and histories are not paired per "
                               "request, so the attended pairs are unknown")
            values.append(float(structure.paired_work()))
        elif feature == "query_rows":
            values.append(float(structure.query_total))
        elif feature == "history_rows":
            values.append(float(structure.history_total))
        elif feature == "context_rows":
            values.append(float(structure.history_total + structure.query_total))
        elif feature == "active":
            values.append(float(structure.sequences))
        elif feature == "sequences":
            values.append(float(len(structure.queries)))
        elif feature == "chunks":
            values.append(float(structure.chunks()))
        elif feature == "bucket_pad":
            # Derived from the recorded bucket, never guessed. A call whose
            # bucket nobody recorded has an unknown amount of padded work, and
            # calling it zero would be inventing the answer.
            if structure.bucket is None:
                return Refusal(
                    "the replay bucket this call ran at was not recorded, so "
                    "how many padded rows the kernel executed is unknown; "
                    "that is a question for the padding owner, not a zero",
                    missing=("capture_bucket",))
            values.append(float(max(int(structure.bucket)
                                    - structure.sequences, 0)))
        else:  # pragma: no cover - guarded by REGIMES
            return Refusal(f"unknown feature {feature!r}")
    return values


# -- fitting ---------------------------------------------------------------


def _solve(matrix, rhs):
    """Least squares by normal equations, with a rank check. Pure stdlib.

    Small systems -- a handful of features -- so the normal equations are
    adequate and keeping this dependency-free matters more: this module is
    imported wherever a price is looked up.
    """
    n = len(matrix[0])
    ata = [[sum(row[i] * row[j] for row in matrix) for j in range(n)]
           for i in range(n)]
    atb = [sum(row[i] * y for row, y in zip(matrix, rhs)) for i in range(n)]
    # Gaussian elimination with partial pivoting.
    aug = [ata[i][:] + [atb[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None  # rank deficient
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


class Fit:
    """A regime's law, and everything a reader needs to distrust it."""

    __slots__ = ("regime", "coefficients", "scales", "points", "residual_df",
                 "relative_error", "domain", "scope", "scope_undeclared")

    def __init__(self, regime, coefficients, scales, points, residual_df,
                 relative_error, domain, scope, scope_undeclared=()):
        self.regime = regime
        self.coefficients = tuple(coefficients)
        self.scales = tuple(scales)
        self.points = points
        self.residual_df = residual_df
        self.relative_error = relative_error
        self.domain = domain
        self.scope = scope
        self.scope_undeclared = tuple(scope_undeclared)

    def predict(self, values):
        total = 0.0
        for value, coefficient, scale in zip(values, self.coefficients,
                                             self.scales):
            total += coefficient * (value / scale if scale else 0.0)
        return total

    def describe(self) -> str:
        terms = ", ".join(
            "%s=%.4e" % (name, coefficient / scale if scale else 0.0)
            for name, coefficient, scale in zip(self.regime.features,
                                                self.coefficients, self.scales))
        note = ("" if not self.scope_undeclared else
                "; scope undeclared: " + ", ".join(self.scope_undeclared))
        return ("%s from %d point(s), %d residual df, in-sample %.1f%% [%s]%s"
                % (self.regime.name, self.points, self.residual_df,
                   self.relative_error * 100, terms, note))


class _Absent:
    """Distinct from a declared ``None``.

    `sliding_window: None` is a statement -- there is no window. Not carrying
    the key at all is the absence of a statement. Collapsing the two would let
    an observation that declares nothing pool with one that declares a window
    is off.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<undeclared>"


ABSENT = _Absent()


def _distinct(values):
    """The distinct values among these, by equality, tolerating lists."""
    out = []
    for value in values:
        if any((value is ABSENT) == (other is ABSENT)
               and (value is ABSENT or value == other) for other in out):
            continue
        out.append(value)
    return out


def _scope_of(observations):
    """The static scope these observations share, or a refusal to pool them.

    Agreement is required on **every** key any observation declares, not only
    on :data:`REQUIRED_SCOPE`. The required ones name the kernel; the rest name
    the conditions -- tensor-parallel geometry, which rotation the capture used,
    whether a file is superseded -- and two measurements that differ in any of
    them are measurements of different things. A key one observation carries and
    another does not is a disagreement too: the second has not said it matches,
    and reading its silence as agreement is the failure this is closed against.
    """
    keys = set(REQUIRED_SCOPE)
    for obs in observations:
        keys.update(obs[3] or {})
    declared, undeclared = {}, []
    for field in sorted(keys):
        seen = _distinct([(obs[3] or {}).get(field, ABSENT)
                          for obs in observations])
        if len(seen) == 1 and seen[0] is ABSENT:
            undeclared.append(field)
            continue
        if len(seen) > 1:
            shown = ", ".join(sorted(repr(s) for s in seen))
            if field in REQUIRED_SCOPE:
                return Refusal(
                    "these observations disagree on %s (%s); they are "
                    "measurements of different kernels and pooling them would "
                    "average work that is not the same work" % (field, shown))
            return Refusal(
                "these observations disagree on %s (%s); they were taken "
                "under different conditions, and pooling them into one "
                "training observation would average measurements of "
                "different deployments" % (field, shown))
        declared[field] = seen[0]
    return declared, tuple(f for f in undeclared if f in REQUIRED_SCOPE)


def fit_regime(regime, observations, *, strict=True, min_residual_df=1):
    """Fit one regime, or refuse and say what would close the gap.

    ``observations`` are ``(structure, seconds, source, scope)`` tuples, each
    from a source primitive measurement of this one operator.

    Refuses, rather than producing a number, when: the observations do not
    declare the static scope (under ``strict``); they disagree on it; there are
    fewer points than features plus the required residual degrees of freedom;
    the design is rank deficient, which is what perfectly collinear features
    look like; or the fit wants a negative cost for some work.
    """
    scope = _scope_of(observations)
    if isinstance(scope, Refusal):
        return scope
    declared, undeclared = scope
    if strict and undeclared:
        return Refusal(
            "these measurements do not declare %s, so nothing says they were "
            "taken on one kernel; a price from them would be pooling regimes "
            "that may differ" % ", ".join(undeclared),
            missing=undeclared)

    rows, rhs, domain = [], [], []
    for structure, seconds, _source, _scope in observations:
        values = features_for(regime, structure)
        if isinstance(values, Refusal):
            return values
        rows.append(values)
        rhs.append(float(seconds))
        domain.append(values)
    wanted = len(regime.features) + min_residual_df
    if len(rows) < wanted:
        return Refusal(
            "%s has %d independent point(s) and needs at least %d to fit %d "
            "term(s) with anything left over to check them against"
            % (regime.name, len(rows), wanted, len(regime.features)))

    # Scale each column by its largest value: the features differ by many
    # orders of magnitude -- attended pairs against sequence counts -- and the
    # normal equations would otherwise be conditioned by the units.
    scales = [max((abs(row[i]) for row in rows), default=0.0) or 1.0
              for i in range(len(regime.features))]
    scaled = [[row[i] / scales[i] for i in range(len(row))] for row in rows]
    solved = _solve(scaled, rhs)
    if solved is None:
        return Refusal(
            "%s: the design is rank deficient -- two or more of %s do not "
            "vary independently across these points, so their coefficients "
            "cannot be told apart. Measure a point that separates them."
            % (regime.name, ", ".join(regime.features)))
    negative = [name for name, c in zip(regime.features, solved) if c < 0]
    if negative:
        return Refusal(
            "%s: the fit wants a negative cost for %s, which is not a cost. "
            "Either a term is missing or these points are not one regime."
            % (regime.name, ", ".join(negative)))

    predicted = [sum(c * v for c, v in zip(solved, row)) for row in scaled]
    errors = [abs(p - y) / y for p, y in zip(predicted, rhs) if y]
    return Fit(regime, solved, scales, len(rows), len(rows) - len(solved),
               max(errors) if errors else 0.0, domain, declared, undeclared)


class Model:
    """Every regime that could be fitted, and a price for a structure."""

    __slots__ = ("fits", "refusals", "strict")

    def __init__(self, strict=True):
        self.fits: dict = {}
        self.refusals: dict = {}
        self.strict = strict

    @classmethod
    def from_observations(cls, grouped, *, strict=True, min_residual_df=1):
        """``grouped`` maps a regime name to its observation list."""
        model = cls(strict=strict)
        for name, observations in grouped.items():
            regime = REGIMES.get(name)
            if regime is None:
                model.refusals[name] = Refusal(f"{name} is not a known regime")
                continue
            outcome = fit_regime(regime, observations, strict=strict,
                                 min_residual_df=min_residual_df)
            if isinstance(outcome, Refusal):
                model.refusals[name] = outcome
            else:
                model.fits[name] = outcome
        return model

    def price(self, op: dict):
        """Seconds for this call, or a `Refusal` naming what is missing."""
        structure = structure_of(op)
        if structure is None:
            return Refusal("this operator records no ragged structure")
        regime = regime_of(op, structure)
        if isinstance(regime, Refusal):
            return regime
        fit = self.fits.get(regime.name)
        if fit is None:
            known = self.refusals.get(regime.name)
            return Refusal(
                "%s has no fitted law: %s" % (
                    regime.name,
                    known.reason if known else "no measurement reached it"))
        values = features_for(regime, structure)
        if isinstance(values, Refusal):
            return values
        outside = _outside_domain(fit, values)
        if outside:
            return Refusal(
                "%s: %s is outside the measured range %s, and this law is a "
                "fit over what was measured rather than a claim about "
                "everywhere" % (regime.name, outside[0], outside[1]))
        seconds = fit.predict(values)
        if not (seconds > 0 and math.isfinite(seconds)):
            return Refusal(
                "%s: the law returns %r for this structure, which is not a "
                "duration" % (regime.name, seconds))
        return seconds

    def coverage(self) -> dict:
        """What is modelled and what is not, for a report to state plainly."""
        return {
            "fitted": {name: fit.describe() for name, fit in
                       sorted(self.fits.items())},
            "refused": {name: refusal.reason for name, refusal in
                        sorted(self.refusals.items())},
            "strict": self.strict,
        }


def _outside_domain(fit, values):
    """Which feature, if any, sits outside the measured hull, and its range."""
    for index, name in enumerate(fit.regime.features):
        column = [row[index] for row in fit.domain]
        low, high = min(column), max(column)
        if values[index] < low or values[index] > high:
            return name, "[%g, %g]" % (low, high)
    return None
