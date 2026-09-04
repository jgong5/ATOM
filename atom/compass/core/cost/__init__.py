"""Cost oracles, and the words for talking about them.

Two things are costed, and they are not the same subject:

**model step**
    The forward pass, for a step of a given shape.
**serving**
    Everything around it -- admission, scheduling, KV cache management. Note
    that serving *decisions* are not modelled at all: Compass runs ATOM's real
    scheduler and replaces only the forward, so what is costed here is the time
    serving consumes, not the choices it makes.

A cost is obtained in one of two ways:

**analytical**
    Computed without measuring the subject. A closed form from the device's
    specification, for instance.
**empirical**
    Derived from measuring the subject. This is a genus, not a single method,
    and it is written ``empirical/<species>``:

    ``empirical/measured``
        The unit itself is timed. `priced` prices each operator and sums them.
    ``empirical/fitted``
        A functional form is chosen and its coefficients regressed over measured
        steps. `calibrated`.
    ``empirical/interpolated``
        No form is assumed; nearby measurements are looked up. `interpolated`.

    The species above are the ones built so far, not a closed set --
    ``empirical/extrapolated`` and others are equally admissible, and naming the
    genus separately is what leaves room for them.

So:

============ ============ =========================
oracle       subject      how
============ ============ =========================
`constant`   model step   declared -- a stub, for proving the plumbing
`calibrated` model step   empirical/fitted
`interpolated` model step empirical/interpolated
`priced`     model step   empirical/measured
============ ============ =========================

`priced` reaches a model step by summing its operators and `calibrated` by
fitting the step directly. That is a difference of method, not of subject: both
answer what one step costs.

Nothing analytical exists here yet. The word is reserved rather than aspirational
-- the memory-budget code already uses it in the same sense, for a figure derived
from the checkpoint index rather than measured.

Running one workload under two oracles and subtracting gives a gap analysis,
because everything except the oracle is held fixed.

(This replaces an earlier taxonomy that described a design which was never built:
it defined `calibrated` as "the analytical shape scaled by a measured efficiency
factor", which has no analytical component and never did, and gave the genus
name `empirical` to what is one species of it.)
"""

from atom.compass.core.cost.base import CostOracle, StepCost

__all__ = ["CostOracle", "StepCost"]
