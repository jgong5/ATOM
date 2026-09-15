"""Per-family parametric prices, with the support each one is valid over.

The exact-signature library in ``core/cost/library.py`` prices an operator only
when some pricing run met that operator's whole key. For most of the graph that
key is shapes, dtypes and architectural scalars, and it repeats. For attention
it also carries ``slot_mapping``, ``positions`` and the per-request
``context_lens``, which are step-specific allocator and position state -- so the
key cannot repeat across cohorts by construction, and a library assembled from
one cohort prices its own four cases completely and a neighbouring cohort barely
at all.

This package does not replace that library and does not simulate anything. It
adds one thing: for a family whose key differs from a measured key *only* in
components that family's price is a declared function of, it answers with a
price derived from the measured ones, carrying the provenance of every
measurement it came from and the uncertainty the measurements themselves show.
Where the difference is in a component nobody has measured the price's
dependence on, it refuses and says which component.

The three pieces:

* :mod:`features` -- what each family's price may depend on, and what it may
  not. A nuisance component is one the price is declared independent of, and
  every such declaration names the evidence for it or is marked unmeasured.
* :mod:`support` -- the region measurements actually cover, as measured points
  rather than a bounding box, with the rule for when two of them are close
  enough to interpolate between.
* :mod:`adapter` -- a :class:`PriceLibrary` subclass that consults the above
  when the exact key misses. It overrides ``lookup`` and nothing else, so
  ``PriceLibrary.body`` and ``LibraryCostOracle`` use it unchanged.
"""

from atom.compass.core.cost.families.features import (
    FAMILY_CONTRACTS,
    FamilyContract,
    Nuisance,
    aligns,
    contract_for,
    grouping_key,
    infer_rows,
)
from atom.compass.core.cost.families.support import (
    MeasuredCurve,
    MeasuredPoint,
    Price,
    Refusal,
    RowSupport,
)
from atom.compass.core.cost.families.adapter import (
    INTERPOLATED_SCHEME,
    ParametricPriceLibrary,
    coverage_split,
)

__all__ = [
    "FAMILY_CONTRACTS",
    "INTERPOLATED_SCHEME",
    "FamilyContract",
    "MeasuredCurve",
    "MeasuredPoint",
    "Nuisance",
    "ParametricPriceLibrary",
    "Price",
    "Refusal",
    "RowSupport",
    "aligns",
    "contract_for",
    "coverage_split",
    "grouping_key",
    "infer_rows",
]
