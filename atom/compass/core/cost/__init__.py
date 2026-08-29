"""Cost oracles: the F1 dial.

An oracle answers one question — how long does this step take — and every oracle
answers it from the same input, so they are interchangeable at run time:

``analytical``
    Closed form from the device's specification. A speed-of-light bound.
``calibrated``
    The analytical shape scaled by a measured efficiency factor.
``empirical``
    Measured on this hardware and this engine.

Running one workload under two oracles and subtracting gives a gap analysis
attributable per operator, because everything except the oracle is held fixed.
"""

from atom.compass.core.cost.base import CostOracle, StepCost

__all__ = ["CostOracle", "StepCost"]
