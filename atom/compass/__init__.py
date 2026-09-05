"""ATOMCompass — performance modelling for ATOM.

Compass runs ATOM's own control plane — scheduler, block manager, KV admission —
and replaces only the GPU compute with a predicted duration. Nothing about
batching or scheduling is re-implemented, so behaviour tracks the engine by
construction rather than by maintenance.

The package is split so that the interesting part is testable without a GPU,
without ATOM, and without a container:

``atom.compass.core``
    Pure modelling. Imports nothing from ATOM. Given operator shapes it yields
    times. This is where cost oracles live.

``atom.compass.runtime``
    The only ATOM-aware code: the model runner that ATOM injects, and the
    tracing and profiling that feed the core.
"""

from atom.compass.config import CompassConfig

__all__ = ["CompassConfig"]
