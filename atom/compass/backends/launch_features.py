# SPDX-License-Identifier: MIT
"""What a step's launch ran, per model architecture, as host-law columns.

A host law prices ``(1, tokens, seqs)``, so a model whose launch takes another
path from step to step, a graph replayed or run eagerly, names those paths as
its provider's `features`, which a host law may adopt as columns
(`step_linear_fit.fit_host`). `REGISTRY` maps an architecture, as ATOM keys its
models (``hf_config.architectures[0]``), to its provider; any other takes
`LaunchFeatures`, which declares none and changes no record, key or law. A
recording real runner writes a declaring provider's `observe` on each line and
the graph sets ATOM recorded in its host key's engine config under `GRAPHS`
(`atom.compass.parity`); `predict` restates `observe` from a step's view and
those sets, for a fit to check and a run to price (`HostDeviceRecurrence`).
"""

#: The host key's engine-config field holding a recording runner's graph sets:
#: ``architecture``, ``token``, each phase's token-graph buckets, and
#: ``request``, each request graph's keys.
GRAPHS = "launch_graphs"


class LaunchFeatures:
    """A model's launch features: none."""

    features: tuple[str, ...] = ()

    def install(self, runner) -> None:
        """Ready `runner`, a real runner just built, for `observe`."""

    def observe(self, runner) -> dict[str, float]:
        """Each feature's value for the step `runner` just ran."""
        return {}

    def predict(self, view, graphs: dict) -> dict[str, float]:
        """Each feature's value for a step of the `BatchView` `view` on a
        runner whose ATOM recorded `graphs`, a `GRAPHS` mapping."""
        return {}


REGISTRY: dict[str, LaunchFeatures] = {}


def provider(architecture: str | None) -> LaunchFeatures:
    return REGISTRY.get(architecture) or LaunchFeatures()


def architecture(config) -> str | None:
    """The architecture ATOM builds `config`'s model from, None without one."""
    hf = getattr(config, "hf_config", None)
    return (getattr(hf, "architectures", None) or [None])[0]


def stated(key: dict | None) -> tuple[LaunchFeatures, dict | None]:
    """The provider and graph sets the host key `key` states, if any."""
    graphs = ((key or {}).get("engine") or {}).get(GRAPHS)
    return provider(graphs and graphs["architecture"]), graphs
