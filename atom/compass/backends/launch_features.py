# SPDX-License-Identifier: MIT
"""What a step's launch ran, per model architecture, as host-law columns.

A host law prices ``(1, tokens, seqs)``, so a model whose launch takes another
path from step to step, a graph replayed or run eagerly, a batch split into
micro-batches or not, names those paths as its provider's `features`, which a
host law may adopt as columns
(`step_linear_fit.fit_host`). `REGISTRY` maps an architecture, as ATOM keys its
models (``hf_config.architectures[0]``), to its provider; any other takes
`LaunchFeatures`, which declares none and changes no record, key or law. A
recording real runner writes a declaring provider's `observe` on each line and
the graph sets ATOM recorded in its host key's engine config under `GRAPHS`
(`atom.compass.parity`); `predict` restates `observe` from a step's view and
those sets, for a fit to check and a run to price (`HostDeviceRecurrence`).
"""

import collections
import dataclasses
import functools
import os

from atom.compass.backends.shape import sum_tokens

#: The host key's engine-config field holding a recording runner's graph sets:
#: ``architecture``, ``token``, each phase's token-graph buckets, ``request``,
#: each request graph's keys, and the provider's own (`LaunchFeatures.recorded`).
GRAPHS = "launch_graphs"


class LaunchFeatures:
    """A model's launch features: none."""

    features: tuple[str, ...] = ()

    def install(self, runner) -> None:
        """Ready `runner`, a real runner just built, for `observe`."""

    def recorded(self, runner) -> dict:
        """The graph sets of `runner`'s model, once ATOM has recorded them,
        that `predict` reads beyond `GRAPHS`' own."""
        return {}

    def observe(self, runner) -> dict[str, float]:
        """Each feature's value for the step `runner` just ran."""
        return {}

    def predict(self, view, graphs: dict) -> dict[str, float]:
        """Each feature's value for a step of the `BatchView` `view` on a
        runner whose ATOM recorded `graphs`, a `GRAPHS` mapping."""
        return {}


class DeepseekV41(LaunchFeatures):
    """DeepSeek-V4.1's prefill launch: its body, the layers before the decoder,
    replays the smallest prefill token-graph bucket holding its tokens or runs
    eagerly past the largest; a prefill finishing a prompt then runs the
    decoder on its tail, replaying the request graph recorded for the step's
    `running_bs` or eagerly. A decode is neither. `install` counts what
    replayed; `recorded` states the decoder's buckets, None where it does not
    replay; `predict` reads `running_bs` off the view's `target_bs`, which
    `ForwardMode.decide` settles, and runs a DP dummy's tail where its group's
    runs (`group_tail`). A step ATOM runs eagerly for a reason a view
    does not carry, a score workspace too small for one band among them,
    refuses its fit by name."""

    features = ("eager_body", "eager_tail")

    def install(self, runner) -> None:
        model, seen = _v41_model(runner), collections.Counter()
        runner.compass_launch_seen = seen
        body = model.early if model.replay else model.backbone

        def counted(name, call):
            def count(*args, **kwargs):
                seen[name] += 1
                return call(*args, **kwargs)

            return count

        builder = runner.attn_metadata_builder
        prepare = builder.prepare_prefill

        def prepare_prefill(*args, **kwargs):
            # A step's counts start here. The body's token graph is made when
            # ATOM records it (`region_graph._run_token_graph`), after
            # `install`; the decoder's runs while recording are not a step's.
            graph = vars(body).get("_token_graph")
            if graph is not None and "replay" not in vars(graph):
                graph.replay = counted("body", graph.replay)
            seen.clear()
            seen["prefill"] = 1
            return prepare(*args, **kwargs)

        builder.prepare_prefill = prepare_prefill
        if model.replay:
            model.decoder.run = counted("tail", model.decoder.run)
            graph = model.decoder.graph
            graph.replay = counted("tail_replays", graph.replay)

    def recorded(self, runner) -> dict:
        model = _v41_model(runner)
        keys = model.decoder.graph.keys() if model.replay else None
        return {"decoder": None if keys is None else sorted(map(list, keys))}

    def observe(self, runner) -> dict[str, float]:
        seen = runner.compass_launch_seen
        prefill = bool(seen["prefill"])
        observed = {
            "eager_body": float(prefill and not seen["body"]),
            "eager_tail": float(prefill and seen["tail"] > seen["tail_replays"]),
        }
        seen.clear()
        return observed

    def predict(self, view, graphs: dict) -> dict[str, float]:
        if all(r.decode for r in view.requests):
            return dict.fromkeys(self.features, 0.0)
        # Under DP, ATOM picks the body's token-graph bucket by the group's
        # most tokens (`ForwardMode._pad_to_token_graph`), and a tail any rank
        # of the group finishes runs the decoder on every rank, its MoE
        # collectives being the group's (`BatchView.group_tail`).
        tokens = max(sum_tokens(view.requests), view.group_max_tokens or 0)
        decoder = graphs["decoder"]
        # A DP dummy runs the decoder only beside a peer's tail: it samples
        # nothing of its own, though its line produces output.
        own = bool(view.produces_output) and not view.dp_dummy
        tail = (own or view.group_tail) and decoder is not None
        running_bs = view.target_bs or len(view.requests)
        return {
            "eager_body": float(all(n < tokens for n in graphs["token"]["prefill"])),
            "eager_tail": float(tail and [running_bs, True] not in decoder),
        }


def _v41_model(runner):
    """The V4.1 runtime model `runner` serves: the module that holds its
    replay stages and decides whether its prefills replay the decoder."""
    for module in runner.model.modules():
        if hasattr(module, "decoder_replay_active"):
            return module
    raise ValueError(
        f"{type(runner.model).__name__} holds no DeepSeek-V4.1 runtime model, "
        "so its launch features cannot be observed"
    )


class DeepseekV4(LaunchFeatures):
    """DeepSeek-V4's launch under two-batch overlap: a step its data-parallel
    group splits (`BatchView.tbo_tokens`) launches each micro-batch in turn
    through ATOM's `UBatchWrapper`, unless it is a decode replaying a graph
    recorded split (`capture_rung`). `install` counts the wrapper's split
    runs, `_run_ubatches`, found on the class so that a module forwarding
    unknown names to the wrapper is not taken for it, from the start of each
    launch (`run_model`): ATOM records its graphs after the runner is built,
    a bucket that splits through the same path. A model built without
    two-batch overlap has no wrapper."""

    features = ("split",)

    def install(self, runner) -> None:
        seen = runner.compass_launch_seen = collections.Counter()
        for module in runner.model.modules():
            if callable(getattr(type(module), "_run_ubatches", None)):
                module._run_ubatches = functools.partial(
                    _split, seen, module._run_ubatches
                )
        runner.run_model = functools.partial(_launch, seen, runner.run_model)

    def observe(self, runner) -> dict[str, float]:
        return {"split": float(runner.compass_launch_seen["split"] > 0)}

    def predict(self, view, graphs: dict) -> dict[str, float]:
        split = view.tbo_tokens is not None and view.capture_rung is None
        return {"split": float(split)}


def _split(seen, run, *args, **kwargs):
    seen["split"] += 1
    return run(*args, **kwargs)


def _launch(seen, run_model, *args, **kwargs):
    seen.clear()
    return run_model(*args, **kwargs)


REGISTRY: dict[str, LaunchFeatures] = {
    "DeepseekV41ForCausalLM": DeepseekV41(),
    "DeepseekV4ForCausalLM": DeepseekV4(),
}


def provider(architecture: str | None) -> LaunchFeatures:
    return REGISTRY.get(architecture) or LaunchFeatures()


def architecture(config) -> str | None:
    """The architecture ATOM builds `config`'s model from, None without one."""
    hf = getattr(config, "hf_config", None)
    return (getattr(hf, "architectures", None) or [None])[0]


def stated(
    key: dict | None, long_tokens: int | None = None
) -> tuple[LaunchFeatures, dict | None]:
    """The provider and graph sets the host key `key` states, if any; with
    `long_tokens`, a host entry's `LONG_TOKENS`, the provider with the
    `LONG_CONTEXT` feature at that bound (`LongContext`)."""
    graphs = ((key or {}).get("engine") or {}).get(GRAPHS)
    base = provider(graphs and graphs["architecture"])
    return (LongContext(base, long_tokens) if long_tokens else base), graphs


#: The variable that, set to a token count, has a host fit price a forward whose
#: longest prefill row reaches that much context with the `LONG_CONTEXT` feature
#: (`LongContext`), a launch column its laws may adopt; unset or ``0``, no
#: feature is added, and a fit is as before.
LONG_ENV = "COMPASS_FIT_HOST_LONG_CONTEXT"
#: The feature: 1 on a forward whose longest prefill row's context reaches the
#: bound, else 0.
LONG_CONTEXT = "long_context"
#: A host entry's bound for `LONG_CONTEXT`, in tokens, which a run's provider
#: then predicts it at (`stated`).
LONG_TOKENS = "long_context_tokens"


def long_tokens_fit() -> int | None:
    """`LONG_ENV`'s bound, None unset or ``0``; anything not a whole number of
    tokens is refused."""
    value = os.environ.get(LONG_ENV) or "0"
    if not value.isdigit():
        raise ValueError(f"{LONG_ENV}={value!r} is not a whole number of tokens")
    return int(value) or None


def long_context(view, tokens: int) -> float:
    """`LONG_CONTEXT` of a step of the `BatchView` `view` at the bound `tokens`."""
    return float(any(r.context_tokens >= tokens for r in view.prefill))


def long_samples(samples: list, tokens: int) -> list:
    """`samples` (`step_labels.Sample`), each one's recorded launch features
    with its `LONG_CONTEXT` at the bound `tokens`, as `LongContext` predicts
    it."""
    return [
        dataclasses.replace(
            s, launch={**s.launch, LONG_CONTEXT: long_context(s.view, tokens)}
        )
        for s in samples
    ]


class LongContext(LaunchFeatures):
    """A provider, `base`, with one more feature, `LONG_CONTEXT` at `tokens`,
    read off a step's view alone: a record states none, so a fit adds it to
    each sample's recorded features (`fit.host`) and a run predicts it. On
    #752's Kimi-K3.1 (TP8 DCP8) a prefill's or middle chunk's pre-launch host
    time (`W_pre`) steps from 2-4 ms to 45-55 ms where its context passes
    about 128k tokens (16384 a DCP rank), on the calibration record and on the
    c=1 and c=8 targets alike, which ``(1, tokens, seqs)`` cannot state."""

    def __init__(self, base: LaunchFeatures, tokens: int) -> None:
        self.base, self.tokens = base, int(tokens)
        self.features = (*base.features, LONG_CONTEXT)

    def install(self, runner) -> None:
        self.base.install(runner)

    def recorded(self, runner) -> dict:
        return self.base.recorded(runner)

    def observe(self, runner) -> dict[str, float]:
        return self.base.observe(runner)

    def predict(self, view, graphs: dict) -> dict[str, float]:
        return {
            **self.base.predict(view, graphs),
            LONG_CONTEXT: long_context(view, self.tokens),
        }


#: The host entry's key holding launch-path constants: per kind and segment,
#: the constant a forward on each launch path (`path`) takes in place of its
#: law's own, which prices every path it does not name
#: (`fit.host_blocks.refit` fits them).
PATHS = "paths"


def path(view, launch: dict) -> str:
    """The launch path of a forward of the `BatchView` `view` whose launch
    ran `launch`, its launch features (`LaunchFeatures.predict`): a decode's
    CUDA-graph rung, ``rung<r>``, or ``eager`` where it captured none; any
    other forward's ``replay``; then each feature `launch` states true,
    sorted, all joined by ``+``, a feature replacing ``replay``. A fit reads
    it on a sample's view and recorded features, a run on the view it prices
    and the features its provider predicts. DeepSeek-V4.1: a decode
    ``rung16``; a prefill ``replay``, ``eager_body``, ``eager_tail`` or
    ``eager_body+eager_tail``."""
    flags = sorted(f for f, v in launch.items() if v)
    if view.produces_output and not view.prefill:
        head = ["eager" if view.capture_rung is None else f"rung{view.capture_rung}"]
    else:
        head = [] if flags else ["replay"]
    return "+".join(head + flags)
