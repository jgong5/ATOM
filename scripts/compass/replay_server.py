#!/usr/bin/env python3
"""Start ATOM's OpenAI server for a replay, on a machine with no GPU.

    python scripts/compass/replay_server.py --compass-replay-target target.json \
        --model ... --compass --compass-mode predict [...]

Same arguments as ``python -m atom.entrypoints.openai_server``, and the same
server: this does exactly two things before handing over.

1. Answers AITER's import-time architecture query from the captured target, so
   that building a ``Config`` does not require a device to be present. See
   :mod:`atom.compass.replay.bootstrap` for what that does and does not supply.
2. Hands over. ``main`` is ATOM's, ``argv`` is untouched, and nothing about the
   server, the scheduler or the engine differs from the GPU path.

It exists as a separate entry point rather than a flag on the server because
step 1 has to happen before ``atom`` is imported at all, and by the time a flag
could be parsed inside the server the import has already failed.

A GPU-free run still needs AITER, Triton and Torch *installed* -- it just no
longer needs them to find a device. The two requirements are separate; this
prints both so the evidence does not conflate them.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _target_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--compass-replay-target" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--compass-replay-target="):
            return arg.split("=", 1)[1]
    raise SystemExit(
        "scripts/compass/replay_server.py: --compass-replay-target is "
        "required. Without it there is no record of the deployment being "
        "replayed, and nothing to take the architecture from.")


def main() -> None:
    from atom.compass.replay.bootstrap import install_from_target

    target = _target_from_argv(sys.argv[1:])
    state = install_from_target(target)

    # And again in every process ATOM spawns. The engine core is a `spawn`
    # child with a fresh interpreter, so it inherits environment but not
    # `sys.modules`; `_sitedir/sitecustomize.py` re-installs there from these
    # two variables. Set even when this process did not need the stub, so that
    # a child with a different view of the devices behaves the same way.
    from atom.compass.replay import bootstrap as _bootstrap

    os.environ["ATOM_COMPASS_REPLAY_ARCH"] = str(state["arch"] or "")
    os.environ["ATOM_COMPASS_REPLAY_ARCH_SOURCE"] = str(state["source"] or "")
    sitedir = os.path.join(os.path.dirname(os.path.abspath(
        _bootstrap.__file__)), "_sitedir")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [sitedir] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                     if p and p != sitedir])

    print(f"### replay bootstrap: arch={state['arch']} "
          f"installed={state['installed']} ({state['reason']}); "
          f"children via {sitedir}", flush=True)

    from atom.utils import envs, set_ulimit

    if envs.USE_ATOMESH_ENTRYPOINTS:
        from atom.entrypoints.atomesh.server import main as server_main
    else:
        from atom.entrypoints.openai.api_server import main as server_main

    set_ulimit()
    server_main()


if __name__ == "__main__":
    main()
