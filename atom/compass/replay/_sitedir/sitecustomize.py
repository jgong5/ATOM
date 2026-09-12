"""Carry the replay's architecture into the processes ATOM spawns.

The launcher installs the answer to AITER's import-time architecture query in
its own process, but ATOM's engine core is a *spawned* child, and `spawn`
starts a fresh interpreter: nothing in `sys.modules` survives. The child
unpickles its arguments, that unpickling imports AITER, and AITER asks for an
architecture again -- in a process that was never told.

`sitecustomize` is the interpreter's own startup hook and runs before anything
the child imports, which is the only place early enough. It is reached because
the launcher puts this directory on `PYTHONPATH`, which children inherit; it
does nothing at all unless `ATOM_COMPASS_REPLAY_ARCH` is set, so a stray
`PYTHONPATH` cannot silently change an unrelated run.

This directory holds exactly one file, and this file holds no logic of its own:
the stub is `atom/compass/replay/bootstrap.py`, loaded here by path so that a
child pays for stdlib and that one module rather than for importing `atom` at
interpreter startup. One implementation of the seam, in one place.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain() -> None:
    """Run the `sitecustomize` this interpreter would have run without us.

    Debian ships one. Shadowing it silently would make this module responsible
    for whatever it does, which is not a responsibility a replay should take.
    """
    import importlib.machinery
    import importlib.util

    paths = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    try:
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
    except Exception:  # noqa: BLE001 - startup must not fail over this
        return
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    # Under its own name: `sys.modules["sitecustomize"]` is this module, mid
    # execution, and rebinding it here would leave the import machinery
    # holding a half-built object.
    sys.modules["sitecustomize_chained"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        pass


def _install_arch() -> None:
    arch = os.environ.get("ATOM_COMPASS_REPLAY_ARCH")
    if not arch:
        return
    import importlib.util

    path = os.path.join(os.path.dirname(_HERE), "bootstrap.py")
    spec = importlib.util.spec_from_file_location(
        "atom_compass_replay_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install(arch, source=os.environ.get(
        "ATOM_COMPASS_REPLAY_ARCH_SOURCE", "ATOM_COMPASS_REPLAY_ARCH"))


_chain()
_install_arch()
