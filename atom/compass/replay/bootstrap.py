"""Tell an import-time architecture query what the captured device was.

The GPU-free replay runs ATOM's real control plane, and ATOM's real control
plane builds a ``Config``. Building a ``Config`` resolves ``aiter.QuantType``,
and importing AITER runs::

    # aiter/ops/triton/utils/_triton/arch_info.py
    try:
        _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib
        _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]

With no device visible Triton raises ``RuntimeError: 0 active drivers``, so the
import lands on the fallback, and this container has no JAX. AITER is then
unimportable and the replay cannot start -- not because it needs a GPU, but
because a string describing one is unavailable.

So supply the string. The replay already carries a record of the machine the
capture ran on; ``arch`` is one more field of it, and answering the query from
that record is the honest answer to "which architecture is this replay about?".
It also fixes the capability flags AITER derives from the arch (``is_fp8_avail``
and friends) to the *captured* deployment's rather than the replay host's, which
is what a replay of that deployment should see.

What this does and does not do:

- It supplies an architecture **name**, and nothing else, to the two places
  AITER asks: the Triton/JAX fallback above, and ``GPU_ARCHS``, which AITER's
  JIT already documents as the way to name the chip instead of shelling out to
  ``rocminfo``. No kernel, no allocator, no numerical result is substituted;
  there is no fake ``aiter`` module. Everything AITER exports is the genuine
  AITER object, as before.
- It uses the fallback branch AITER's own authors wrote for "no active Triton
  driver". Nothing in the installed third-party tree is patched or shadowed.
- It is a **no-op when a device is present**: if Triton can answer, Triton
  answers, and this module installs nothing. A replay cannot use it to hide a
  real device, and a mismatch against the captured arch is reported.
- Real GPU compute still fails. Nothing here makes a kernel launchable; with no
  device visible, an unexpected device call raises as it would have anyway.

This is a *software* dependency seam, not a hardware one. The replay still
needs AITER, Triton and Torch installed -- it just no longer needs them to find
a GPU. Those two requirements are separate and are reported separately.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import types
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["install", "install_from_target", "state", "ArchUnavailable"]

_STATE: dict = {"installed": False, "arch": None, "gpu_archs": None,
                "redundant_installs": 0,
                "source": None, "reason": None, "calls": 0,
                "chip_info_hook": False, "chip_info_calls": 0}


class ArchUnavailable(RuntimeError):
    """The replay has no architecture to answer with, from anywhere."""


def state() -> dict:
    """What the bootstrap did, for a run to report rather than assert."""
    return dict(_STATE)


def _live_arch() -> Optional[str]:
    """What Triton says, if a driver is active. ``None`` if none is."""
    try:
        import triton
    except Exception:  # noqa: BLE001 - no Triton is not this module's problem
        return None
    try:
        return str(triton.runtime.driver.active.get_current_target().arch)
    except Exception:  # noqa: BLE001 - RuntimeError('0 active drivers') and kin
        return None


def _jax_installed() -> bool:
    """Is there a real JAX to leave the fallback to?"""
    try:
        return importlib.util.find_spec("jax") is not None
    except Exception:  # noqa: BLE001 - a broken jax install is not one we use
        return False


class _ChipInfoLoader:
    """AITER's own `chip_info`, executed unmodified, then told the chip."""

    def __init__(self, inner, arch: str) -> None:
        self._inner, self._arch = inner, arch

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        arch = self._arch

        def _detect_native() -> list:
            _STATE["chip_info_calls"] += 1
            return [arch]

        module._detect_native = _detect_native



class _ChipInfoFinder:
    TARGET = "aiter.jit.utils.chip_info"

    def __init__(self, arch: str) -> None:
        self._arch = arch

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.TARGET:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _ChipInfoLoader(spec.loader, self._arch)
                return spec
        return None


def _seed_chip_info(arch: str) -> None:
    """Answer AITER's *runtime* chip query, which has no env seam.

    `get_gfx_custom_op_core` reads ``GPU_ARCHS``; `get_gfx_runtime` is
    documented to ignore it and always shell out to ``rocminfo`` -- right on a
    machine with a device, an exception in a container without one. It runs
    during ``import aiter`` itself (``utility/dtypes.py`` picks the fp8 dtype
    from it), so there is no post-import moment at which to answer.

    So AITER's own `chip_info` is loaded, unmodified, from its own files, and
    the single function that asks the driver is replaced by the answer the
    capture recorded. Nothing on disk is patched, nothing else in the module is
    touched, and any other device query still reaches the driver -- and, here,
    still fails.
    """
    if any(isinstance(f, _ChipInfoFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _ChipInfoFinder(arch))
    _STATE["chip_info_hook"] = True


def _process_state():
    """The bootstrap state of this *process*, which may not be this module's.

    `_sitedir/sitecustomize.py` loads this very file by path, under the name
    `atom_compass_replay_bootstrap`, so that a spawned child pays for stdlib
    and one module instead of importing all of `atom` at interpreter startup.
    That copy is a different module object with its own `_STATE`, while the
    thing it installs -- the `sys.meta_path` finder and the `jax` stub -- is
    per-process. So by the time the engine core imports
    `atom.compass.replay.bootstrap` the child is bootstrapped, and this module
    object is the only thing that does not know it.

    That is how a correctly bootstrapped worker came to refuse the tracer's
    `install_from_target` with "this process bootstrapped None": not a late
    bootstrap, which the guard is right to refuse, but the same bootstrap seen
    through the other copy. Read back rather than re-derived, so the adopted
    state keeps the original call's source and counters.
    """
    here = os.path.abspath(__file__)
    for name, module in list(sys.modules.items()):
        if module is None or name == __name__:
            continue
        path = getattr(module, "__file__", None)
        if not path or os.path.abspath(path) != here:
            continue
        other = getattr(module, "_STATE", None)
        if isinstance(other, dict) and other.get("installed"):
            return dict(other), name
    return None


def install(arch: str, *, source: str = "unknown") -> dict:
    """Make the architecture query answerable with ``arch``.

    Returns :func:`state`. Raises if AITER has already been imported, because
    ``_CACHED_ARCH`` is resolved once at import time and a later answer would be
    read as having been used when it was not.
    """
    if not arch:
        raise ArchUnavailable(
            "ATOMCompass: no architecture to replay with. The captured target "
            "must record one -- re-capture with a build that writes "
            "`hardware.arch`, or pass the architecture explicitly.")
    if "aiter" in sys.modules:
        # Already imported. Before deciding that nothing answered, ask the
        # process rather than this module object: under `_sitedir` the answer
        # lives in a second copy of this file. See `_process_state`.
        if not _STATE["installed"]:
            found = _process_state()
            if found is not None:
                adopted, where = found
                _STATE.update(adopted)
                _STATE["adopted_from"] = where
        # Already answered, with the architecture being asked for: the
        # derivation path's second call is redundant rather than wrong.
        # Refusing it here is what stopped a device-free `derive=1` server
        # from reaching a single step.
        if _STATE["installed"] and str(_STATE["arch"]) == str(arch):
            # Not "calls", which counts architecture queries answered. Its
            # own field, so a report can say the derivation path asked again
            # rather than the second call leaving no trace at all.
            _STATE["redundant_installs"] = (
                _STATE.get("redundant_installs", 0) + 1)
            return state()
        raise RuntimeError(
            "ATOMCompass: aiter is already imported, so its cached "
            "architecture is already resolved and this call would change "
            f"nothing (this process bootstrapped {_STATE['arch']!r} and is "
            f"now asked for {str(arch)!r}). Bootstrap the replay before "
            "importing atom.")
    live = _live_arch()
    if live is not None:
        # A device is visible. Triton will answer and the fallback is never
        # reached, so installing anything would be inert -- and claiming to have
        # replayed GPU-free on a machine with a GPU would be worse than inert.
        if live.split(":")[0] != str(arch).split(":")[0]:
            logger.warning(
                "ATOMCompass WARNING: this host's GPU is %s and the captured "
                "target is %s. The run will use the host's, because a visible "
                "device answers first; the prediction is therefore about a "
                "different architecture than the one captured.", live, arch)
        _STATE.update(installed=False, arch=live, source="live-triton-driver",
                      reason="a device is visible; Triton answered")
        return state()

    if _jax_installed():
        # Real JAX is installed. Shadowing it would be a much larger claim than
        # this module is entitled to make, and its own answer comes from a real
        # device if there is one.
        _STATE.update(installed=False, arch=None, source="installed-jax",
                      reason="jax is installed; leaving the fallback to it")
        logger.warning(
            "ATOMCompass WARNING: no Triton driver, but jax is installed, so "
            "AITER's fallback will ask jax for the architecture and this "
            "captured value (%s) is unused. If that fails, the replay cannot "
            "start.", arch)
        return state()

    def get_arch_details(device: str) -> str:
        """Stand in for JAX's device query with what the capture recorded."""
        _STATE["calls"] += 1
        return str(arch)

    # AITER's *other* architecture query, and the one it documents: the JIT
    # asks `rocminfo` for the chip unless `GPU_ARCHS` names it. In a container
    # with no device nodes `rocminfo` exits 1, so answer with the captured chip
    # instead. This is AITER's own metadata seam, not a stand-in -- the value is
    # the same one the capture ran on, and it selects nothing that runs here.
    bare = str(arch).split(":")[0]
    os.environ["GPU_ARCHS"] = bare
    _seed_chip_info(bare)

    gpu_triton = types.ModuleType("jax._src.lib.gpu_triton")
    gpu_triton.__doc__ = (
        "Not JAX. ATOMCompass supplies this one function so that AITER's "
        "import-time architecture fallback can be answered from a captured "
        "record instead of from a device. See "
        "atom.compass.replay.bootstrap.")
    gpu_triton.get_arch_details = get_arch_details
    gpu_triton.__atom_compass_stub__ = True

    jax = types.ModuleType("jax")
    jax.__atom_compass_stub__ = True
    jax.__doc__ = gpu_triton.__doc__
    src = types.ModuleType("jax._src")
    lib = types.ModuleType("jax._src.lib")
    lib.gpu_triton = gpu_triton
    src.lib = lib
    jax._src = src
    for name, module in (("jax", jax), ("jax._src", src),
                         ("jax._src.lib", lib),
                         ("jax._src.lib.gpu_triton", gpu_triton)):
        sys.modules[name] = module

    _STATE.update(installed=True, arch=str(arch), gpu_archs=bare, source=source,
                  reason="no Triton driver and no jax; answering from the "
                         "captured target")
    logger.info("ATOMCompass: replaying as %s, from %s. No device was asked.",
                arch, source)
    return state()


def install_from_target(path: str) -> dict:
    """Read the architecture out of a captured replay target and install it.

    Deliberately reads the file by hand rather than through
    :class:`~atom.compass.replay.runner.TargetRecord`. This has to run before
    ``atom`` is imported at all, so it takes no import it does not need; the
    record's own validation happens later, when the runner loads it properly.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"ATOMCompass: no replay target at {path!r}, so there is no "
            f"architecture to replay as.")
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    hardware = (blob.get("hardware") or {})
    arch = hardware.get("arch")
    if not arch:
        raise ArchUnavailable(
            f"ATOMCompass: {path} records no `hardware.arch`. It was captured "
            f"by a build that did not write one; re-capture this configuration "
            f"so the replay knows which architecture it is about.")
    return install(arch, source=f"{path} (captured on "
                                f"{hardware.get('device_name') or 'unknown'})")
