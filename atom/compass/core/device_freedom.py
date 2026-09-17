"""Whether the process that predicted could have reached a device.

The gate is a GPU-free replay, so the claim is about a *process*: the one that
produced the prediction could not have used an accelerator. Every reading here
is taken by that process, about itself, and the record carries enough of its
identity for someone else to check which process it was.

What this replaces is a probe that ran afterwards, in its own container, and
hashed the JSON files it found there. That is a true statement about a
container and a directory, and its own output said so -- it "does not mean that
the artifacts named in `covers` were produced by this exact process". A
GPU-container replay followed by a CPU-container probe of the same shared
artifacts satisfies it exactly.

Two readings, at the two moments that can differ:

``launch``
    taken when the predictor's state is built, before it has served anything.
``readback``
    taken when the record is asked for, after it has. A device that appeared
    in between -- a node bind-mounted in, a driver opened by something in the
    container -- is visible as a difference between the two.

**What is evidence, and what is only recorded.** The verdict rests on facts
about the container and this process: whether the device nodes exist, and
whether this process holds an open file descriptor onto one. It does *not*
rest on what the runtime says, and that distinction is load-bearing rather than
fastidious. `atom.compass.replay.bootstrap` answers architecture queries from
the captured target so that AITER can be imported with no device present, and a
replay interpreter is in the business of describing the hardware being modelled.
A device count read out of that interpreter is a statement about the
simulation. It is kept, under ``reported_by_runtime``, because a reader should
see it -- and it is kept *out* of the verdict, which the record says in terms.

Stdlib only, and nothing imported from ATOM: a reader must be able to check
this without standing up an engine, exactly as for
:mod:`atom.compass.core.process_identity`.
"""

from __future__ import annotations

import os

from atom.compass.core import process_identity

__all__ = ["DEVICE_NODES", "NAMESPACES", "observe", "same_process"]

#: Nodes whose presence means a device is reachable from this container.
#: `/dev/dri` is a directory; the rest are character devices. Presence, not
#: usability, is the test: a fallback that is reachable is a fallback that can
#: be taken.
#:
#: This is the list the protocol asks about. `cc_traces_validate` keeps its own
#: copy, because it must stay loadable without importing `atom`, and a test
#: pins the two equal: they drifted by one entry while this was being written,
#: and the reading that omitted `/dev/nvidia-uvm` passed as a reading that had
#: found nothing.
DEVICE_NODES = (
    "/dev/kfd",
    "/dev/dri",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia0",
)

#: The namespaces that decide what a process can see of the machine's devices.
#: Recorded as the kernel's own identifiers, so two processes can be compared
#: without either being trusted about which container it was in.
NAMESPACES = ("mnt", "pid", "net", "user", "cgroup")

#: What the record says about the runtime reading, every time, so that no
#: reader has to infer why a device count is not the answer.
RUNTIME_IS_NOT_THE_VERDICT = (
    "recorded, not counted: the replay bootstrap answers hardware queries "
    "from the captured target, so a device count read inside this interpreter "
    "describes the deployment being modelled and not the devices this process "
    "could reach"
)


def _namespaces() -> dict:
    """This process's namespace identifiers, as the kernel names them."""
    found = {}
    for name in NAMESPACES:
        try:
            found[name] = os.readlink(f"/proc/self/ns/{name}")
        except OSError:
            found[name] = None
    return found


def _device_cgroup() -> list | None:
    """The cgroup lines that decide device access, where the kernel shows them."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except OSError:
        return None


def _own_driver_handles() -> list:
    """Driver file descriptors held by *this* process.

    Its own, not the container's. A container-wide scan is a stronger claim and
    a much weaker reading: it walks every `/proc/<pid>` it is allowed to, so it
    reports whatever else is running beside the predictor, and it says nothing
    about which process held what. What the gate needs is whether the process
    that produced the prediction had a device open.
    """
    handles = []
    try:
        entries = sorted(os.listdir("/proc/self/fd"))
    except OSError:
        return handles
    for entry in entries:
        try:
            target = os.readlink(f"/proc/self/fd/{entry}")
        except OSError:
            continue
        if target.startswith(("/dev/kfd", "/dev/dri", "/dev/nvidia")):
            handles.append({"fd": entry, "target": target})
    return handles


def observe(when: str, *, runtime_report: dict | None = None) -> dict:
    """One reading, taken by the process it is about.

    ``runtime_report`` is whatever the caller wants recorded about what the
    runtime says it can see. It is stored and ignored; see the module
    docstring. The caller supplies it rather than this module reading it,
    because importing a deep-learning runtime to answer a question its answer
    is not trusted for would be a strange thing to do here.
    """
    nodes = {node: os.path.exists(node) for node in DEVICE_NODES}
    handles = _own_driver_handles()
    return {
        "when": when,
        "process": process_identity.identity(),
        "namespaces": _namespaces(),
        "device_cgroup": _device_cgroup(),
        "device_nodes": nodes,
        "own_driver_handles": handles,
        "device_free": not any(nodes.values()) and not handles,
        "reported_by_runtime": runtime_report,
        "runtime_note": RUNTIME_IS_NOT_THE_VERDICT,
        "means": (
            "the process that took this reading held no open driver handle, "
            "and no device node existed in its mount namespace at the moment "
            "it looked"
        ),
    }


def same_process(one: dict, other: dict) -> bool:
    """Whether two readings were taken by the same run of the same process.

    Pid alone is not identity -- pids are reused -- so this is pid with start
    time, on the same boot of the same machine. That is the tuple a caller can
    also read out of `/proc` for itself, which is the point: the record is
    checkable rather than merely self-consistent.
    """
    keys = ("host", "boot_id", "pid", "start_ticks")
    first = (one or {}).get("process") or {}
    second = (other or {}).get("process") or {}
    if not first or not second:
        return False
    return all(first.get(key) == second.get(key) for key in keys)
