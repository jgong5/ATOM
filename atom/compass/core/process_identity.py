"""Which process is answering, as distinct from what code it runs.

A digest of the source says two servers were built from the same bytes. It
says nothing about *who replied*: a server left over from an earlier run,
holding the same port and started from the same tree with the same flags,
matches on every byte-and-configuration field there is. Telling the two apart
needs facts about the running process, and those are what this module reads.

The facts are deliberately ones that two sides can read independently. A
caller that launched a process can look up the same numbers in `/proc` and
compare them with what the server said about itself. That makes the server's
answer checkable rather than merely believable -- which is the whole point,
since a stale server is perfectly capable of describing itself honestly and
still not being the process anyone asked for.

Stdlib only, and no import from ATOM, per this subpackage's contract: a caller
must be able to check who served a request without loading an engine.
"""

from __future__ import annotations

import os

__all__ = [
    "boot_id",
    "identity",
    "parent_of",
    "start_ticks",
]


def _stat_fields(pid) -> list[str] | None:
    """The fields of `/proc/<pid>/stat` from field 3 onward.

    Read from after the last `)` rather than by splitting the whole line:
    field 2 is the executable name in parentheses and may contain spaces, so a
    plain split puts every later field at an offset that depends on the name.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read().decode("utf-8", "replace")
        return raw[raw.rindex(")") + 1 :].split()
    except Exception:  # noqa: BLE001 - identity is best effort
        return None


def start_ticks(pid="self") -> int | None:
    """When a process started, in clock ticks since this machine booted.

    A pid alone is not an identity. Pids are reused, and a reused one names a
    different process that the caller never started; start time is what tells
    those two apart.
    """
    fields = _stat_fields(pid)
    try:
        return int(fields[19])  # field 22, counting from field 3
    except (IndexError, TypeError, ValueError):
        return None


def parent_of(pid) -> int | None:
    """The parent pid, so a caller can walk up to the process it launched.

    A server need not be the direct child: a launcher may fork, and then the
    process holding the socket is a descendant. Being inside the caller's
    process tree is the claim worth checking, and it is one a stale server
    cannot make.
    """
    fields = _stat_fields(pid)
    try:
        return int(fields[1])  # field 4
    except (IndexError, TypeError, ValueError):
        return None


def boot_id() -> str | None:
    """Which boot of which machine, so a tick count means something.

    Ticks are counted from boot. The same number after a reboot, or on another
    machine, is a different instant.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id") as handle:
            return handle.read().strip() or None
    except Exception:  # noqa: BLE001 - identity is best effort
        return None


def identity(pid="self") -> dict:
    """A process's own account of itself, for someone else to check."""
    try:
        ticks = os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - identity is best effort
        ticks = None
    return {
        "pid": os.getpid() if pid == "self" else int(pid),
        "ppid": parent_of(pid),
        "host": os.uname().nodename,
        "boot_id": boot_id(),
        "start_ticks": start_ticks(pid),
        "ticks_per_second": ticks,
    }
