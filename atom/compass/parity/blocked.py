# SPDX-License-Identifier: MIT
"""The launch thread's wall time inside HIP's blocking calls, from `blocked.cpp`.

    python -m atom.compass.parity.blocked DIR

builds the tool, a rocprofiler-sdk tool library, into DIR and prints its path.
A server started with ``ROCP_TOOL_LIBRARIES`` naming it loads it into each
process that initializes HIP, and the tool sums, per thread and per call, the
wall time inside stream, event and device synchronizes, synchronous copies and
frees. A recording real runner reads its own thread's sums (`loaded`) at its
launch's and token copy wait's stamps, so a step's line states what the launch
spent blocked (`parity.StepRecording`).
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

#: The tool's file and soname, by which a process finds it loaded.
LIBRARY = "libcompass_blocked.so"
SOURCE = Path(__file__).with_name("blocked.cpp")


def build(directory) -> Path:
    """Compile the tool into `directory` against ROCm's rocprofiler-sdk."""
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    out = Path(directory) / LIBRARY
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            *("g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-D__HIP_PLATFORM_AMD__"),
            *(f"-I{rocm}/include", str(SOURCE), f"-L{rocm}/lib", "-lrocprofiler-sdk"),
            *(f"-Wl,-soname,{LIBRARY},-rpath,{rocm}/lib", "-o", str(out)),
        ],
        check=True,
    )
    return out


class Blocked:
    """The tool loaded in this process: calling it reads the calling thread's
    nanoseconds inside each traced call, in the order of `names`."""

    def __init__(self, lib) -> None:
        lib.compass_blocked_name.restype = ctypes.c_char_p
        n = lib.compass_blocked_calls()
        self.names = tuple(lib.compass_blocked_name(i).decode() for i in range(n))
        self._sums = (ctypes.c_uint64 * n)()
        self._read = lib.compass_blocked_read

    def __call__(self) -> tuple[int, ...]:
        self._read(self._sums)
        return tuple(self._sums)

    def between(self, before, after) -> dict[str, int]:
        """The nanoseconds per call from reading `before` to reading `after`,
        each call that blocked."""
        return {n: b - a for n, a, b in zip(self.names, before, after) if b != a}


def loaded() -> Blocked | None:
    """The tool, if HIP's initialization loaded it into this process; None
    without it. Loaded with no tracing started, it refuses: its sums would
    read nothing blocked."""
    try:
        lib = ctypes.CDLL(LIBRARY, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    except OSError:
        return None
    if lib.compass_blocked_calls() < 0:
        raise RuntimeError(
            f"{LIBRARY} is loaded but rocprofiler-sdk started no tracing in it; "
            "it is loaded through ROCP_TOOL_LIBRARIES before HIP initializes"
        )
    return Blocked(lib)


if __name__ == "__main__":
    print(build(sys.argv[1]))
