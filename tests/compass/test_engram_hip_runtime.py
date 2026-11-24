# SPDX-License-Identifier: MIT
"""Engram's page-locked shard allocator binds the HIP runtime torch loaded.

Where `LD_LIBRARY_PATH` resolves `libamdhip64.so` to a file other than the one
torch maps, loading HIP by that name starts a second runtime in the process,
and memory it page-locks is pageable to torch's: a torch copy from it faulted
on MI455. These tests put a copy of torch's `libamdhip64` first on
`LD_LIBRARY_PATH`, a second file as that image's SDK directory is, and hold
that the allocator maps no runtime beside torch's, and that with one already
mapped it declines, so the shard is registered in place through torch.

CPU-only: nothing calls into the runtime, so no driver is needed. Each case
runs in its own interpreter, since a mapped runtime cannot be unmapped.
"""

import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch


def hip_files():
    """The `libamdhip64` files this process maps, sorted."""
    with open("/proc/self/maps") as maps:
        names = {"".join(line.split(maxsplit=5)[5:]).strip() for line in maps}
    return sorted(n for n in names if os.path.basename(n).startswith("libamdhip64.so"))


def file_at(addr):
    """The file mapped at `addr` in this process."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            lo, hi = (int(x, 16) for x in fields[0].split("-"))
            if lo <= addr < hi:
                return "".join(fields[5:]).strip()
    return None


@pytest.fixture
def torch_hip(tmp_path):
    """Torch's `libamdhip64` file, copied to `tmp_path/libamdhip64.so`."""
    if not torch.version.hip:
        pytest.skip("torch has no HIP runtime here")
    (path,) = hip_files()
    shutil.copyfile(path, tmp_path / "libamdhip64.so")
    return path


def child(lib_dir, body):
    """`body` run in a fresh interpreter, after `import torch` and with
    `lib_dir` first on `LD_LIBRARY_PATH`; the JSON it prints last."""
    env = dict(os.environ, ATOM_ENGRAM_UVA_COPY="1")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(lib_dir), os.environ.get("LD_LIBRARY_PATH")) if p
    )
    script = "\n".join(
        [
            "import ctypes, json, os",
            "import torch",
            "from atom.model_ops.engram import host_memory",
            inspect.getsource(hip_files),
            inspect.getsource(file_at),
            textwrap.dedent(body),
        ]
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_allocator_binds_the_runtime_torch_maps_not_the_one_named_first(
    torch_hip, tmp_path
):
    seen = child(
        tmp_path,
        """
        before = hip_files()
        allocator = host_memory._host_allocator()
        malloc = ctypes.cast(allocator._malloc, ctypes.c_void_p).value
        print(json.dumps([before, hip_files(), file_at(malloc)]))
        """,
    )
    before, after, malloc_in = seen
    assert before == [torch_hip]
    assert after == [torch_hip]
    assert malloc_in == torch_hip


def test_with_a_second_runtime_mapped_the_shard_is_registered_in_place(
    torch_hip, tmp_path
):
    seen = child(
        tmp_path,
        """
        ctypes.CDLL("libamdhip64.so")
        loaded = hip_files()
        host_memory._register = lambda sources, leases, **kw: "registered in place"
        out = host_memory.pin_rows([torch.zeros(4096, dtype=torch.uint8)])
        print(json.dumps([loaded, out]))
        """,
    )
    loaded, out = seen
    assert len(loaded) == 2 and torch_hip in loaded
    assert out == "registered in place"
