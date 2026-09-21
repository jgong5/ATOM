# SPDX-License-Identifier: MIT
"""CPU-only cover for whether a capture keeps its shapes symbolic.

A concrete inventory is valid only at the shapes it was taken at. A symbolic
one can be evaluated anywhere in its guard domain, which is the difference
between an inventory that can price an unseen batch and one that cannot.

Four facts, each pinned rather than asserted in prose:

* The mechanism does carry free symbols -- through a GEMM and a softmax, with
  nothing specialised.
* A tensor allocated inside the mode is already static, and the specialisation
  check prescribed for a capture says nothing about that case.
* On ATOM's real forward the trace is concrete, and the cause is ATOM's own
  input staging: `CpuGpuBuffer.copy_to_gpu(n)` is `self.gpu[:n].copy_(...)`,
  and a Python `int` bound makes the staged tensor a constant -- which every
  operator downstream of the staging then inherits.
* The same copy with the symbol itself as the bound keeps the symbol, through
  `CpuGpuBuffer` unmodified. The repair belongs at the caller that chooses
  `n`, not in the buffer.
"""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

from atom.compass.capture.fake_trace import Recorder, shape_entry_census

# The staged buffer in a decode step is `[max_num_tokens, ...]` and is filled
# to the token count of the batch. Both numbers are small here; only their
# relationship matters.
BUFFER_ROWS = 17
FILLED_ROWS = 5


def _mode() -> tuple[ShapeEnv, FakeTensorMode]:
    shape_env = ShapeEnv()
    return shape_env, FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def _dynamic_leading_dim(fake_mode: FakeTensorMode, real: torch.Tensor):
    """Convert a REAL tensor, keeping its first dimension a free symbol.

    Allocating inside the mode instead would produce a fake tensor that is
    already static, with plain `int` shapes and no sign anything went wrong.
    """
    return fake_mode.from_tensor(
        real,
        static_shapes=False,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC] + [DimDynamic.STATIC] * (real.dim() - 1)
        ),
    )


def _is_symbolic(dim) -> bool:
    """A dimension the inventory would record as something other than a number."""
    return not str(dim).lstrip("-").isdigit()


def test_the_mechanism_carries_a_free_symbol_through_a_gemm():
    """The capture can be symbolic at all.

    The contracted dimension is static and the batch dimension is not, which
    is the shape a decode step has. Nothing specialises: `replacements` stays
    empty, which on a trace that DID create a symbol is a real check.
    """
    shape_env, fake_mode = _mode()
    x = _dynamic_leading_dim(fake_mode, torch.empty(BUFFER_ROWS, 64))
    w = fake_mode.from_tensor(torch.empty(64, 64), static_shapes=True)

    rec = Recorder()
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        y = torch.matmul(x, w)
        out = torch.nn.functional.softmax(y, dim=-1)

    assert _is_symbolic(out.shape[0]), out.shape
    entries, free = shape_entry_census(rec)
    assert free > 0, f"0 of {entries} recorded shape entries carried a symbol"
    assert not shape_env.replacements, dict(shape_env.replacements)


def test_a_tensor_allocated_inside_the_mode_is_already_static():
    """The trap that makes a capture concrete while reporting nothing."""
    shape_env, fake_mode = _mode()
    rec = Recorder()
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        x = torch.empty(BUFFER_ROWS, 64)
        x + x

    entries, free = shape_entry_census(rec)
    assert entries > 0
    assert free == 0, "a tensor allocated inside the mode should be static"
    # And the specialisation check says nothing about it: an empty
    # `replacements` here means nothing was checked, not that it is clean.
    assert not shape_env.replacements


def _staging_buffer(fake_mode: FakeTensorMode):
    """ATOM's own staging buffer, with a symbolic destination.

    Built outside the mode, as the runner builds it, then the device side is
    converted with a free leading dimension -- the most favourable starting
    point a symbolic capture of the staging path could have.
    """
    from atom.utils import CpuGpuBuffer

    buf = CpuGpuBuffer(
        BUFFER_ROWS,
        8,
        dtype=torch.int32,
        device=torch.device("cpu"),
        pin_memory=False,
        with_numpy=True,
    )
    buf.gpu = _dynamic_leading_dim(fake_mode, buf.gpu)
    assert _is_symbolic(buf.gpu.shape[0]), "the destination must start symbolic"
    return buf


def test_staging_a_batch_by_a_python_int_specialises_what_the_model_then_reads():
    """The measured cause of a concrete capture of ATOM's forward.

    `copy_to_gpu(n)` returns `self.gpu[:n]`, and that tensor -- not the buffer
    -- is what the forward reads. With `n` a Python `int` its shape is a
    constant however symbolic the buffer was, so every operator downstream of
    the staging is priced at one batch size.
    """
    _shape_env, fake_mode = _mode()
    buf = _staging_buffer(fake_mode)

    rec = Recorder()
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        staged = buf.copy_to_gpu(FILLED_ROWS)

    assert list(staged.shape) == [FILLED_ROWS, 8], staged.shape
    assert not any(_is_symbolic(d) for d in staged.shape)
    # The copy itself records the constant, so the inventory carries no trace
    # of the dimension having ever been free.
    copies = [o for o in rec.ops if "copy" in o.op]
    assert copies, [o.op for o in rec.ops]
    assert copies[-1].out_shapes == [[str(FILLED_ROWS), "8"]], copies[-1].out_shapes


def test_staging_the_same_batch_by_the_symbol_keeps_it():
    """The repair, through `CpuGpuBuffer` unmodified.

    `copy_to_gpu` takes whatever bound it is given; passing the symbol rather
    than an integer is enough. So what a symbolic capture needs changing is
    the caller that decides `n`, not the buffer.
    """
    shape_env, fake_mode = _mode()
    buf = _staging_buffer(fake_mode)

    rec = Recorder()
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        staged = buf.copy_to_gpu(buf.gpu.shape[0])

    assert _is_symbolic(staged.shape[0]), staged.shape
    entries, free = shape_entry_census(rec)
    assert free > 0, f"0 of {entries} recorded shape entries carried a symbol"
    assert not shape_env.replacements, dict(shape_env.replacements)


def test_the_staging_source_being_a_real_tensor_is_not_the_cause():
    """Separates the two candidate explanations.

    Both sides symbolic, the bound still a Python `int`: what the forward
    reads is concrete anyway. So it is the integer bound that specialises,
    not the source being a real numpy-backed CPU tensor.
    """
    _shape_env, fake_mode = _mode()
    src = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    dst = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )

    rec = Recorder()
    with fake_mode, torch._C._EnablePythonDispatcher(), rec:
        staged = dst[:FILLED_ROWS].copy_(src[:FILLED_ROWS])

    assert list(staged.shape) == [FILLED_ROWS, 8], staged.shape


# --------------------------------------------------------------------------
# whether the symbol can reach the staging copy at all on ATOM's decode path


def test_a_symbolic_bound_used_as_a_numpy_index_is_specialised_silently():
    """The reason the repair cannot be made at the caller.

    Staging fills the CPU side through the buffer's numpy view before the
    device copy happens. numpy takes `__index__` of whatever bound it is
    given, and on a `SymInt` that resolves to the hint -- so the symbol is
    replaced by a constant, with no error and no warning, before
    `copy_to_gpu` is ever reached.
    """
    import numpy as np

    shape_env, fake_mode = _mode()
    sym = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    bound = sym.shape[0]
    assert _is_symbolic(bound), bound
    assert not shape_env.replacements

    staging = np.zeros((BUFFER_ROWS, 8), dtype=np.int32)
    staging[:bound] = -1

    # It did not raise, and it did not write the batch: it wrote the hint.
    assert int((staging == -1).all(axis=1).sum()) == BUFFER_ROWS
    # And the symbol is now a constant, which `capture()` refuses outright.
    assert shape_env.replacements, "the numpy bound left the symbol free"
    assert str(BUFFER_ROWS) in str(dict(shape_env.replacements))


def test_the_same_bound_stays_symbolic_when_numpy_does_not_see_it():
    """The control for the test above: the slice itself is not the problem."""
    shape_env, fake_mode = _mode()
    sym = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    bound = sym.shape[0]

    with fake_mode, torch._C._EnablePythonDispatcher():
        staged = sym[:bound]

    assert _is_symbolic(staged.shape[0]), staged.shape
    assert not shape_env.replacements, dict(shape_env.replacements)


def test_the_decode_path_shares_one_bound_between_numpy_and_the_device_copy():
    """The precondition the finding above rests on, checked against ATOM.

    `prepare_decode` derives one count per staged buffer and uses it twice:
    to fill the numpy view, and as the device copy's bound. Only the second
    may be symbolic, and the first runs first. Separating them is a change to
    this file, which is why a symbolic capture is not reachable from the
    caller alone.

    If ATOM ever does separate them, this fails, and the conclusion drawn
    from it has to be retaken rather than inherited.
    """
    import pathlib

    import atom

    source = (
        pathlib.Path(atom.__file__).parent
        / "model_ops"
        / "attentions"
        / "aiter_attention.py"
    ).read_text()

    # the numpy fill, the shared count, and the device copy that follows it
    assert 'var["slot_mapping"].np[:running_tokens]' in source
    assert '("slot_mapping", running_tokens),' in source
    assert "copy_to_gpu(num) for el, num in vars_used" in source
