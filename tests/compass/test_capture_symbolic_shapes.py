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
* On ATOM's real forward it does not, and the cause is ATOM's own input
  staging -- in **two independent places**. The bound: `copy_to_gpu(n)`
  returns `self.gpu[:n]`, and a Python `int` there makes the staged tensor a
  constant, which every operator downstream inherits. And the copy itself:
  `self.cpu` is a real numpy-backed tensor with constant dimensions, so
  `copy_` solves every symbolic dimension of the destination that the slice
  does not cover.
* Only the first is reachable from the caller. Passing the symbol as the
  bound keeps the sliced dimension and leaves the other site untouched, so
  the buffer is not unchanged by a repair -- it holds one of the two sites.
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
# a second staging dimension, which the copy never slices
BUFFER_COLS = 16384


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


def test_staging_the_same_batch_by_the_symbol_keeps_the_sliced_dimension():
    """Half the repair: the bound can be carried, on the dimension it slices.

    `copy_to_gpu` takes whatever bound it is given, and with the symbol rather
    than an integer the staged tensor keeps it. That closes one of the two
    measured specialisation sites and not the other -- the buffer has a second
    one inside it, which no choice of bound reaches. See
    `test_the_staging_copy_specialises_every_dimension_it_does_not_slice`.
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


def test_an_int_bound_constants_the_sliced_dimension_even_with_a_symbolic_source():
    """Isolates what the bound alone does, on the dimension it slices.

    Both sides symbolic and the bound still a Python `int`: the sliced
    dimension of what the forward reads is a constant anyway, so the bound is
    sufficient on its own to specialise that one.

    It says nothing about the others. The staging source being a real tensor
    with constant dimensions is a **separate and independent** cause, and the
    earlier reading of this test -- that the real source is *not* a cause --
    is withdrawn; see
    `test_the_staging_copy_specialises_every_dimension_it_does_not_slice`.
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
# what actually stops the staged tensor being symbolic, measured as two
# independent sites rather than one


def test_a_symbolic_bound_is_collapsed_by_taking_its_index():
    """Site one: any consumer that needs an `int` resolves the symbol.

    Staging fills the CPU side before the device copy, and that fill indexes
    by the same count. Taking `__index__` of a `SymInt` returns its hint and
    records the symbol as a constant -- with no error and no warning. This is
    not a numpy behaviour: a bare `__index__` and a plain list slice do it
    too, so it is every `int`-consuming use of the bound, not one library's.
    """
    shape_env, fake_mode = _mode()
    sym = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    bound = sym.shape[0]
    assert _is_symbolic(bound), bound
    assert not shape_env.replacements

    assert bound.__index__() == BUFFER_ROWS

    assert shape_env.replacements, "taking __index__ left the symbol free"
    assert str(BUFFER_ROWS) in str(dict(shape_env.replacements))


def test_the_same_collapse_happens_through_a_plain_list_slice():
    """The generality of site one, without numpy or torch in the way."""
    shape_env, fake_mode = _mode()
    sym = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    bound = sym.shape[0]

    assert len(list(range(BUFFER_ROWS))[:bound]) == BUFFER_ROWS
    assert shape_env.replacements, dict(shape_env.replacements)


def test_the_same_bound_stays_symbolic_when_nothing_takes_its_index():
    """The control: the slice itself is not what collapses it."""
    shape_env, fake_mode = _mode()
    sym = _dynamic_leading_dim(
        fake_mode, torch.empty(BUFFER_ROWS, 8, dtype=torch.int32)
    )
    bound = sym.shape[0]

    with fake_mode, torch._C._EnablePythonDispatcher():
        staged = sym[:bound]

    assert _is_symbolic(staged.shape[0]), staged.shape
    assert not shape_env.replacements, dict(shape_env.replacements)


def _two_dim_buffer(fake_mode: FakeTensorMode, dynamic_sizes):
    """ATOM's buffer at a staging shape with two dimensions.

    `block_tables` is `[max_num_seqs, max_blocks]`; the staging copy slices the
    first and never the second.
    """
    from atom.utils import CpuGpuBuffer

    buf = CpuGpuBuffer(
        BUFFER_ROWS,
        BUFFER_COLS,
        dtype=torch.int32,
        device=torch.device("cpu"),
        pin_memory=False,
        with_numpy=True,
    )
    buf.gpu = fake_mode.from_tensor(
        buf.gpu,
        static_shapes=False,
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=dynamic_sizes),
    )
    return buf


def test_the_staging_copy_specialises_every_dimension_it_does_not_slice():
    """Site two, and it is inside the buffer rather than at its caller.

    `copy_to_gpu` is `self.gpu[:n].copy_(self.cpu[:n])`, and `self.cpu` is a
    real numpy-backed tensor with concrete dimensions. `copy_` requires the
    shapes to match, so every symbolic dimension of the destination other than
    the sliced one is solved against the source's constant.

    This is the site the measured `block_tables: ['512', 's64']` -> 16384 runs
    through, and no choice of `n` reaches it.
    """
    shape_env, fake_mode = _mode()
    buf = _two_dim_buffer(fake_mode, [DimDynamic.STATIC, DimDynamic.DYNAMIC])
    assert _is_symbolic(buf.gpu.shape[1]), buf.gpu.shape

    with fake_mode, torch._C._EnablePythonDispatcher():
        staged = buf.copy_to_gpu(4)

    assert list(staged.shape) == [4, BUFFER_COLS], staged.shape
    assert shape_env.replacements, "the unsliced dimension stayed symbolic"
    assert str(BUFFER_COLS) in str(dict(shape_env.replacements))


def test_site_two_is_reached_whatever_the_bound_is():
    """So the two sites are independent, and fixing the bound cannot close it.

    With the bound the symbol itself -- the repair that closes site one -- the
    unsliced dimension is specialised exactly as before.
    """
    shape_env, fake_mode = _mode()
    buf = _two_dim_buffer(fake_mode, [DimDynamic.DYNAMIC, DimDynamic.DYNAMIC])

    with fake_mode, torch._C._EnablePythonDispatcher():
        staged = buf.copy_to_gpu(buf.gpu.shape[0])

    # the sliced dimension survives, the other does not
    assert _is_symbolic(staged.shape[0]), staged.shape
    assert not _is_symbolic(staged.shape[1]), staged.shape
    assert str(BUFFER_COLS) in str(dict(shape_env.replacements))


def test_the_decode_path_shares_one_bound_between_the_fill_and_the_copy():
    """The precondition site one rests on, checked against ATOM.

    `prepare_decode` derives one count per staged buffer and uses it to fill
    the CPU side and then as the device copy's bound. Both builders the
    recorded replacement stack names do it -- `aiter_attention` through the
    buffer's numpy view, `gdn_attn` through its CPU tensor directly. If ATOM
    ever separates the two uses, this fails, and the conclusion drawn from it
    is retaken rather than inherited.
    """
    import pathlib

    import atom

    attentions = pathlib.Path(atom.__file__).parent / "model_ops" / "attentions"

    aiter_src = (attentions / "aiter_attention.py").read_text()
    assert 'var["slot_mapping"].np[:running_tokens]' in aiter_src
    assert '("slot_mapping", running_tokens),' in aiter_src
    assert "copy_to_gpu(num) for el, num in vars_used" in aiter_src

    # The recorded replacement stack names a second builder, which stages the
    # same way through `.cpu[...]` rather than `.np[...]`. Compared with
    # whitespace removed so reflowing the call does not break the check.
    gdn_src = "".join((attentions / "gdn_attn.py").read_text().split())
    assert '["cu_seqlens_q"].cpu[running_bs:]=batch.total_tokens_num_decode' in gdn_src
    assert '"cu_seqlens_q"].copy_to_gpu(running_bs+1)' in gdn_src
