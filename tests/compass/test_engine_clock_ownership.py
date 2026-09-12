"""Who owns the process clock, and for how long.

`LLMEngine` puts the process on a virtual clock when Compass asks for one,
because arrival is stamped in this process and first-token in the engine core,
and a TTFT that subtracts one from the other needs both readings from the same
clock. The clock is process-wide, so installing it is a loan, not a gift: an
ordinary engine constructed afterwards in the same process would otherwise
stamp every arrival at a frozen instant and report a wrong TTFT rather than a
missing one.

Two boundaries are covered here: a Compass engine that closes, and a
constructor that raises.
"""

import pytest

from atom.model_engine.llm_engine import LLMEngine, _install_compass_clock
from atom.utils.clock import VirtualClock, WallClock, get_clock, reset_clock, set_clock


class _Compass:
    def __init__(self, enabled=True, virtual_clock=True, epoch=1000.0):
        self.enabled = enabled
        self.virtual_clock = virtual_clock
        self.epoch = epoch


class _Config:
    def __init__(self, compass=None):
        self.compass_config = compass


@pytest.fixture(autouse=True)
def _wall_clock():
    reset_clock()
    yield
    reset_clock()


def _engine(config):
    """An object carrying just the clock state, for the unbound methods."""
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine._compass_clock, engine._clock_before_compass = _install_compass_clock(
        config
    )
    return engine


def test_compass_off_leaves_the_clock_alone():
    engine = _engine(_Config(_Compass(enabled=False)))
    assert engine._compass_clock is None
    assert isinstance(get_clock(), WallClock)


def test_a_virtual_clock_is_installed_and_reported():
    engine = _engine(_Config(_Compass()))
    assert get_clock() is engine._compass_clock
    assert isinstance(get_clock(), VirtualClock)
    assert get_clock().time() == pytest.approx(1000.0)


def test_close_hands_the_process_back():
    """The point of the whole file: after close, a later arrival is wall time."""
    before = get_clock()
    engine = _engine(_Config(_Compass()))
    assert get_clock() is not before
    LLMEngine._restore_clock(engine)
    assert get_clock() is before
    assert isinstance(get_clock(), WallClock)


def test_close_is_idempotent():
    engine = _engine(_Config(_Compass()))
    LLMEngine._restore_clock(engine)
    LLMEngine._restore_clock(engine)
    assert isinstance(get_clock(), WallClock)


def test_close_does_not_take_a_clock_someone_else_installed():
    """Restoring over a third party's clock is the same bug pointed the other
    way, so the engine only takes back a clock that is still its own."""
    engine = _engine(_Config(_Compass()))
    later = VirtualClock(epoch=7.0)
    set_clock(later)
    LLMEngine._restore_clock(engine)
    assert get_clock() is later


def test_close_restores_even_when_the_core_shutdown_raises():
    class _Boom:
        def close(self):
            raise RuntimeError("core is already gone")

    engine = _engine(_Config(_Compass()))
    engine.core_mgr = _Boom()
    with pytest.raises(RuntimeError, match="core is already gone"):
        LLMEngine.close(engine)
    assert isinstance(get_clock(), WallClock)


def test_a_constructor_that_raises_leaves_no_clock_behind(monkeypatch):
    """Through the real __init__: a failure has no engine to close, so the
    clock must not be installed until construction has succeeded."""
    import atom.model_engine.llm_engine as mod

    # The real Config is a dataclass the constructor introspects, so it stays.
    # What is replaced is the install itself, by one that always installs: if
    # __init__ reaches it before the failure below, the clock leaks and this
    # test says so. That is the ordering property, stated directly.
    def _always_install(config):
        clock = VirtualClock(epoch=5.0)
        return clock, set_clock(clock)

    monkeypatch.setattr(mod, "_install_compass_clock", _always_install)

    def _no_tokenizer(*a, **kw):
        raise OSError("no checkpoint on this box")

    monkeypatch.setattr(mod, "_load_tokenizer", _no_tokenizer)
    # Which failure comes first does not matter -- config resolution reaches
    # for the checkpoint before the tokenizer does. What matters is that the
    # process is still on wall time afterwards.
    with pytest.raises(OSError):
        LLMEngine("no-such-model-for-this-test")
    assert isinstance(get_clock(), WallClock)
