"""Who owns the process clock, and for how long.

`LLMEngine` puts the process on a virtual clock when Compass asks for one,
because arrival is stamped in this process and first-token in the engine core,
and a TTFT that subtracts one from the other needs both readings from the same
clock. The clock is process-wide, so installing it is a loan, not a gift: an
ordinary engine constructed afterwards in the same process would otherwise
stamp every arrival at a frozen instant and report a wrong TTFT rather than a
missing one.

Covered here: a Compass engine that closes, a constructor that raises, and two
overlapping engines -- closed in reverse order of construction, which is the
supported lifetime, and closed out of order, which is not and says so.

The close cases go through the real `close()`. `_restore_clock` is only what
close calls; a test that exercised the helper alone would pass on an engine
whose close never called it.
"""

import pytest

from atom.model_engine.llm_engine import (
    LLMEngine,
    _install_compass_clock,
    _retired_clocks,
)
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
    _retired_clocks.clear()
    yield
    reset_clock()
    _retired_clocks.clear()


def _engine(config):
    """An engine carrying just the clock state, installed as __init__ does."""
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
    engine.close()
    assert get_clock() is before
    assert isinstance(get_clock(), WallClock)


def test_close_is_idempotent():
    engine = _engine(_Config(_Compass()))
    engine.close()
    engine.close()
    assert isinstance(get_clock(), WallClock)


def test_close_does_not_take_a_clock_someone_else_installed():
    """Restoring over a third party's clock is the same bug pointed the other
    way, so the engine only takes back a clock that is still its own."""
    engine = _engine(_Config(_Compass()))
    later = VirtualClock(epoch=7.0)
    set_clock(later)
    engine.close()
    assert get_clock() is later


def test_close_restores_even_when_the_core_shutdown_raises():
    class _Boom:
        def close(self):
            raise RuntimeError("core is already gone")

    engine = _engine(_Config(_Compass()))
    engine.core_mgr = _Boom()
    with pytest.raises(RuntimeError, match="core is already gone"):
        engine.close()
    assert isinstance(get_clock(), WallClock)


def test_nested_engines_closed_in_reverse_order_unwind_cleanly():
    """The supported lifetime: last constructed is first closed."""
    first = _engine(_Config(_Compass(epoch=100.0)))
    second = _engine(_Config(_Compass(epoch=200.0)))
    assert get_clock() is second._compass_clock

    second.close()
    assert get_clock() is first._compass_clock
    assert get_clock().time() == pytest.approx(100.0)

    first.close()
    assert isinstance(get_clock(), WallClock)


def test_engines_closed_out_of_order_do_not_resurrect_a_closed_clock(caplog):
    """The unsupported order, handled rather than hidden.

    `first` closes while `second` owns the clock, so it correctly leaves the
    clock alone. When `second` closes, the clock it saved is `first`'s -- an
    engine that is already gone. Reinstalling it would put the process on a
    dead engine's frozen time, which is the original bug wearing a different
    hat. So: wall time, and a warning naming the order that avoids it.
    """
    first = _engine(_Config(_Compass(epoch=100.0)))
    second = _engine(_Config(_Compass(epoch=200.0)))
    retired = first._compass_clock

    first.close()
    assert get_clock() is second._compass_clock

    with caplog.at_level("WARNING"):
        second.close()
    assert get_clock() is not retired
    assert isinstance(get_clock(), WallClock)
    assert "out of order" in caplog.text


def test_a_constructor_that_raises_leaves_no_clock_behind(monkeypatch):
    """Through the real __init__, with config construction succeeding.

    The failure is forced *after* the config exists, which is where both the
    old install site and the new one sit: a test whose config resolution failed
    first would pass against the buggy order too, since neither install had
    been reached. Nothing here touches the network -- `Config` is replaced by a
    subclass that skips resolution but is still the dataclass `__init__`
    introspects, and the tokenizer load is what raises.
    """
    import atom.model_engine.llm_engine as mod

    class _OfflineConfig(mod.Config):
        def __init__(self, model, **kwargs):  # no super().__init__: no lookup
            self.model = model
            self.trust_remote_code = False
            self.compass_config = _Compass()

    monkeypatch.setattr(mod, "Config", _OfflineConfig)

    # Installs unconditionally, so reaching it before the failure leaks a clock
    # and this test says so. That is the ordering property, stated directly.
    def _always_install(config):
        clock = VirtualClock(epoch=5.0)
        return clock, set_clock(clock)

    monkeypatch.setattr(mod, "_install_compass_clock", _always_install)

    reached = []

    def _no_tokenizer(*args, **kwargs):
        reached.append(True)
        raise OSError("tokenizer load failed in this test")

    monkeypatch.setattr(mod, "_load_tokenizer", _no_tokenizer)
    with pytest.raises(OSError, match="tokenizer load failed in this test"):
        LLMEngine("some-model")
    assert reached, "the constructor must get past config before failing"
    assert isinstance(get_clock(), WallClock)
