"""Passive observation uses native ZMQ messages without controlling credits."""
import asyncio
import time

import pytest

pytest.importorskip("aiperf")


def test_passive_subscriber_records_actual_native_phase_message(tmp_path):
    from aiperf.common.config import ServiceConfig, ZMQIPCConfig
    from aiperf.common.enums import CommAddress, CreditPhase
    from aiperf.common.models import CreditPhaseStats
    from aiperf.credit.messages import CreditPhaseStartMessage
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.zmq.pub_client import ZMQPubClient
    from atom.compass.replay.aiperf_native import observe_native_phases

    ipc = tmp_path / "ipc"
    ipc.mkdir()
    services = ServiceConfig(zmq_ipc=ZMQIPCConfig(path=ipc))
    message = CreditPhaseStartMessage(
        service_id="ordinary-timing",
        config=CreditPhaseConfig(phase=CreditPhase.PROFILING, timing_mode=TimingMode.AGENTIC_REPLAY,
                                 concurrency=1, expected_duration_sec=900),
        stats=CreditPhaseStats(phase=CreditPhase.PROFILING, start_ns=time.time_ns()))
    async def send():
        publisher = ZMQPubClient(
            address=services.comm_config.get_address(CommAddress.EVENT_BUS_PROXY_BACKEND), bind=True)
        await publisher.initialize_and_start()
        try:
            with observe_native_phases(services) as observed:
                # Native PUB/SUB needs its subscription handshake before publish.
                await asyncio.sleep(.1)
                await publisher.publish(message)
                deadline = time.monotonic() + 2
                while not observed and time.monotonic() < deadline:
                    await asyncio.sleep(.01)
            assert observed == [message.model_dump(mode="json")]
        finally:
            await publisher.stop()
    asyncio.run(send())

def test_terminal_published_during_teardown_is_drained(tmp_path):
    import threading
    from aiperf.common.config import ServiceConfig, ZMQIPCConfig
    from aiperf.common.enums import CommAddress
    from aiperf.credit.messages import CreditsCompleteMessage
    from aiperf.zmq.pub_client import ZMQPubClient
    from atom.compass.replay.aiperf_native import observe_native_phases

    ipc = tmp_path / "late_ipc"
    ipc.mkdir()
    services = ServiceConfig(zmq_ipc=ZMQIPCConfig(path=ipc))
    ready, publish, done = threading.Event(), threading.Event(), threading.Event()
    errors = []

    async def send():
        import zmq.asyncio
        context = zmq.asyncio.Context()
        publisher = ZMQPubClient(
            address=services.comm_config.get_address(CommAddress.EVENT_BUS_PROXY_BACKEND), bind=True)
        publisher.context = context
        try:
            await publisher.initialize_and_start()
            ready.set()
            while not publish.is_set():
                await asyncio.sleep(.01)
            # Main thread has entered observer teardown before this arrives.
            await asyncio.sleep(.05)
            await publisher.publish(CreditsCompleteMessage(service_id="native"))
            while not done.is_set():
                await asyncio.sleep(.01)
        finally:
            await publisher.stop()
            context.term()

    def run():
        try:
            asyncio.run(send())
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert ready.wait(2)
        with observe_native_phases(services, require_terminal=True, drain_seconds=1) as observed:
            time.sleep(.1)
            publish.set()
        assert [m["message_type"] for m in observed] == ["credits_complete"]
    finally:
        done.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert not errors


def test_missing_terminal_evidence_fails_after_bounded_drain(tmp_path):
    from aiperf.common.config import ServiceConfig, ZMQIPCConfig
    from atom.compass.replay.aiperf_native import observe_native_phases
    ipc = tmp_path / "empty_ipc"
    ipc.mkdir()
    services = ServiceConfig(zmq_ipc=ZMQIPCConfig(path=ipc))
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="lacks terminal"):
        with observe_native_phases(services, require_terminal=True, drain_seconds=.05):
            pass
    assert time.monotonic() - start < 1
