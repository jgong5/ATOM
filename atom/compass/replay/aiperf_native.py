"""Ordinary AIPerf service execution with passive phase evidence collection."""
import asyncio
from contextlib import contextmanager
import threading
import time


@contextmanager
def observe_native_phases(service_config, *, require_terminal=False, drain_seconds=2.0):
    """Listen to the native event bus without issuing credits or commands."""
    from aiperf.common.enums import CommAddress, MessageType
    from aiperf.zmq.sub_client import ZMQSubClient

    address = service_config.comm_config.get_address(CommAddress.EVENT_BUS_PROXY_BACKEND)
    ready, stop = threading.Event(), threading.Event()
    messages, failures = [], []
    terminal = threading.Event()
    last_message = [time.monotonic()]

    async def listen():
        # A standalone subscriber avoids the service communication singleton;
        # SystemController still constructs and owns its normal service fleet.
        import zmq.asyncio

        context = zmq.asyncio.Context()
        subscriber = ZMQSubClient(address=address, bind=False)
        subscriber.context = context

        async def record(message):
            if (str(message.message_type) != "command"
                    or str(getattr(message, "command", "")) == "profile_cancel"):
                messages.append(message.model_dump(mode="json"))
                last_message[0] = time.monotonic()
                if (str(message.message_type) == "credits_complete"
                        or (str(message.message_type) == "credit_phase_complete"
                            and str(message.stats.phase) == "profiling")
                        or (str(message.message_type) == "command"
                            and str(getattr(message, "command", "")) == "profile_cancel")):
                    terminal.set()

        try:
            await subscriber.initialize()
            for topic in (
                MessageType.DATASET_CONFIGURED_NOTIFICATION, MessageType.CREDIT_PHASE_START,
                MessageType.CREDIT_PHASE_SENDING_COMPLETE, MessageType.CREDIT_PHASE_COMPLETE,
                MessageType.COMMAND, MessageType.CREDITS_COMPLETE,
            ):
                await subscriber.subscribe(topic, record)
            await subscriber.start()
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(.01)
            # The controller may return while terminal messages are queued in
            # this independent subscriber. Drain passively to a bounded quiet
            # point; no event here changes native credits or phase control.
            deadline = time.monotonic() + drain_seconds
            while time.monotonic() < deadline:
                if terminal.is_set() and time.monotonic() - last_message[0] >= .1:
                    break
                await asyncio.sleep(.01)
        finally:
            if not subscriber.was_stopped:
                await subscriber.stop()
            context.term()

    def run():
        try:
            asyncio.run(listen())
        except BaseException as exc:
            failures.append(exc)
        finally:
            ready.set()

    thread = threading.Thread(target=run, name="aiperf-phase-observer", daemon=True)
    thread.start()
    if not ready.wait(10):
        stop.set()
        thread.join(timeout=10)
        raise RuntimeError("native AIPerf phase observer did not initialize")
    if failures:
        raise RuntimeError("native AIPerf phase observer failed") from failures[0]
    try:
        yield messages
    finally:
        stop.set()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("native AIPerf phase observer did not stop")
        if failures:
            raise RuntimeError("native AIPerf phase observer failed") from failures[0]
        if require_terminal and not terminal.is_set():
            error = RuntimeError("native AIPerf observer lacks terminal phase/cancel evidence after bounded drain")
            error.phase_messages = messages
            raise error


def run_native_profile(user_config, service_config):
    """Invoke the exact ordinary CLI entry, retaining native ServiceConfig/ZMQ."""
    from aiperf.cli_runner import run_system_controller
    from aiperf.plugin.enums import CommunicationBackend

    if service_config.comm_config.comm_backend not in (
        CommunicationBackend.ZMQ_IPC, CommunicationBackend.ZMQ_TCP,
    ):
        raise ValueError("proper native replay requires ordinary AIPerf communication")
    with observe_native_phases(service_config, require_terminal=True) as messages:
        run_system_controller(user_config, service_config)
    return messages
