# SPDX-License-Identifier: MIT
"""The traffic LP publishes every stamp of a burst past ZMQ's high-water mark
to a worker that reads none of it until the burst is out: a PUB socket drops
what passes its mark (1000 messages a subscriber by default), and a worker
whose stamp is dropped never sends its request, so the run waits for ever
(#3809's 3P4D C1228 run hung at the agentx warmup's cache-pressure phase, which
sends a request a lane at one instant). Skips as ``test_harness_traffic_lp.py``
does."""

# It skips this module as its own where compass-harness is not installed.
from test_harness_traffic_lp import ScriptedRuntime, TrafficLP, _key, _run

# isort: split
import zmq
import zmq.asyncio
from compass_harness.transport import addresses, stamp_subscriber

#: Far past both ends' default marks and the socket's own buffer.
BURST = 20000


def test_a_burst_of_stamps_reaches_a_worker_that_reads_it_late(tmp_path):
    async def main():
        prefix = str(tmp_path / "lp")
        traffic = TrafficLP(ScriptedRuntime([]), prefix)
        # A worker's socket, as its transport connects it.
        sub = stamp_subscriber(zmq.asyncio.Context.instance(), addresses(prefix)[0])
        await traffic.subscribed(1)
        for num in range(BURST):
            traffic.send(_key(num))
        traffic._stamp_sends()  # the hold's stamping of one instant's sends
        got = []
        while len(got) < BURST and await sub.poll(2000):
            got.append((await sub.recv_pyobj())[0])
        sub.close(linger=0)
        traffic.done.cancel()
        return got

    got = _run(main())
    assert len(got) == BURST
    assert sorted(got) == sorted(_key(num) for num in range(BURST))
