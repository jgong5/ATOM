# SPDX-License-Identifier: MIT
"""One protocol, carried two ways, and the evidence that it is still one.

A run that fits in one container reaches the clock by a direct call; a run split
across containers reaches it over a socket. The whole value of the first is that
it is evidence about the second, and that only holds while the two carry the
same frames to the same code. So the central test here is not that either works:
it is that the same script driven over each produces the same bytes.

What each test is defending:

* **The named result.** One fixed script, three participants, driven over each
  carriage. The `(participant, virtual time, event)` rows it produces are
  compared as bytes, not as numbers, so a difference of a floating-point digit
  or of ordering fails.
* **The stamps survive the cheap arrangement.** A direct call is exactly where
  someone would stop building the frame, because building it is the only cost
  the call has. Every request is recorded on the way through both carriages and
  the two sequences of `(kind, participant, virtual send time)` are compared.
  Then the frames themselves are compared, which is the stronger statement: the
  co-hosted carriage does not merely carry equivalent stamps, it carries the
  same bytes.
* **There is nowhere to put a shortcut.** A carrier takes bytes and returns
  bytes, and the rule is reachable through exactly one method that takes a
  frame. Handing the in-process carrier a message object is refused, which is
  what makes the first property structural rather than a matter of care.
* **A refusal is carried, not flattened.** An abort crosses either carriage as
  the same type, with its reason and its participant table, so a run that must
  stop stops the same way in both arrangements.
* **Two events in flight on one participant stay two.** The clock keeps every
  event a peer has placed on a participant rather than the earliest of them,
  and a carriage must not be where that becomes one again. Nothing here could
  collapse it -- what crosses is a participant's own declaration and an event's
  own timestamp, never the clock's horizon -- but the arrangement where a
  collapse would be invisible is the standalone one, so the case is driven over
  both, with a single-event control beside it.
* **Nothing a participant touches says where the clock is.** That claim is
  mechanical and lives with the package's other source guards, in
  `test_clock_lp_identity.py`, because it is a statement about the whole package
  rather than about the transport alone.

A socket is tested without a second container and without a GPU: the standalone
clock binds `127.0.0.1` on a port the kernel picks, runs its accept loop in a
thread of this process, and is reached over a real loopback connection carrying
real framed bytes. One test does spawn a separate interpreter, because "its own
process, started before the engines" is a claim about processes and a thread
cannot make it.
"""

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from atom.compass.clock import (
    BackdatedEvent,
    ClockAuthority,
    ClockDeadlock,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
)
from atom.compass.clock.transport import (
    BEFORE_THE_RUN,
    ClockService,
    ClockSession,
    InProcessCarrier,
    MalformedMessage,
    Message,
    MessageKind,
    carrier_for,
    connect,
    decode,
    encode,
    serve,
)

REPO = Path(__file__).resolve().parents[2]

TRAFFIC = LpId("traffic-source")
PREFILL = LpId("prefill")
DECODE = LpId("decode")

#: Floors chosen so the three participants bound each other unequally: the role
#: boundary is a thousand times tighter than the admission delay, so a grant
#: names a different peer depending on who is asking.
ADMISSION_SECONDS = 9.0e-3
ROLE_SECONDS = 1.0e-3

#: The two ways a clock is written down. A socket asks for any free port, so
#: nothing here depends on one being available, and neither name says anything
#: about the run -- which is the point: the same script is driven over each.
IN_PROCESS = "inproc:carried"
OVER_A_SOCKET = "tcp://127.0.0.1:0"
BOTH = (IN_PROCESS, OVER_A_SOCKET)


def _clock(floor=None):
    """Three participants, every ordered pair sized.

    With no floor given the pairs are sized unequally, so the peer that bounds a
    grant depends on who is asking. Given one, every pair is sized the same --
    and at zero that is the arrangement in which nobody may run ahead of anybody,
    which is what a run has to reach before it can stall.
    """
    registry = LpRegistry()
    for lp_id in (TRAFFIC, PREFILL, DECODE):
        registry.register(lp_id)
    matrix = LookaheadMatrix(registry)
    for source in (TRAFFIC, PREFILL, DECODE):
        for target in (TRAFFIC, PREFILL, DECODE):
            if source == target:
                continue
            role = TRAFFIC not in (source, target)
            declared = ROLE_SECONDS if role else ADMISSION_SECONDS
            matrix.declare(
                source,
                target,
                LinkClass.PREFILL_TO_DECODE if role else LinkClass.TRAFFIC_TO_ENGINE,
                declared if floor is None else floor,
            )
    return ClockAuthority(registry, matrix)


class _Recorded:
    """A carrier that keeps every frame that crossed it, in both directions.

    Wrapping rather than instrumenting: the carriage under test is unchanged,
    and what is recorded is exactly what it moved. It can only record bytes,
    which is the point -- a carriage that had stopped carrying frames would have
    nothing here to record.
    """

    def __init__(self, carrier, log):
        self._carrier = carrier
        self._log = log

    def exchange(self, frame):
        reply = self._carrier.exchange(frame)
        self._log.append((frame, reply))
        return reply

    def close(self):
        self._carrier.close()


def _row(lp_id, when, event):
    return f"{lp_id}\t{when:.9g}\t{event}"


def _scenario(endpoint):
    """Drive one fixed script and return its rows and the frames it moved.

    The script is sequential and always in the same order, so arrival order is
    held fixed and the only thing varying between two runs of it is the
    carriage. Rows are built from what came back over the carriage, never from
    the clock object, so a run against a clock in another process produces them
    the same way.
    """
    rows = []
    frames = []
    sessions = {}

    def note_grants(released):
        for grant in released:
            rows.append(
                _row(
                    grant.lp_id,
                    grant.advance_to,
                    f"granted from {grant.advance_from:.9g}, "
                    f"bound {grant.bound:.9g} by {grant.bound_from}",
                )
            )

    def take(lp_id):
        grant = sessions[lp_id].take_up_grant()
        rows.append(_row(lp_id, sessions[lp_id].now, f"took up {grant.seconds:.9g}"))

    def ask(lp_id, horizon):
        """Ask, note what was released, and take up every grant that came out.

        A grant issued to a peer has to be taken up too, or that peer is holding
        one the next time it asks and the clock refuses it. Returns False once
        nothing anywhere can move.
        """
        try:
            released = sessions[lp_id].request_advance(horizon)
        except ClockDeadlock as stall:
            rows.append(_row(lp_id, sessions[lp_id].now, "refused: ClockDeadlock"))
            rows.append(_row(lp_id, sessions[lp_id].now, stall.reason))
            return False
        rows.append(_row(lp_id, sessions[lp_id].now, f"asked, next {horizon:.9g}"))
        note_grants(released)
        for grant in released:
            take(grant.lp_id)
        return True

    def place(source, target, when):
        try:
            released = sessions[source].schedule_event(target, when)
        except BackdatedEvent as abort:
            rows.append(_row(source, sessions[source].now, "refused: BackdatedEvent"))
            rows.append(_row(source, sessions[source].now, abort.reason))
            return
        rows.append(
            _row(source, sessions[source].now, f"event on {target} at {when:.9g}")
        )
        note_grants(released)
        for grant in released:
            take(grant.lp_id)

    for lp_id in (DECODE, PREFILL, TRAFFIC):
        session = ClockSession(lp_id, _Recorded(carrier_for(endpoint), frames))
        session.attach()
        sessions[lp_id] = session
        rows.append(_row(lp_id, session.now, "attached"))

    ask(TRAFFIC, 20.0e-3)
    ask(PREFILL, 50.0e-3)
    ask(DECODE, math.inf)
    place(TRAFFIC, PREFILL, 40.0e-3)
    place(DECODE, TRAFFIC, 0.0)
    stalled = False
    for _ in range(4):
        for lp_id, horizon in (
            (DECODE, math.inf),
            (PREFILL, 40.0e-3),
            (TRAFFIC, 20.0e-3),
        ):
            stalled = not ask(lp_id, horizon)
            if stalled:
                break
        if stalled:
            break

    for session in sessions.values():
        session.close()
    return "\n".join(rows).encode("utf-8"), tuple(frames)


#: Two events on `decode`, placed by two peers before it has reached either,
#: and declared out of the order they will be reached in.
TWO_IN_FLIGHT = ((PREFILL, 30.0e-3), (TRAFFIC, 20.0e-3))


def _events_in_flight(endpoint, placed=TWO_IN_FLIGHT):
    """Events placed on one participant before it has reached any of them.

    The clock keeps what a participant has declared apart from what peers have
    placed on it, and it keeps the second as a list rather than as the earliest
    of them, because reaching the first of two would otherwise forget the
    second. That record is the clock's own and does not cross a carriage: a
    participant declares only the events it has seen, so a request carries one
    number and an event's timestamp and there is nothing in it that could be
    collapsed in transit. This drives the case over each carriage anyway,
    because the arrangement in which a collapse would be invisible is the
    standalone one -- the one the cheap tests are supposed to stand in for.

    Every floor at zero, so a grant lands exactly on the earliest event
    anywhere rather than one floor further on, and the two events are reached
    as two grants rather than as a creep.

    Returns the rows, the frames, and the times `decode` was granted.
    """
    rows = []
    frames = []
    reached = []
    sessions = {}
    for lp_id in (DECODE, PREFILL, TRAFFIC):
        sessions[lp_id] = ClockSession(lp_id, _Recorded(carrier_for(endpoint), frames))
        sessions[lp_id].attach()
        rows.append(_row(lp_id, sessions[lp_id].now, "attached"))

    for source, when in placed:
        sessions[source].schedule_event(DECODE, when)
        rows.append(
            _row(source, sessions[source].now, f"event on decode at {when:.9g}")
        )

    stalled = False
    for _ in range(6):
        for lp_id in (DECODE, PREFILL, TRAFFIC):
            try:
                released = sessions[lp_id].request_advance()
            except ClockDeadlock:
                rows.append(_row(lp_id, sessions[lp_id].now, "refused: ClockDeadlock"))
                stalled = True
                break
            rows.append(_row(lp_id, sessions[lp_id].now, "asked, next inf"))
            for grant in released:
                rows.append(
                    _row(
                        grant.lp_id,
                        grant.advance_to,
                        f"granted from {grant.advance_from:.9g}",
                    )
                )
                if grant.lp_id == DECODE:
                    reached.append(grant.advance_to)
                sessions[grant.lp_id].take_up_grant()
        if stalled:
            break

    for session in sessions.values():
        session.close()
    return "\n".join(rows).encode("utf-8"), tuple(frames), tuple(reached)


def _stamps(frames):
    """`(kind, participant, virtual send time)` for every frame, both ways."""
    return tuple(
        (str(message.kind), str(message.participant), message.sent_at)
        for pair in frames
        for message in (decode(pair[0]), decode(pair[1]))
    )


@pytest.fixture
def clocks():
    """Serve a clock in either arrangement, and take it down again.

    The socket arrangement runs its accept loop in a thread of this process and
    binds a port the kernel picks, so it needs no second container and no fixed
    port -- what the thread changes is who runs the loop. The framing, the
    frames and the rule are the ones a second container would meet.
    """
    served = []

    def serving(written, authority=None):
        served.append(serve(ClockService(authority or _clock()), written))
        return served[-1].endpoint

    yield serving
    for server in served:
        server.close()


# --- the named result --------------------------------------------------------


def test_the_same_scenario_over_both_carriages_is_byte_identical(clocks):
    co_hosted_clock, standalone_clock = _clock(), _clock()
    cheap, _ = _scenario(clocks(IN_PROCESS, co_hosted_clock))
    expensive, _ = _scenario(clocks(OVER_A_SOCKET, standalone_clock))
    assert hashlib.sha256(cheap).hexdigest() == hashlib.sha256(expensive).hexdigest(), (
        "the two carriages produced different runs:\n"
        + cheap.decode("utf-8")
        + "\n--- versus ---\n"
        + expensive.decode("utf-8")
    )
    assert cheap == expensive
    # Not vacuous: the script has to have done something, including being
    # refused, or two empty runs would agree.
    lines = cheap.decode("utf-8").splitlines()
    assert len(lines) > 20
    assert any("granted from" in line for line in lines)
    assert any("refused: BackdatedEvent" in line for line in lines)
    # The protocol's own cost is the same too, which it need not have been: the
    # count depends on arrival order, and the script fixes that.
    assert co_hosted_clock.grants_issued() == standalone_clock.grants_issued() > 0


def test_both_carriages_carry_the_same_stamps(clocks):
    _, cheap = _scenario(clocks(IN_PROCESS))
    _, expensive = _scenario(clocks(OVER_A_SOCKET))
    assert _stamps(cheap) == _stamps(expensive)
    # And the stamps are real rather than a constant that happens to match: only
    # the three attach requests are sent from outside simulated time, and every
    # other frame quotes a clock.
    outside = [stamp for stamp in _stamps(cheap) if stamp[2] == BEFORE_THE_RUN]
    assert len(outside) == 3
    assert all(kind == "attach" for kind, _, _ in outside)
    assert len({stamp[2] for stamp in _stamps(cheap)}) > 3


def test_two_events_in_flight_are_both_reached_over_either_carriage(clocks):
    # The clock keeps every event a peer has placed on a participant, not the
    # earliest of them, because reaching the first of two would otherwise
    # forget the second. Nothing on a carriage could collapse that record --
    # what crosses is a participant's own declaration and an event's own
    # timestamp, never the clock's horizon -- but the arrangement in which a
    # collapse would be invisible is the standalone one, so the case is driven
    # over both and compared as bytes.
    cheap, cheap_frames, cheap_reached = _events_in_flight(
        clocks(IN_PROCESS, _clock(floor=0.0))
    )
    expensive, expensive_frames, expensive_reached = _events_in_flight(
        clocks(OVER_A_SOCKET, _clock(floor=0.0))
    )
    assert cheap == expensive
    assert _stamps(cheap_frames) == _stamps(expensive_frames)
    assert [pair[0] for pair in cheap_frames] == [pair[0] for pair in expensive_frames]
    # Both events are reached, later one first placed, and neither is stepped
    # over: two grants landing exactly on the two timestamps.
    assert cheap_reached == expensive_reached
    assert cheap_reached == (pytest.approx(20.0e-3), pytest.approx(30.0e-3))
    # And the run ends by saying it cannot go on, rather than by going quiet.
    assert cheap.decode("utf-8").splitlines()[-1].endswith("refused: ClockDeadlock")


def test_one_event_in_flight_never_reaches_the_second_timestamp(clocks):
    # The control for the test above, varying one thing: the same script with
    # the 30 ms event not placed. `decode` then reaches 20 ms and runs past
    # 30 ms to the stall, so the second grant in that test is evidence that the
    # second event was carried and kept rather than an artefact of the script.
    _, _, reached = _events_in_flight(
        clocks(OVER_A_SOCKET, _clock(floor=0.0)), placed=(TWO_IN_FLIGHT[1],)
    )
    assert reached == (pytest.approx(20.0e-3),)


def test_the_co_hosted_carriage_puts_the_same_frames_on_the_wire(clocks):
    # The strongest form of the property, and the one that makes the others
    # hard to lose by accident: it is not that the direct call carries
    # equivalent information, it is that it carries the same bytes a socket
    # would have. A shortcut past the frame fails here first.
    _, cheap = _scenario(clocks(IN_PROCESS))
    _, expensive = _scenario(clocks(OVER_A_SOCKET))
    assert [pair[0] for pair in cheap] == [pair[0] for pair in expensive]
    assert [pair[1] for pair in cheap] == [pair[1] for pair in expensive]
    assert all(isinstance(pair[0], bytes) for pair in cheap)


# --- there is nowhere to put a shortcut --------------------------------------


def test_a_co_hosted_carrier_moves_bytes_and_not_a_message(clocks):
    # The prohibition, stated as a type. The cheap carriage is the one where
    # skipping the encoding costs nothing and saves everything it is there to
    # test, so the door is shut rather than watched.
    carrier = carrier_for(clocks(IN_PROCESS))
    assert isinstance(carrier, InProcessCarrier)
    with pytest.raises(MalformedMessage, match="never message objects"):
        carrier.exchange(Message(MessageKind.ATTACH, DECODE, 0.0))


def test_the_only_way_into_the_rule_is_a_frame():
    authority = _clock()
    public = sorted(name for name in vars(ClockService) if not name.startswith("_"))
    assert public == ["authority", "handle"], (
        f"the rule is reachable through {public}; a second entry point taking a "
        "request object is how the two arrangements stop being one"
    )
    # The one public reader is a reader: it hands back the clock, and the clock
    # is not a way to send it a request from a carriage.
    assert ClockService(authority).authority is authority


# --- a refusal is carried, not flattened -------------------------------------


@pytest.mark.parametrize("written", BOTH, ids=["co-hosted", "standalone"])
def test_an_abort_crosses_either_carriage_as_itself(written, clocks):
    traffic = connect(TRAFFIC, clocks(written))
    with pytest.raises(BackdatedEvent) as excinfo:
        traffic.schedule_event(DECODE, 0.0)
    abort = excinfo.value
    assert "earlier than" in abort.reason
    # The table is the whole value of an abort, and it has to survive the trip.
    assert "participant" in abort.table and "traffic-source" in abort.table
    assert "decode" in abort.table and "prefill" in abort.table
    # The refusal left the participant's clock where it was.
    assert traffic.now == 0.0
    traffic.close()


@pytest.mark.parametrize("written", BOTH, ids=["co-hosted", "standalone"])
def test_a_stalled_run_crosses_either_carriage_as_itself(written, clocks):
    # Every floor at zero, so nobody may run ahead of anybody and the run
    # reaches the state where no event exists anywhere. That aborts rather than
    # waiting to see whether it frees up, and the abort has to survive the trip
    # with its table: a stall reported as a closed connection would be a
    # timeout wearing a different hat.
    endpoint = clocks(written, _clock(floor=0.0))
    sessions = [connect(lp_id, endpoint) for lp_id in (DECODE, PREFILL, TRAFFIC)]
    assert sessions[0].request_advance() == ()
    assert sessions[1].request_advance() == ()
    with pytest.raises(ClockDeadlock) as excinfo:
        sessions[2].request_advance()
    assert "none knows of a future event" in excinfo.value.reason
    assert "traffic-source" in excinfo.value.table
    for session in sessions:
        session.close()


@pytest.mark.parametrize("written", BOTH, ids=["co-hosted", "standalone"])
def test_a_name_the_clock_does_not_hold_is_refused_at_the_first_message(
    written, clocks
):
    # Membership is fixed when the clock is built, so this can only ever be a
    # set-up mistake, and the cheapest place to find it is the first message
    # rather than a missing term in a minimum halfway through a run.
    with pytest.raises(KeyError, match="not a participant"):
        connect(LpId("pp-stage-7"), clocks(written))


# --- what a participant knows, and what it does not --------------------------


def test_a_session_starts_outside_simulated_time_and_learns_its_clock(clocks):
    session = ClockSession(DECODE, carrier_for(clocks(IN_PROCESS)))
    assert session.now == BEFORE_THE_RUN
    assert session.attach() == 0.0
    assert session.now == 0.0
    assert "decode" in repr(session)
    session.close()


def test_a_session_belongs_to_an_identity_rather_than_a_name(clocks):
    with pytest.raises(TypeError):
        ClockSession("decode", carrier_for(clocks(IN_PROCESS)))


def test_a_clock_above_another_is_reached_through_the_same_surface(clocks):
    # The arrangement with a clock between two others is not built here. What is
    # asserted is that nothing precludes it: such a clock answers its own
    # participants through one of these services and is itself a participant of
    # whatever is above it, through the session type a participant uses. Neither
    # side is told which it is talking to, and there is nothing on either to
    # find out with.
    above = clocks(OVER_A_SOCKET)
    below = clocks("inproc:below")
    upward = connect(PREFILL, above)
    downward = connect(DECODE, below)
    assert type(upward) is type(downward) is ClockSession
    assert upward.now == downward.now == 0.0
    # Both directions are live through the one surface.
    assert upward.request_advance(1.0)[0].lp_id == PREFILL
    assert downward.request_advance(1.0)[0].lp_id == DECODE
    upward.close()
    downward.close()


# --- resolving an endpoint ---------------------------------------------------


def test_a_standalone_clock_says_which_port_it_ended_up_on(clocks):
    there = clocks(OVER_A_SOCKET)
    assert there.startswith("tcp://127.0.0.1:")
    assert int(there.rsplit(":", 1)[1]) > 0
    # Asking for a free port and then quoting the request back would hand a
    # participant an endpoint nothing is listening on.
    assert not there.endswith(":0")


def test_nothing_is_reachable_in_this_process_before_it_is_served():
    with pytest.raises(KeyError, match="no clock is served"):
        carrier_for("inproc:not-started")


def test_two_clocks_at_one_endpoint_are_refused():
    first = serve(ClockService(_clock()), "inproc:twice")
    try:
        with pytest.raises(ValueError, match="already has a clock"):
            serve(ClockService(_clock()), "inproc:twice")
    finally:
        first.close()
    # Closing releases the name, so a run can be set up again in one process.
    again = serve(ClockService(_clock()), "inproc:twice")
    again.close()


@pytest.mark.parametrize(
    "written, message",
    [
        ("clock", "names no carriage"),
        ("zmq:clock", "which nothing here carries"),
        ("tcp:/127.0.0.1:9", "is not //<host>:<port>"),
        ("tcp://127.0.0.1", "is not //<host>:<port>"),
        ("tcp://127.0.0.1:nine", "is not a port number"),
    ],
)
def test_an_endpoint_that_reaches_nothing_is_refused_rather_than_guessed(
    written, message
):
    with pytest.raises(ValueError, match=message):
        carrier_for(written)


def test_an_endpoint_is_written_down_rather_than_assembled():
    with pytest.raises(TypeError, match="an endpoint is a str"):
        carrier_for(("127.0.0.1", 9))


# --- the wire format ---------------------------------------------------------


def test_a_message_survives_the_round_trip_it_is_built_for(clocks):
    session = connect(TRAFFIC, clocks(IN_PROCESS))
    released = session.request_advance(math.inf)
    original = Message(
        MessageKind.GRANTS,
        TRAFFIC,
        0.0,
        target=DECODE,
        when=math.inf,
        grants=released,
        detail=("ClockDeadlock", "reason", "table"),
    )
    assert decode(encode(original)) == original
    session.close()


def test_a_message_encodes_the_same_bytes_every_time():
    # Two arrangements are compared by comparing their traffic, which only works
    # if the traffic is a function of the message and nothing else.
    message = Message(MessageKind.ADVANCE, DECODE, 1.0, when=math.inf)
    assert encode(message) == encode(message)
    assert b'"sent_at":1.0' in encode(message)
    assert b'"when":Infinity' in encode(message)


@pytest.mark.parametrize(
    "frame", [b"", b"{}", b'{"kind":"gossip"}', b"\xff\xfe", "a string"]
)
def test_a_frame_that_is_not_a_message_is_refused(frame):
    with pytest.raises(MalformedMessage):
        decode(frame)


def test_a_reply_is_not_something_a_participant_may_send():
    service = ClockService(_clock())
    reply = decode(service.handle(encode(Message(MessageKind.CLOCK, DECODE, 0.0))))
    assert reply.kind is MessageKind.REFUSAL
    assert reply.detail[0] == "ValueError"
    assert "not something a participant may ask for" in reply.detail[1]


# --- a clock in a process of its own ------------------------------------------

CHILD = """
import sys
import atom
from atom.compass.clock import (
    ClockAuthority, LinkClass, LookaheadMatrix, LpId, LpRegistry
)
from atom.compass.clock.transport import ClockService, serve

registry = LpRegistry()
names = [LpId(name) for name in ("decode", "prefill", "traffic-source")]
for lp_id in names:
    registry.register(lp_id)
matrix = LookaheadMatrix(registry)
for source in names:
    for target in names:
        if source != target:
            matrix.declare(source, target, LinkClass.PREFILL_TO_DECODE, 1.0e-3)

server = serve(ClockService(ClockAuthority(registry, matrix)), "tcp://127.0.0.1:0")
print(atom.__file__, flush=True)
print(server.endpoint, flush=True)
sys.stdin.readline()
"""


def test_a_standalone_clock_runs_in_a_process_of_its_own():
    # The thread-hosted fixture above covers the socket; this covers the claim
    # the thread cannot make, which is that the clock is startable on its own
    # before anything that uses it exists. Nothing but the endpoint crosses
    # between the two processes.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        resolved = child.stdout.readline().strip()
        endpoint = child.stdout.readline().strip()
        assert resolved.startswith(str(REPO)), (
            f"the clock process resolved atom at {resolved!r}, not under {REPO}; "
            f"stderr: {child.stderr.read()}"
        )
        assert endpoint.startswith("tcp://127.0.0.1:"), child.stderr.read()
        decode_session = connect(DECODE, endpoint)
        prefill = connect(PREFILL, endpoint)
        traffic = connect(TRAFFIC, endpoint)
        assert decode_session.now == 0.0
        granted = decode_session.request_advance(5.0)
        assert granted and granted[0].lp_id == DECODE
        assert granted[0].advance_to == pytest.approx(1.0e-3)
        for session in (decode_session, prefill, traffic):
            session.close()
    finally:
        child.stdin.close()
        child.terminate()
        child.wait(timeout=60)
