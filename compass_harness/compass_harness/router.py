# SPDX-License-Identifier: MIT
"""aiperf's credit router with the traffic LP on both directions.

Every credit leaves through ``send_credit`` (``CreditIssuer._issue_credit_internal``
is its one call site), so the stamp is taken there; every return reaches the
strategy only through the return callback, so the hold wraps it. aiperf's
cancel of the requests in flight goes through the traffic LP first. The
controller hears when the traffic LP is done, or the error that failed it.
"""

import asyncio

from aiperf.common.enums import CommAddress
from aiperf.common.messages import BaseServiceErrorMessage, ProfileCancelCommand
from aiperf.common.models.error_models import ErrorDetails
from aiperf.credit.sticky_router import StickyCreditRouter

from compass_harness.controller import TrafficDone
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.traffic_lp import TrafficLP
from compass_harness.transport import credit_key


class CompassCreditRouter(StickyCreditRouter):
    def __init__(self, *args, service_id: str, **kwargs) -> None:
        super().__init__(*args, service_id=service_id, **kwargs)
        self.service_id = service_id
        # Tells the controller when the traffic LP is done or fails.
        self._bus = self.comms.create_pub_client(CommAddress.EVENT_BUS_PROXY_FRONTEND)
        self.traffic = TrafficLP.from_env()
        ClockPacedLoopScheduler.clock = self.traffic
        self.traffic.done.add_done_callback(self._on_finish)

    async def send_credit(self, credit) -> None:
        self.traffic.send(credit_key(credit.phase, credit.phase_index, credit.id))
        await super().send_credit(credit)

    def set_return_callback(self, callback) -> None:
        super().set_return_callback(self.traffic.hold(callback))

    async def wait_for_workers(self, timeout: float) -> None:
        await super().wait_for_workers(timeout)
        await self.traffic.subscribed(len(self._workers))

    async def cancel_all_credits(self) -> None:
        """Close the requests in flight on the clock first, then cancel upstream:
        each one's stamped cancel is written before its worker drops it."""
        if not self.traffic.done.done():
            await self.traffic.cancel()
        await super().cancel_all_credits()

    def mark_credits_complete(self) -> None:
        super().mark_credits_complete()
        self.traffic.finish()

    def _on_finish(self, done: asyncio.Future) -> None:
        if done.cancelled():
            return
        if done.exception() is None:
            # The controller's shutdown waits for this: until the +inf grant the
            # loop was blocked in the traffic LP's last clock wait.
            asyncio.ensure_future(
                self._bus.publish(TrafficDone(service_id=self.service_id))
            )
        else:
            # The traffic LP failed: a request was open at the +inf grant, or one
            # failed mid-stream before it.
            self.exception(f"traffic LP: {done.exception()!r}")
            asyncio.ensure_future(self._fail(done.exception()))

    async def _fail(self, error: BaseException) -> None:
        """Fail the run in aiperf, then stop the loop, which fails this service.

        The controller records the error and exits non-zero once the records
        manager has finished on the cancel; nothing else would tell it, since
        it never watches a service process after registration."""
        try:
            await self._bus.publish(
                BaseServiceErrorMessage(
                    service_id=self.service_id, error=ErrorDetails.from_exception(error)
                )
            )
            await self._bus.publish(
                ProfileCancelCommand(
                    service_id=self.service_id, reason_detail=f"traffic LP: {error!r}"
                )
            )
        finally:
            asyncio.get_running_loop().stop()
