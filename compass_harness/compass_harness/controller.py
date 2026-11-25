# SPDX-License-Identifier: MIT
"""aiperf's system controller, which shuts down only after the traffic LP's ``+inf`` grant.

aiperf shuts down once every result domain has joined, and kills each service
still alive shortly after. The timing manager's event loop is blocked in the
traffic LP's last clock wait until that grant, so it never reads the shutdown,
and a kill before the grant closes its clock connection and ends the run.
Here the timing manager also joins a ``traffic`` domain, which completes on
the `TrafficDone` the router publishes once the traffic LP is done. A failing
traffic LP's service error unregisters it, as it does any service's.
"""

from aiperf.common.hooks import on_message
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.types import MessageTypeT
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType

TRAFFIC = "traffic"
TRAFFIC_DONE = "compass_traffic_done"


class TrafficDone(BaseServiceMessage):
    """The traffic LP had its ``+inf`` grant with every final response in."""

    message_type: MessageTypeT = TRAFFIC_DONE


class CompassSystemController(SystemController):
    async def _profile_configure_all_services(self) -> None:
        # Every required service has registered by now.
        for info in self.service_manager.service_map[ServiceType.TIMING_MANAGER]:
            self._result_join_coordinator.register(TRAFFIC, info.service_id)
        await super()._profile_configure_all_services()

    @on_message(TRAFFIC_DONE)
    async def _on_traffic_done(self, message: TrafficDone) -> None:
        self._result_join_coordinator.complete(TRAFFIC, message.service_id)
        await self._check_and_trigger_shutdown()
