# SPDX-License-Identifier: MIT
"""The runner the bench starts on every worker: ATOM's recording runner with the
bench's window.

Kept apart from the package because importing `ModelRunner` needs a driver.
"""

from aiter.dist.parallel_state import get_world_group

from atom.compass.bench import routing
from atom.compass.bench.replay import seat_states
from atom.compass.bench.window import BenchWindow
from atom.compass.parity.runner import RecordingModelRunner


class BenchModelRunner(BenchWindow, RecordingModelRunner):
    """`RecordingModelRunner`, measuring a batch when asked for `bench_step`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calibrate_preload()

    def forward(self, batch):
        """`batch`'s forward, each row's state first seated at its cached
        context (`replay.seat_states`) on the stream, ahead of the preload and
        the window's entry event."""
        seat_states(self.attn_metadata_builder, batch)
        return super().forward(batch)

    def route_uniformly(self) -> int:
        """Every MoE layer, the drafter's too, routed by a uniform draw of as
        many tokens as a pass routes at data-parallel width 1, ATOM's MoE
        token capacity there
        (`model_ops/fused_moe/config.py::moe_kernel_token_capacity`); how many
        layers, a reply, which the runner manager waits for."""
        return routing.route_uniformly(
            self.config.compilation_config.static_forward_context,
            self.config.max_num_batched_tokens,
            self.device,
        )

    @staticmethod
    def bench_group():
        """The workers' CPU group; None for one worker.

        The bench refuses data and pipeline parallelism, so the world is this
        engine's workers.
        """
        group = get_world_group()
        return group.cpu_group if group.world_size > 1 else None
