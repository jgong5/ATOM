# SPDX-License-Identifier: MIT
"""Drive ATOM's own `ModelRunner` under the `04` D18 capture mechanism.

Why the runner and not the model object: the operators a forward dispatches
depend on the *metadata* (`ForwardContext.attn_metadata`, `kv_cache_data`,
`Context`), and every branch that reads it lives in ATOM. Hand-building that
metadata would be a reimplementation of `ModelRunner.prepare_inputs` and the
attention builders -- ~1.5k lines whose agreement with ATOM nothing would
check. Principle 1 says reuse; so the capture subclasses `ModelRunner` and
substitutes the smallest set of behaviours that genuinely need hardware.

Every substitution is named in `SUBSTITUTIONS` and copied into the run record,
because a substitution is the part of the inventory that is *not* ATOM's.
"""

from __future__ import annotations

import torch

from atom.compass.capture.fake_trace import CaptureRefusal, init_single_rank_group

SUBSTITUTIONS = {
    "_setup_device_and_distributed": (
        "ModelRunner calls aiter `init_dist_env(backend='nccl')`, which needs "
        "tp_world_size real devices and an RCCL rendezvous. Replaced by one "
        "gloo rank plus ATOM's own `apply_simulated_tp` -- the same mechanism "
        "ATOM ships for running a logical TP width on fewer ranks."
    ),
    "_build_and_load_model": (
        "the base method calls `load_model(...)` which opens the checkpoint. "
        "`02` D10.1: construction only, no checkpoint bytes. `load_dummy` is "
        "not enough on its own because the loader still walks the safetensors "
        "index to decide what to skip."
    ),
    "_maybe_warmup": (
        "warmup IS the traced forward here, so it must run inside the "
        "recorder rather than inside `__init__`. Deferred, not dropped."
    ),
}


class CaptureModelRunner:
    """Factory, not a class: the subclass is built at call time so that
    importing this module does not import `model_runner` (which pulls aiter's
    kernel modules and reads device properties at import)."""

    @staticmethod
    def build(config, fake_mode, device: str = "cuda:0"):
        from atom.distributed.simulated_tp import apply_simulated_tp
        from atom.model_engine.model_runner import (
            ModelRunner,
            support_model_arch_dict,
        )
        from atom.utils import resolve_obj_by_qualname

        class _Capture(ModelRunner):
            def _setup_device_and_distributed(self, rank, cfg):
                init_single_rank_group()
                # `build_config` already applied it; the marker attribute is
                # `simulated_tp_physical_world_size`, set last by _patch_group.
                from aiter.dist.parallel_state import get_tp_group

                if not hasattr(get_tp_group(), "simulated_tp_physical_world_size"):
                    apply_simulated_tp(cfg)
                self.device = torch.device(device)

            def _build_and_load_model(self, model_class):
                with fake_mode:
                    self.model = model_class(self.config)
                torch.set_default_device(None)

            def _maybe_warmup(self):
                return

        arch = config.hf_config.architectures[0]
        if arch not in support_model_arch_dict:
            raise CaptureRefusal(f"{arch} is not in ATOM's support_model_arch_dict")
        model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])
        runner = _Capture(0, config)
        return runner, model_class
