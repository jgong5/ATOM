"""A model runner with no model and no device.

``CompassModelRunner`` in ``predict`` mode already computes its forward from a
cost oracle rather than from weights -- but it is a ``ModelRunner``, and
``ModelRunner.__init__`` sets up a device, initialises the distributed group,
picks an attention backend and loads real weights before it returns. Under
``predict`` none of that is read by the forward pass. It is paid for anyway.

This runner keeps the predict path exactly as it is -- inherited, not copied,
so a change to how a step is priced or how its output is deferred lands here
too -- and replaces only the startup. The RPCs the engine makes during startup
(``get_num_blocks``, ``allocate_kv_cache``, ``capture_cudagraph``) are answered
from a **target record**: what those calls returned when the configuration was
last measured on real hardware, or what a memory model says they would return.

That record is the honest boundary of a GPU-free replay. Everything downstream
of it -- how many blocks the scheduler may hand out, what the block manager
does when it runs out, which requests are admitted, when a sequence finishes --
is ATOM's own code operating on those numbers.
"""

from __future__ import annotations

import hashlib
import logging

import json
import os
from typing import Optional

from atom.compass.runtime.predict import CompassPredictMixin

logger = logging.getLogger(__name__)

__all__ = ["ReplayModelRunner", "TargetRecord"]

TARGET_VERSION = 1


class TargetRecord:
    """What a device said about a configuration, kept so it need not say it again.

    Deliberately thin: it holds the reply to each startup RPC and the
    configuration those replies belong to. It does not hold a model, weights, or
    anything that would let this become a second implementation of sizing.
    """

    def __init__(self, blob: dict, source: str, *, sha256: Optional[str] = None,
                 nbytes: Optional[int] = None) -> None:
        self.source = source
        #: Digested by `load` from the bytes it parsed, and never recomputed by
        #: re-opening the path. `replay_target` is outside the hashed oracle
        #: options and it names a *file*: the name is not the input, and a
        #: digest taken at report time attests to whatever is on disk then.
        self.sha256 = sha256
        self.bytes = nbytes
        self.version = int(blob.get("version") or 0)
        self.blocks: dict = dict(blob.get("blocks") or {})
        self.config: dict = dict(blob.get("config") or {})
        self.graph: dict = dict(blob.get("graph") or {})
        # Optional on purpose, and not guarded by the version above. The
        # version guard is for fields a replay would silently *misread*; a
        # missing block that raises by name when something wants it is not
        # silent, and bumping the version would force a re-capture on runs
        # that never look at this. `atom.compass.replay.bootstrap` is what
        # wants it, and it says so when it is absent.
        self.hardware: dict = dict(blob.get("hardware") or {})

    @classmethod
    def load(cls, path: str) -> "TargetRecord":
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"ATOMCompass: no replay target at {path!r}. A GPU-free replay "
                f"needs the startup answers a device would have given -- "
                f"capture them with --compass-replay-target-out on a run of "
                f"this configuration, or model them."
            )
        with open(path, "rb") as fh:
            raw = fh.read()
        blob = json.loads(raw.decode("utf-8"))
        record = cls(blob, path, sha256=hashlib.sha256(raw).hexdigest(),
                     nbytes=len(raw))
        if record.version != TARGET_VERSION:
            raise ValueError(
                f"ATOMCompass: {path} is a version {record.version} replay "
                f"target and this build writes version {TARGET_VERSION}. "
                f"Re-capture rather than reinterpret: the fields a replay "
                f"depends on are exactly the ones that move."
            )
        if not record.blocks.get("num_kvcache_blocks"):
            raise ValueError(
                f"ATOMCompass: {path} records no KV block count, so there is "
                f"nothing for the block manager to allocate from. A target "
                f"captured from a run that failed to size is not a target."
            )
        return record

    def disagreements(self, config) -> list[str]:
        """Where this record's configuration differs from the one being replayed.

        Reported rather than enforced. A replay of a *different* configuration
        is exactly what the PoC eventually wants -- predicting TP=4 from a TP=1
        capture is the point -- so the differences are surfaced for the run to
        declare, not treated as corruption. What must not happen is a difference
        going unnoticed and the result being read as a like-for-like replay.
        """
        checks = {
            "model": str(getattr(config, "model", "")),
            "tensor_parallel_size": int(
                getattr(config, "tensor_parallel_size", 1) or 1),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
            "gpu_memory_utilization": float(
                getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
        }
        out = []
        for key, now in checks.items():
            was = self.config.get(key)
            if was is not None and was != now:
                out.append(f"{key}: captured {was!r}, replaying {now!r}")
        return out


class ReplayModelRunner(CompassPredictMixin):
    """Predict mode, with the device startup replaced by a captured record."""

    def __init__(self, rank: int, config, *args, **kwargs) -> None:
        # No `super().__init__` on purpose. `ModelRunner.__init__` selects a
        # device, joins a distributed group, builds an attention backend and
        # loads weights; under predict none of it is read, and requiring it is
        # what ties a simulated run to the hardware it is simulating.
        self.rank = int(rank)
        self.config = config
        self.device = None
        self.model = None

        compass = getattr(config, "compass_config", None)
        mode = getattr(compass, "mode", None)
        if mode != "predict":
            raise ValueError(
                f"ATOMCompass: GPU-free replay only makes sense in predict "
                f"mode, and this run is in {mode!r}. Trace and measure exist to "
                f"observe a real forward; there is none here to observe."
            )
        self.target = TargetRecord.load(getattr(compass, "replay_target", ""))
        self._check_parallel_contract(config)
        #: Filled in by `get_num_blocks`: the sealed record of the files that
        #: actually produced the capacity this run plans from. Kept on the
        #: runner rather than folded into the RPC reply, because the wire form
        #: is the engine's and provenance is not part of it.
        self.compass_loaded_inputs: Optional[dict] = None
        differences = self.target.disagreements(config)
        if differences:
            logger.warning(
                "ATOMCompass WARNING: replaying a configuration the target "
                "was not captured from (%s). The block count and pool layout "
                "below are the captured ones; if the difference affects them, "
                "this replay is sizing the wrong deployment.",
                "; ".join(differences))

        # `_capture_bucket` asks these two directly. A replay pays no graph
        # launch, but it must still report the rung a real run would replay at,
        # because that is what the cost table is keyed on.
        self.enforce_eager = bool(getattr(config, "enforce_eager", False))
        self.capture_sizes = list(self.target.graph.get("capture_sizes") or [])

        self._init_compass_state()
        logger.info(
            "ATOMCompass: GPU-free replay of a logical TP%d deployment on %d "
            "executor, target %s (%d KV blocks, %d capture sizes), no device "
            "acquired",
            self.logical_tp, self.physical_executors, self.target.source,
            self.target.blocks.get("num_kvcache_blocks", 0),
            len(self.capture_sizes))

    # -- the parallelism contract --------------------------------------------

    def _check_parallel_contract(self, config) -> None:
        """Separate the width being predicted from the count doing the work.

        Three numbers, and conflating any two of them is a different wrong
        answer:

        `logical_tp` is how wide the deployment under evaluation is. It governs
        the topology the oracle is asked about, the rank coordinates its
        artifacts are keyed on, where the LM head sits, and the block and pool
        specification -- everything except who runs the step.

        `physical_executors` is how many processes hold a runner, and in a
        GPU-free replay it is one: there is no device for a second to occupy
        and no forward for it to run. That one executor is *rank 0 of the
        logical group*, not a TP1 deployment. Its step costs what the oracle
        says the group's rank-0 work plus the modelled collectives cost.

        The record's own `tensor_parallel_size` is a third number, and it is
        the one enforced here. The startup answers -- block count, pool
        entries, state runtime -- were measured at *that* width, and block
        count is close to linear in it. Replaying them under a different
        logical width is not a transfer prediction, it is the wrong deployment
        sized by the wrong record, so it refuses rather than warns. Predicting
        a wider configuration is done by deriving a record at that width from
        the memory model and replaying against that; the difference is that the
        derived record's numbers are attributable to the width they claim.
        """
        self.logical_tp = int(getattr(config, "tensor_parallel_size", 1) or 1)
        # Not read from the config: asserted. `Config.tp_world_size` collapses
        # to one under a replay and `LocalProcManager` refuses anything else,
        # so this records the contract rather than discovering it.
        self.physical_executors = 1

        pp = int(getattr(config, "pipeline_parallel_size", 1) or 1)
        if pp > 1:
            raise ValueError(
                f"ATOMCompass: a GPU-free replay has one executor and this "
                f"deployment asks for {pp} pipeline stages. Stages are not "
                f"ranks of one step -- each runs a different slice of the "
                f"model and the schedule between them is the thing a pipeline "
                f"configuration is evaluated for. Nothing here models it."
            )

        captured_tp = int(self.target.config.get("tensor_parallel_size") or 0)
        if captured_tp and captured_tp != self.logical_tp:
            raise ValueError(
                f"ATOMCompass: {self.target.source} records the startup "
                f"answers of a TP{captured_tp} deployment and this replay is "
                f"of a logical TP{self.logical_tp} one. "
                f"{self.target.blocks.get('num_kvcache_blocks', 0)} KV blocks "
                f"and the pool entries beside them were sized at TP"
                f"{captured_tp}; handing them to a TP{self.logical_tp} "
                f"scheduler would let it admit a workload the target cannot "
                f"hold, and the resulting throughput would be read as a "
                f"prediction. Derive a target record at TP{self.logical_tp} "
                f"from the memory model and replay against that, so its "
                f"numbers are attributable to the width they claim."
            )

    # -- the startup RPCs, answered from the record ---------------------------

    def get_num_blocks(self) -> dict:
        """What the device said -- unless this run asked what a model says.

        Without ``--compass-memory-model`` the recorded reply is returned
        verbatim, including the state-runtime wire form: the engine rebuilds
        its own ``StateRuntime`` from it and the block manager plans from
        ``pool_entries``, so passing it through is what keeps the replay
        running ATOM's arithmetic rather than a copy of it.

        With a profile the pool is *sized* here instead. The two flags already
        composed on the command line and the reply was still the capture: a
        scheduler sized by a measurement of the configuration the profile was
        meant to replace, read as a forecast of the one being replayed. Only
        the device-backed runner consumed a profile, and a GPU-free replay is
        the run that has no device.

        No path leads from a refusal back to ``self.target.blocks``. A run that
        asked to be modelled and cannot be has to stop, because the number it
        would otherwise serve is the one it was told not to use.

        Either way the files that produced the answer are recorded in
        ``compass_loaded_inputs``, digested where they were read.
        """
        from atom.compass.core.memory_blocks import (
            LoadedInputs, derived_block_info)

        inputs = LoadedInputs()
        # Not a read of our own: the target was digested when it was parsed, in
        # `TargetRecord.load`, and re-opening it here would attest to the file
        # as it is now rather than as it was used.
        inputs.note("replay_target", path=self.target.source,
                    abspath=os.path.abspath(self.target.source),
                    sha256=self.target.sha256, bytes=self.target.bytes)

        path = (getattr(self._compass_config, "memory_model", "") or "").strip()
        try:
            if not path:
                blocks = dict(self.target.blocks)
            else:
                blocks = derived_block_info(
                    path, self.config,
                    state_runtime=self.target.blocks.get("state_runtime"),
                    captured=self.target.blocks,
                    inputs=inputs)
        finally:
            # Published on the refusal path too. What a run that stopped had
            # already read is evidence about the run that stopped.
            inputs.seal()
            self.compass_loaded_inputs = inputs.manifest(
                **self._capacity_context(path))
        self.compass_loaded_inputs["num_kvcache_blocks"] = int(
            blocks.get("num_kvcache_blocks") or 0)
        return blocks

    def _capacity_context(self, memory_model: str) -> dict:
        """The deployment terms a block count cannot be checked without.

        The digests say which bytes were read; these say what they were read
        *for*. Both are needed to say a number was founded: the same profile
        sizes a different pool at another width, block size or utilization.
        """
        config = self.config
        return {
            "modelled": bool(memory_model),
            "memory_model": memory_model or None,
            "deployment": {
                "model": str(getattr(config, "model", "")),
                "tensor_parallel_size": int(
                    getattr(config, "tensor_parallel_size", 1) or 1),
                "pipeline_parallel_size": int(
                    getattr(config, "pipeline_parallel_size", 1) or 1),
                "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
                "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
                "max_num_batched_tokens": int(
                    getattr(config, "max_num_batched_tokens", 0) or 0),
                "gpu_memory_utilization": float(
                    getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
                "kv_cache_block_size": int(
                    getattr(config, "kv_cache_block_size", 0) or 0),
                "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
                "enforce_eager": bool(getattr(config, "enforce_eager", False)),
            },
        }

    def allocate_kv_cache(self, num_kvcache_blocks) -> bool:
        """There is no cache to allocate; the accounting for it is real.

        The scheduler and block manager still hand out, split and free these
        blocks. What does not happen is the tensor being made -- which is the
        whole memory footprint of the deployment and the reason a replay fits on
        a machine the deployment does not.
        """
        logger.info("ATOMCompass: %d KV blocks accounted for, none allocated",
                    int(num_kvcache_blocks))
        return True

    def capture_cudagraph(self):
        """Nothing to capture. Reports the captured run's cost, not zero.

        The engine logs this and moves on, but the number is not cosmetic: graph
        capture is part of what a deployment pays before it serves, and a replay
        that reports zero has quietly dropped a real startup cost from any
        amortisation claim.
        """
        graph = self.target.graph
        return (float(graph.get("capture_seconds") or 0.0),
                list(self.capture_sizes),
                int(graph.get("pool_bytes") or 0))

    def warmup_model(self):
        """No forward to warm. The cost of the one a deployment runs is data.

        Whatever the first real forward costs above steady state belongs in the
        oracle as a first-use term, not here -- this runner has no way to
        discover it and must not invent one.
        """

    def process_kvconnector_output(self, connector_meta_output):
        raise NotImplementedError(
            "ATOMCompass: KV transfer is not replayed. PD disaggregation is "
            "out of the PoC's scope and a replay that silently ignored the "
            "connector would report a schedule that cannot happen.")

    def dummy_execution(self):
        """Data-parallel lockstep filler. No collective here, so nothing to do."""
        return None

    def freeze_gc_heap(self, *args, **kwargs):
        """The engine freezes its workers after startup; there is no worker."""

    def exit(self):
        fh = getattr(self, "_measure_fh", None)
        if fh is not None:
            try:
                fh.close()
            finally:
                self._measure_fh = None

    # -- anything else the engine reaches for ---------------------------------

    def __getattr__(self, name: str):
        # Only consulted for attributes that do not exist, so this cannot mask
        # a real one. It exists to fail *loudly and by name*: the alternative is
        # an AttributeError raised deep inside an engine RPC, reported as a
        # worker that vanished, naming neither Compass nor the missing method.
        raise AttributeError(
            f"ATOMCompass: the GPU-free replay runner has no {name!r}. The "
            f"engine reached for something a device-backed runner supplies and "
            f"this one does not model. Add it deliberately, with what it should "
            f"answer off a device -- do not inherit it by accident."
        )
