# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for helpers in ``atom.entrypoints.openai.api_server`` that do
not require a GPU or a running engine.

The ``api_server`` module pulls in transformers + uvicorn + fastapi + an
engine-ready ``atom`` package at import time. The repo's ``tests/conftest.py``
already stubs several heavy imports; here we only test small pure-python
helpers, so if any transitive dependency is unavailable we skip the module
rather than block the rest of the suite.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import types
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing


def _install_api_server_stubs() -> list[str]:
    """Ensure attribute access ``atom.SamplingParams`` works under the stubbed
    ``atom`` package that ``tests/conftest.py`` installs, and stub any heavy
    transitive deps (``aiter``-backed engine core manager and its argparse
    helper) that ``api_server`` would otherwise drag in at import time.

    Stubs are only installed when the corresponding real module cannot be
    imported in this environment (e.g. Windows without ``aiter``). Any
    module we inject here is recorded and torn down in a module-level
    fixture so we don't leak stubs into tests that run later and expect
    the real implementation (notably ``tests/test_arg_utils_spec.py``).
    """
    import importlib

    from atom.sampling_params import SamplingParams  # real implementation

    atom_pkg = sys.modules.get("atom")
    if atom_pkg is not None and not hasattr(atom_pkg, "SamplingParams"):
        atom_pkg.SamplingParams = SamplingParams  # type: ignore[attr-defined]

    injected: list[str] = []

    def _try_import_else_stub(mod_name: str, attr_name: str, stub_cls) -> None:
        if mod_name in sys.modules:
            return
        try:
            importlib.import_module(mod_name)
        except Exception:
            stub = types.ModuleType(mod_name)
            setattr(stub, attr_name, stub_cls)
            sys.modules[mod_name] = stub
            injected.append(mod_name)

    class _StubCoreManager:
        def __init__(self, *a, **kw):
            pass

        def add_request(self, reqs):
            return None

    class _StubEngineArgs:
        @classmethod
        def add_cli_args(cls, parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine(self, tokenizer=None):
            return None

    _try_import_else_stub(
        "atom.model_engine.engine_core_mgr", "CoreManager", _StubCoreManager
    )
    _try_import_else_stub("atom.model_engine.arg_utils", "EngineArgs", _StubEngineArgs)
    return injected


_injected_modules: list[str] = []  # set in try; kept defined for `finally`
try:
    _injected_modules = _install_api_server_stubs()
    import importlib

    api_server = importlib.import_module("atom.entrypoints.openai.api_server")
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    # Re-raises unless a third-party dependency is what is missing. It used to
    # skip on anything, so a syntax error in `api_server.py` silenced this
    # whole module and the suite still reported a clean run.
    skip_if_dependency_missing(exc, "api_server import unavailable")
    api_server = None  # type: ignore[assignment]
    _import_error = exc
    # NB: do NOT reset _injected_modules here. When api_server import fails
    # (e.g. PIL absent on the non-GPU runner), the stubs injected by
    # _install_api_server_stubs() must still be torn down in `finally`;
    # clearing the list here would leak them into sys.modules and pollute
    # tests collected later (notably tests/test_arg_utils_spec.py, which then
    # sees a stub EngineArgs instead of the real one).
else:
    _import_error = None
finally:
    # Remove any stubs we injected so tests collected *after* this module
    # (notably ``tests/test_arg_utils_spec.py``) can still import the real
    # ``atom.model_engine.arg_utils`` / ``engine_core_mgr``. ``api_server``
    # already bound the names it needed at module import time.
    for _mod_name in list(_injected_modules):
        sys.modules.pop(_mod_name, None)
    _injected_modules = []


pytestmark = pytest.mark.skipif(
    api_server is None,
    reason=f"api_server import unavailable: {_import_error!r}",
)


class TestCoerceN:
    """``_coerce_n`` normalizes the request ``n`` before engine fan-out."""

    def test_none_becomes_one(self):
        assert api_server._coerce_n(None, 0.8) == 1

    def test_zero_becomes_one(self):
        assert api_server._coerce_n(0, 0.8) == 1

    def test_negative_becomes_one(self):
        assert api_server._coerce_n(-2, 0.8) == 1

    def test_non_int_string_becomes_one(self):
        assert api_server._coerce_n("not-a-number", 0.8) == 1  # type: ignore[arg-type]

    def test_n_passes_through_when_temperature_positive(self):
        assert api_server._coerce_n(4, 0.7) == 4

    def test_n_collapses_to_one_under_greedy_sampling(self):
        # temperature==0 => greedy, so n>1 would produce identical siblings.
        assert api_server._coerce_n(4, 0.0) == 1

    def test_n_collapses_to_one_when_temperature_missing(self):
        assert api_server._coerce_n(4, None) == 1

    def test_n_one_with_greedy_stays_one(self):
        assert api_server._coerce_n(1, 0.0) == 1


class TestBuildSamplingParams:
    """``_build_sampling_params`` threads ``n`` into SamplingParams."""

    def test_default_n_is_one(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
        )
        assert sp.n == 1

    def test_n_greater_than_one_propagates(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
            n=4,
        )
        assert sp.n == 4

    def test_invalid_n_rejected_by_sampling_params(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            api_server._build_sampling_params(
                temperature=0.8,
                max_tokens=16,
                stop_strings=None,
                ignore_eos=False,
                n=0,
            )


class TestDPSessionAffinityHeaders:
    def test_disabled_drops_session_metadata(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "0")
        request = SimpleNamespace(
            headers={
                "x-dynamo-session-id": "child",
                "x-dynamo-parent-session-id": "parent",
            }
        )
        assert api_server._get_dp_session_affinity_ids(request) == (None, None)

    def test_extracts_dynamo_session_lineage(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "1")
        request = SimpleNamespace(
            headers={
                "x-dynamo-session-id": "child",
                "x-dynamo-parent-session-id": "parent",
                "x-correlation-id": "fallback",
            }
        )
        assert api_server._get_dp_session_affinity_ids(request) == (
            "child",
            "parent",
        )

    def test_correlation_id_is_session_fallback(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "true")
        request = SimpleNamespace(headers={"x-correlation-id": "session"})
        assert api_server._get_dp_session_affinity_ids(request) == (
            "session",
            None,
        )


class TestAnthropicSamplingParams:
    def test_request_overrides_model_then_neutral_defaults(self, monkeypatch):
        captured = {}
        build_sampling_params = api_server._build_sampling_params

        def capture_sampling_params(**kwargs):
            captured.update(kwargs)
            return build_sampling_params(**kwargs)

        async def fake_nonstream(*_args, **_kwargs):
            return {"text": "", "num_cached_tokens": 0}

        monkeypatch.setattr(
            api_server,
            "engine",
            SimpleNamespace(
                config=SimpleNamespace(
                    generation_config=SimpleNamespace(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=None,
                    ),
                    max_model_len=4096,
                )
            ),
        )
        monkeypatch.setattr(
            api_server, "tokenizer", SimpleNamespace(encode=lambda _text: [1])
        )
        monkeypatch.setattr(api_server, "model_name", "test")
        monkeypatch.setattr(api_server, "apply_chat_template", lambda *_a, **_kw: "")
        monkeypatch.setattr(
            api_server, "_build_sampling_params", capture_sampling_params
        )
        monkeypatch.setattr(
            api_server, "_run_nonstream_with_disconnect", fake_nonstream
        )

        request = api_server.AnthropicMessagesRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
        )
        asyncio.run(api_server.anthropic_messages(request, None))

        assert captured["temperature"] == 0.0
        assert captured["top_p"] == 0.95
        assert captured["top_k"] == -1


class TestValidateContextLength:
    """Oversized OpenAI requests should fail before entering the scheduler."""

    def test_equal_to_max_model_len_is_allowed(self):
        api_server._validate_context_length(
            num_prompt_tokens=120,
            max_tokens=8,
            max_model_len=128,
        )

    def test_total_over_max_model_len_is_rejected(self):
        with pytest.raises(ValueError, match="maximum context length is 128"):
            api_server._validate_context_length(
                num_prompt_tokens=121,
                max_tokens=8,
                max_model_len=128,
            )

    def test_prompt_alone_over_max_model_len_is_rejected(self):
        with pytest.raises(ValueError, match="prompt contains at least 129"):
            api_server._validate_context_length(
                num_prompt_tokens=129,
                max_tokens=0,
                max_model_len=128,
            )

    def test_missing_max_model_len_skips_validation(self):
        api_server._validate_context_length(
            num_prompt_tokens=129,
            max_tokens=8,
            max_model_len=None,
        )


class TestCompletionTokenPrompts:
    class Tokenizer:
        vocab_size = 8

        def __init__(self):
            self.encoded = []

        def __len__(self):
            return 12  # Four added tokens are valid input IDs as well.

        def encode(self, text, **kwargs):
            assert isinstance(text, str), "token IDs must never be re-tokenized"
            self.encoded.append(text)
            return [1, 2, 3]

        def decode(self, tokens, **kwargs):
            return "done"

    def _engine(self, monkeypatch, *, model_vocab_size=10):
        from atom.model_engine.llm_engine import InputOutputProcessor
        from atom.model_engine.request import RequestOutput

        tokenizer = self.Tokenizer()
        config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="test", vocab_size=model_vocab_size),
            max_model_len=32)
        processor = InputOutputProcessor(config, tokenizer, 16)
        received, admitted = [], []
        original = processor.preprocess_fanout

        def preprocess(prompt, *args, **kwargs):
            received.append(prompt)
            return original(prompt, *args, **kwargs)

        processor.preprocess_fanout = preprocess

        def add_request(sequences):
            admitted.extend(sequences)
            for seq in sequences:
                seq.stream_callback(RequestOutput(seq.id, [7], True, "length"))

        engine = SimpleNamespace(config=config, io_processor=processor,
                                 core_mgr=SimpleNamespace(add_request=add_request))
        monkeypatch.setattr(api_server, "engine", engine)
        monkeypatch.setattr(api_server, "tokenizer", tokenizer)
        monkeypatch.setattr(api_server, "model_name", "test")
        return tokenizer, processor, received, admitted

    @pytest.mark.parametrize("n", [1, 2])
    def test_token_ids_reach_real_preprocessing_without_tokenization(self, monkeypatch, n):
        tokenizer, processor, received, admitted = self._engine(monkeypatch)
        request = api_server.CompletionRequest(prompt=[1, 8, 9], max_tokens=1, n=n)
        response = asyncio.run(api_server.completions(request, None))
        assert received == [[1, 8, 9]]
        assert received[0] is request.prompt
        assert all(list(seq.token_ids) == [1, 8, 9] for seq in admitted)
        assert len(admitted) == n
        assert tokenizer.encoded == []
        assert response.usage["prompt_tokens"] == 3
        assert response.usage["completion_tokens"] == n
        assert not processor.requests

    def test_text_prompt_keeps_normal_tokenization(self, monkeypatch):
        tokenizer, _, received, admitted = self._engine(monkeypatch)
        request = api_server.CompletionRequest(prompt="ordinary text", max_tokens=1)
        response = asyncio.run(api_server.completions(request, None))
        assert received == ["ordinary text"]
        assert tokenizer.encoded == ["ordinary text"]
        assert list(admitted[0].token_ids) == [1, 2, 3]
        assert response.usage["prompt_tokens"] == 3

    def test_declared_order_reaches_the_scheduler_handoff(self, monkeypatch):
        from atom.utils.clock import VirtualClock, get_clock, set_clock

        _, _, _, admitted = self._engine(monkeypatch)
        previous = get_clock()
        set_clock(VirtualClock(epoch=1000.0))
        try:
            request = api_server.CompletionRequest(
                prompt=[1, 8, 9], max_tokens=1,
                compass_arrival=5.0, compass_workload_size=4,
                compass_workload_index=2,
            )
            asyncio.run(api_server.completions(request, None))
        finally:
            set_clock(previous)
        assert len(admitted) == 1
        seq = admitted[0]
        assert seq.arrive_time == 1005.0
        assert seq.compass_workload_size == 4
        assert seq.compass_workload_index == 2

    @pytest.mark.parametrize("model_size,invalid_id", [(10, 10), (20, 12)])
    @pytest.mark.parametrize("stream", [False, True])
    def test_out_of_range_ids_refuse_before_preprocessing(
            self, monkeypatch, model_size, invalid_id, stream):
        _, _, received, admitted = self._engine(monkeypatch, model_vocab_size=model_size)
        request = api_server.CompletionRequest(prompt=[invalid_id], max_tokens=1, stream=stream)
        with pytest.raises(api_server.HTTPException) as exc:
            asyncio.run(api_server.completions(request, None))
        assert exc.value.status_code == 400
        assert "Prompt token IDs" in str(exc.value.detail)
        assert received == admitted == []

    def test_token_ids_keep_context_length_validation(self, monkeypatch):
        _, processor, received, admitted = self._engine(monkeypatch)
        request = api_server.CompletionRequest(prompt=[1] * 32, max_tokens=1)
        with pytest.raises(api_server.HTTPException) as exc:
            asyncio.run(api_server.completions(request, None))
        assert exc.value.status_code == 400
        assert "maximum context length" in str(exc.value.detail)
        assert received and not admitted
        assert not processor.requests


class TestOpeningChatPreprocessing:
    @pytest.mark.parametrize("wrong_digest", [False, True])
    def test_chat_metadata_and_consumed_tokens_cross_shared_preprocessing(self, monkeypatch, wrong_digest):
        from collections import OrderedDict
        from atom.compass.prefix_workload import token_digest
        from atom.model_engine.llm_engine import InputOutputProcessor
        from atom.model_engine.request import RequestOutput
        from atom.utils.clock import VirtualClock, get_clock, set_clock

        class Tokenizer:
            def encode(self, text, **kwargs):
                assert text == "rendered opening"
                return [1, 2, 3]

        config = SimpleNamespace(hf_config=SimpleNamespace(model_type="test"), max_model_len=32)
        processor = InputOutputProcessor(config, Tokenizer(), 16)
        admitted = []

        def add(sequences):
            admitted.extend(sequences)
            for seq in sequences:
                seq.stream_callback(RequestOutput(
                    seq.id, [7], True, "length", arrive_time=seq.arrive_time,
                    first_token_time=seq.arrive_time + .1, finish_time=seq.arrive_time + .2))

        monkeypatch.setattr(api_server, "engine", SimpleNamespace(
            config=config, io_processor=processor, core_mgr=SimpleNamespace(add_request=add)))
        monkeypatch.setattr(api_server, "model_name", "test")
        monkeypatch.setattr(api_server, "tokenizer", processor.tokenizer)
        monkeypatch.setattr(api_server, "apply_chat_template", lambda *a, **kw: "rendered opening")
        monkeypatch.setattr(api_server, "_stream_batch_dispatcher", SimpleNamespace(
            new_state=lambda: object(), enqueue=lambda **kw: None))
        monkeypatch.setattr(api_server, "_compass_records", OrderedDict())
        monkeypatch.setattr(api_server, "_compass_prompt_evidence", {})
        digest = token_digest([1, 2, 3])
        request = api_server.ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "source message"}],
            max_completion_tokens=1, temperature=0, ignore_eos=True, stream=True,
            compass_arrival=21.437, compass_workload_size=2, compass_workload_index=1,
            compass_prompt_token_sha256="0" * 64 if wrong_digest else digest)
        previous = get_clock()
        set_clock(VirtualClock(epoch=1000.))
        try:
            if wrong_digest:
                with pytest.raises(api_server.HTTPException) as exc:
                    asyncio.run(api_server.chat_completions(request, None))
                assert exc.value.status_code == 400
                assert "pinned token identity" in str(exc.value.detail)
                assert not admitted and not processor.requests
            else:
                asyncio.run(api_server.chat_completions(request, None))
                assert len(admitted) == 1
                seq = admitted[0]
                assert list(seq.token_ids) == [1, 2, 3]
                assert seq.arrive_time == 1021.437
                assert seq.compass_workload_size == 2 and seq.compass_workload_index == 1
                assert not hasattr(seq, "prompt_token_sha256")  # No Sequence wire-layout change.
                record, = api_server._compass_records.values()
                assert record["shared_preprocessing"]["prompt_token_sha256"] == digest
                assert record["shared_preprocessing"]["input_tokens"] == 3
                assert record["shared_preprocessing"]["api_preprocess_returned_wall_time"] > 0
                api_server.cleanup_stream(seq.id)
                api_server.cleanup_request(record["request_id"])
        finally:
            set_clock(previous)


class TestResolvedCompassProvenance:
    @pytest.mark.parametrize("side", ["real", "modelled"])
    @pytest.mark.parametrize("missing_core", [False, True])
    def test_producer_output_satisfies_cache_on_validator_without_api_policy_defaults(
            self, monkeypatch, side, missing_core):
        import importlib.util
        import queue
        from pathlib import Path
        from conftest import MockConfig
        from atom.compass.core import cache_boundary
        from atom.compass.core.cache_policy import cache_on_policy
        from atom.compass.runtime.predict import CompassPredictMixin
        from atom.model_engine.scheduler import Scheduler
        from atom.model_engine.state_runtime import StateRuntime, StateTransfer
        from atom.utils.clock import VirtualClock, WallClock, get_clock, set_clock

        spec = importlib.util.spec_from_file_location(
            "cache_producer_validator", Path(__file__).resolve().parents[2] / "scripts/compass/cc_traces_validate.py")
        validator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = validator
        spec.loader.exec_module(validator)
        mode = "measure" if side == "real" else "predict"
        previous = get_clock()
        set_clock(WallClock() if side == "real" else VirtualClock(epoch=1000.))
        try:
            state = StateRuntime(transfer=StateTransfer.fork(1))
            config = MockConfig(
                kv_cache_block_size=16, num_kvcache_blocks=200, max_model_len=262144,
                max_num_batched_tokens=16384, max_num_seqs=32, enable_prefix_caching=True,
                pool_entries={"state": 32}, state_checkpoint_interval_tokens=8192,
                state_checkpoint_demand=True)
            core = SimpleNamespace(
                scheduler=Scheduler(config, state_runtime=state), state_runtime=state,
                input_queue=queue.Queue(), stream_output_queue=queue.Queue(),
                has_pending_kv_work=lambda: False,
                runner_mgr=SimpleNamespace(proc_num=1, call_func=lambda *_a, **_k: {
                    "acknowledged": True, "kind": "device_synchronize" if side == "real" else "modelled_no_device",
                    "retained_output_requests": 0, "retained_outputs_preserved": True}))
            reset = cache_boundary.reset(core)
            compass = SimpleNamespace(enabled=True, mode=mode, virtual_clock=side == "modelled",
                                      oracle_qualname="fixture", oracle_options={}, admission_seconds=0.)
            api_config = SimpleNamespace(
                model="fixture", revision="fixture-revision", compass_config=compass,
                tensor_parallel_size=1, pipeline_parallel_size=1, max_model_len=262144,
                max_num_seqs=32, gpu_memory_utilization=.9, enable_prefix_caching=True,
                state_checkpoint_interval_tokens=1, state_checkpoint_demand=False)
            worker_config = SimpleNamespace(
                **{key: getattr(api_config, key) for key in (
                    "model", "tensor_parallel_size", "pipeline_parallel_size", "max_model_len",
                    "max_num_seqs", "gpu_memory_utilization", "enable_prefix_caching")},
                max_num_batched_tokens=16384, kv_cache_block_size=16, kv_cache_dtype="bf16",
                enforce_eager=False, compilation_config=SimpleNamespace(level=3, cudagraph_mode="FULL"),
                capture_sizes=[1, 2, 4, 8, 16, 32, 48, 64, 128, 256])
            buckets = [1, 2, 4, 8, 16, 32]
            target = SimpleNamespace(graph={"capture_sizes": buckets}, loaded_input=SimpleNamespace(
                as_dict=lambda: {"path": "/source/target.json", "sha256": "b" * 64}))
            worker = SimpleNamespace(config=worker_config, rank=0, _compass_config=compass,
                                     capture_sizes=buckets, _compass_native_capture_sizes=buckets,
                                     target=target if side == "modelled" else None,
                                     _compass_oracle_inputs=(), _rank_coords=lambda: {},
                                     _oracle=SimpleNamespace(), _observe_device_freedom=lambda _: {})
            worker_manifest = CompassPredictMixin.compass_input_manifest(worker)
            def core_cache(**kwargs):
                if missing_core:
                    raise RuntimeError("core unavailable")
                return {"schema": cache_boundary.SNAPSHOT_SCHEMA, "ranks": [cache_boundary.snapshot(core)]}
            monkeypatch.setattr(api_server, "engine", SimpleNamespace(config=api_config, get_compass_cache=core_cache))
            monkeypatch.setattr(api_server, "model_name", "fixture")
            monkeypatch.setattr(api_server, "_compass_loaded_inputs", lambda: {"ranks": [worker_manifest]})
            monkeypatch.setattr(api_server, "_server_revision", lambda: "fixture-revision")
            monkeypatch.setattr(api_server, "_server_code_digest", lambda: "a" * 64)
            monkeypatch.setattr(api_server, "_server_process_identity", lambda: {"pid": 123})
            published = asyncio.run(api_server.compass_provenance())
            manifest = {"server": published,
                        "cache_boundary": {"schema": cache_boundary.RESET_SCHEMA,
                                           "acknowledged": True, "ranks": [reset]},
                        "cache_state_after": {"schema": cache_boundary.SNAPSHOT_SCHEMA,
                                              "ranks": [cache_boundary.snapshot(core)]}}
            errors = validator.check_cache_policy_evidence(manifest, cache_on_policy(), side)
            assert bool(errors) is missing_core
            if not missing_core:
                assert published["cache_policy"] == cache_on_policy()
                assert published["cache_policy"]["state_checkpoint_interval_tokens"] == 8192
                assert published["core_cache"]["ranks"][0]["reader"]["component"] == "EngineCore.Scheduler"
            runtime, = published["worker_runtime"]
            assert runtime["configuration"]["declared_capture_sizes"] == worker_config.capture_sizes
            assert runtime["graphs"]["effective_decode_buckets"] == buckets
            if side == "modelled":
                assert runtime["graphs"]["native_capture_sizes"] is None
                assert runtime["graphs"]["origin"] == "borrowed_replay_target"
            else:
                assert runtime["graphs"]["native_capture_sizes"] == buckets
                assert runtime["graphs"]["borrowed_source_capture_sizes"] is None
                del worker._compass_native_capture_sizes
                uncaptured = CompassPredictMixin.compass_input_manifest(worker)["runtime_configuration"]
                assert uncaptured["graphs"]["native_capture_sizes"] is None
                assert uncaptured["graphs"]["origin"] == "not_captured"
        finally:
            set_clock(previous)


class TestARequestIsNotSerialisedForALogNobodyKeeps:
    """Building the log entry is the callee's job, not the caller's.

    ``_log_request_event`` returns immediately when request logging is off,
    but Python evaluates its arguments first, so
    ``_log_request_event("request", rid, request.model_dump())`` dumped every
    message and every tool schema on the event loop and threw the result away
    -- 20-26 us per request on an agent-shaped one, at all four call sites.
    The guard has to run before the dump, which is what
    ``_log_request_model`` is for.

    Asserted on whether ``model_dump`` ran, not on how long it took: the
    defect is an evaluation order, and an order is exactly observable.
    """

    class _Model:
        def __init__(self) -> None:
            self.dumps = 0

        def model_dump(self) -> dict:
            self.dumps += 1
            return {"big": "payload"}

    def test_nothing_is_dumped_while_request_logging_is_off(self, monkeypatch):
        monkeypatch.setattr(api_server, "_request_logger", None)
        model = self._Model()
        api_server._log_request_model("request", "req-1", model)
        assert model.dumps == 0, "the request was serialised for a log that is off"

    def test_it_is_dumped_once_when_request_logging_is_on(self, monkeypatch):
        written: list[str] = []
        monkeypatch.setattr(
            api_server, "_request_logger", SimpleNamespace(info=written.append)
        )
        model = self._Model()
        api_server._log_request_model("request", "req-1", model)
        assert model.dumps == 1
        assert written and "payload" in written[0], "the entry never reached the log"

    @staticmethod
    def _module_ast():
        return ast.parse(inspect.getsource(api_server))

    def _eager_sites(self):
        """Every ``_log_request_event(..., x.model_dump())`` outside the helper.

        ``_log_request_model``'s own body is that call, and there it is
        correct -- it runs after the guard. Excluding it by name rather than
        by weakening the pattern, because the pattern is the whole test.
        """
        tree = self._module_ast()
        helper = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_log_request_model"
            ),
            None,
        )
        exempt = {id(n) for n in ast.walk(helper)} if helper is not None else set()
        return [
            f"line {node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and id(node) not in exempt
            and getattr(node.func, "id", None) == "_log_request_event"
            for arg in node.args
            if isinstance(arg, ast.Call)
            and getattr(arg.func, "attr", None) == "model_dump"
        ]

    def test_the_scan_sees_the_endpoints(self):
        """The positive control: the endpoints must be using the helper.

        Without this the test passes on a module that stopped calling either
        function -- green, and reading nothing. The same silent retirement
        already happened once on this branch to the seeding scan.
        """
        tree = self._module_ast()
        used = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "_log_request_model"
        ]
        assert len(used) >= 4, f"only {len(used)} call sites use the helper: {used}"

    def test_no_call_site_dumps_before_the_guard(self):
        """The four sites this was written for, checked where they live.

        A helper nothing calls fixes nothing, and these sites sit in async
        route handlers no unit test reaches -- so the source is read instead,
        and an endpoint added later that spells it the old way is caught the
        moment it is written.
        """
        eager = self._eager_sites()
        assert not eager, (
            "_log_request_event is being handed a model_dump() built before "
            f"the guard can decline it: {eager}. Use _log_request_model."
        )
