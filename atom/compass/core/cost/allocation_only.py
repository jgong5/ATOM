"""Device-only semantics for uninitialized BF16 allocations.

The pinned native Torch revision routes empty_like through empty/empty_strided.
Those dispatch a fill kernel only when both deterministic flags are enabled.
Allocation footprint and host allocator work are separate from this statement.
"""
from math import prod

from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.records import OperatorEventRecord

SCHEMA = "compass.uninitialized_allocation_semantics/1"
TORCH_REVISION = "3d3aa833db84eed6b7f5595cb5f162c2f78300a4"
SOURCE_FILES = {
    "aten/src/ATen/native/TensorFactories.cpp": "9e0f0ba334a290b1d2a055dd2b8082cfec808e2c6631e1a65be22826348aeda2",
    "aten/src/ATen/native/cuda/TensorFactories.cu": "0170fae2e8eed26f5516418a25cf547084c2e0e169dfa06689fe25c86caabb46",
}


def runtime_policy():
    import torch
    import torch.utils.deterministic

    return dict(torch_version=torch.__version__, torch_git_version=torch.version.git_version,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory)


def uninitialized(policy):
    return (policy.get("torch_git_version") == TORCH_REVISION
            and all(type(policy.get(key)) is bool for key in
                    ("deterministic_algorithms", "fill_uninitialized_memory"))
            and not (policy["deterministic_algorithms"] and policy["fill_uninitialized_memory"]))


def allocation_shape(operator):
    op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
    shapes = op.get("input_shapes") or []
    if (op.get("name") != "aten::empty_like" or op.get("group") is not None
            or op.get("abi", "") or op.get("int_values") or op.get("int_ranges")
            or op.get("launch") or op.get("param_names") or op.get("context")
            or len(shapes) != 1 or not shapes[0]
            or any(type(value) is not int or value <= 0 for value in shapes[0])
            or op.get("dtypes") != ["bfloat16"]
            or op.get("output_shapes") != shapes or op.get("output_dtypes") != ["bfloat16"]
            or op.get("output_aliases") != [None]
            or dict(op.get("scalars") or ()) != {"pin_memory": False}):
        return None
    # The allocation reads the input's shape/stride metadata, not its values.
    # Validate a real native view rather than accepting a malformed descriptor.
    from atom.compass.core.cost.exact_operator_references import argument_views

    views = argument_views(op)
    if len(views) != 1:
        return None
    view = views[0]
    if (view["owner"] != 0 or type(view["offset"]) is not int or view["offset"] < 0
            or len(view["stride"]) != len(shapes[0])
            or any(type(s) is not int or s <= 0 for s in view["stride"])
            or type(view["elements"]) is not int
            or view["offset"] + sum((n - 1) * s for n, s in zip(shapes[0], view["stride"])) >= view["elements"]):
        return None
    return shapes[0]


class AllocationOnly:
    """A code-proved zero device cost, conditioned on both runtimes' flags."""

    def __init__(self, reader, pin):
        proof = reader.read(pin, "allocation_only.proof")
        if (proof.get("schema") != SCHEMA or proof.get("torch_git_version") != TORCH_REVISION
                or proof.get("gpu_kernel_cost_only") is not True
                or proof.get("host_allocation_and_footprint_separate") is not True
                or proof.get("timings_used_to_infer_zero") is not False
                or set(proof["native_sources"]) != set(SOURCE_FILES)):
            raise ValueError("allocation-only proof changes its scope or device-work argument")
        for name, digest in SOURCE_FILES.items():
            source = proof["native_sources"][name]
            if source["sha256"] != digest:
                raise ValueError("allocation-only proof names another native implementation")
            reader.read(source, "allocation_only." + name, json_data=False)
        if (not reader.read(proof["release"], "allocation_only.release")["verified"]
                or not reader.read(proof["collection"], "allocation_only.collection")["copy_complete"]):
            raise ValueError("allocation-only runtime observations lack closed source evidence")
        policies, seeds = [], set()
        for index, observation in enumerate(proof["native_observations"]):
            raw = reader.read(observation["raw"], f"allocation_only.native{index}")
            closed = reader.read(observation["exit"], f"allocation_only.exit{index}")
            acquisition = raw["acquisition"]
            policy = raw["provenance"]["native_layout_context_setup"]["runtime_allocation_policy"]
            if (closed["exit_code"] != 0 or acquisition.get("source_only") is not True
                    or acquisition.get("role") != "reference"
                    or closed["run"]["seed"] != acquisition["seed"]
                    or not uninitialized(policy)):
                raise ValueError("allocation-only native runtime may fill uninitialized memory")
            policies.append(policy)
            seeds.add(acquisition["seed"])
        if len(policies) != 3 or len(seeds) != 3 or any(p != policies[0] for p in policies):
            raise ValueError("allocation-only proof lacks three consistent native runtime observations")
        self.native_policy = policies[0]
        self.proof = pin
        if not uninitialized(runtime_policy()):
            raise ValueError("allocation-only modelled runtime may fill uninitialized memory")

    def configuration_key(self):
        return tuple(sorted(runtime_policy().items()))

    def finish(self, op, topology, original):
        if op.get("name") != "aten::empty_like":
            return original
        policy = runtime_policy()
        if not uninitialized(policy):
            return None, "empty_like allocation proof requires known disabled deterministic filling"
        if original[0] is not None:
            return original
        shape = allocation_shape(op)
        if (shape is None or not topology or topology.get("tp") != 1
                or any(type(v) is not int or v != 1 for v in topology.values())):
            return original
        source = "structural://empty-like/uninitialized-native-allocation"
        return OperatorEventRecord(seconds=0.0, kernels={}, source=source, zero_work=True,
            kernel_count=0, launch_count=0, source_qualified=False,
            whole_forward_validation_required=True, allocation_only=dict(
                proof=self.proof, native_runtime_policy=self.native_policy, modelled_runtime_policy=policy,
                basis="native empty_like allocates through empty/empty_strided without deterministic fill",
                output_logical_bytes=prod(shape) * 2, gpu_kernel_cost_only=True,
                host_allocation_and_footprint_separate=True, timings_used_to_infer_zero=False)), source
