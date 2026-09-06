"""Deriving the activation term by walking the graph.

Not how much memory the operators touch -- how much is live at once. That needs
to know which tensor is which, which shapes alone cannot say, so the trace
records which operator produced each input.
"""

from atom.compass.core.memory_model import peak_activation_bytes, weight_bytes


def _op(out, dtype="float32", inputs_from=()):
    return {"name": "aten::x", "input_shapes": [], "output_shapes": [out],
            "dtypes": [dtype], "inputs_from": list(inputs_from)}


class TestLiveness:
    def test_a_chain_holds_two_tensors_not_all_of_them(self):
        """Each link is dead as soon as the next has read it."""
        graph = {"ops": [_op([1024]), _op([1024], inputs_from=[0]),
                         _op([1024], inputs_from=[1]),
                         _op([1024], inputs_from=[2])]}
        assert peak_activation_bytes(graph) == 2 * 1024 * 4

    def test_a_tensor_read_late_stays_live(self):
        """A residual read at the end of a block is live across the block."""
        graph = {"ops": [_op([1024]), _op([16], inputs_from=[0]),
                         _op([16], inputs_from=[1]),
                         _op([16], inputs_from=[2, 0])]}
        # The 4KB tensor is live throughout. The peak is while an operator
        # runs: its own output exists *and* the input it is still reading, so
        # two of the small ones, not one.
        assert peak_activation_bytes(graph) == 1024 * 4 + 2 * 16 * 4

    def test_weights_are_not_activations(self):
        """An input no operator produced is a weight, and counting it here
        would double it against the weight term."""
        graph = {"ops": [_op([1024], inputs_from=[-1, -1])]}
        assert peak_activation_bytes(graph) == 1024 * 4

    def test_dtype_decides_the_bytes(self):
        small = {"ops": [_op([1024], dtype="bfloat16")]}
        large = {"ops": [_op([1024], dtype="float32")]}
        assert peak_activation_bytes(large) == 2 * peak_activation_bytes(small)


class TestWeights:
    def test_a_checkpoint_that_is_not_there_is_not_guessed(self, tmp_path):
        assert weight_bytes(str(tmp_path)) is None

    def test_a_sharded_checkpoint_reports_its_total(self, tmp_path):
        import json

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8_000_000_000}}))
        assert weight_bytes(str(tmp_path)) == 8_000_000_000
        assert weight_bytes(str(tmp_path), tensor_parallel=4) == 2_000_000_000
