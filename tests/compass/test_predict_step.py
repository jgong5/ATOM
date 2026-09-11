"""Reading a shape list, and keying a graph that is already on disk.

The CLI's own body is a composition of things tested elsewhere. What is only
here is the boundary: turning JSON a caller wrote into a `StepShape`, and
turning a saved graph into the structure it is a template *for*. Both are
places where a wrong answer is quiet -- a shape with a mistyped field priced
as if the field were absent, or a template keyed at a bucket nobody traced it
at, which then answers for a padded graph that does not exist.
"""

import importlib.util
import pathlib

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "compass" / "predict_step.py")
_SPEC = importlib.util.spec_from_file_location("predict_step", _PATH)
predict_step = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(predict_step)


def saved_graph(*, bucket=None, spec=True, topology=(("tp", 2),),
                rank_coords=(("tp", 1),)):
    """A graph as `graph.save` writes one: coordinate maps as pair lists."""
    provenance = {"execution": {"capture_bucket": bucket, "step_kind": "decode"}}
    if spec:
        provenance["batch_spec"] = {
            "kind": "decode", "query_lens": [1, 1, 1, 1],
            "context_lens": [1151] * 4, "block_size": 16}
    return {"key": {"topology": [list(p) for p in topology],
                    "rank_coords": [list(p) for p in rank_coords]},
            "ops": [], "provenance": provenance}


# -- the shape list ---------------------------------------------------------

def test_a_shape_is_read_with_its_declared_fields():
    shape = predict_step._shape_from(
        {"num_scheduled_tokens": [1, 1], "context_lens": [10, 20],
         "topology": {"tp": 2}, "rank_coords": {"tp": 1},
         "capture_bucket": 4, "produces_output": False})
    assert shape.num_scheduled_tokens == (1, 1)
    assert shape.context_lens == (10, 20)
    assert shape.capture_bucket == 4
    assert shape.produces_output is False


def test_a_shape_missing_lengths_is_refused():
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1]})
    assert "context_lens" in str(exc.value)


def test_a_misspelled_field_is_refused_rather_than_ignored():
    """`capture_buckets` would otherwise be dropped and read as no bucket."""
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1],
                                  "context_lens": [10],
                                  "capture_buckets": 4})
    assert "capture_buckets" in str(exc.value)


# -- keying a saved graph ---------------------------------------------------

def test_saved_coordinate_pair_lists_are_read_back_as_maps():
    shape = predict_step._template_shape(saved_graph())
    assert shape.topology == {"tp": 2}
    assert shape.rank_coords == {"tp": 1}


def test_a_coordinate_map_saved_as_a_dict_also_reads():
    graph = saved_graph()
    graph["key"]["topology"] = {"tp": 2}
    assert predict_step._template_shape(graph).topology == {"tp": 2}


def test_the_bucket_comes_from_execution_not_the_batch():
    """A batch spec describes requests; the bucket is how the step was run."""
    assert predict_step._template_shape(saved_graph(bucket=32)).capture_bucket == 32


def test_a_graph_traced_with_no_bucket_keys_as_none():
    """It is a template for the uncaptured structure, and must not claim one."""
    assert predict_step._template_shape(saved_graph()).capture_bucket is None


def test_a_graph_with_no_batch_spec_cannot_be_a_template():
    with pytest.raises(ValueError) as exc:
        predict_step._template_shape(saved_graph(spec=False))
    assert "batch_spec" in str(exc.value)


# -- price entries ----------------------------------------------------------

def test_price_entries_take_one_two_or_three_parts():
    assert predict_step._load_prices(["p.json"]) == [("p.json", None)]
    assert predict_step._load_prices(["p.json:g.json"]) == [("p.json", "g.json")]
    assert (predict_step._load_prices(["p.json:g.json:unregistered"])
            == [("p.json", "g.json", "unregistered")])
    # An empty middle is "no graph, but a regime", not a path named "".
    assert predict_step._load_prices(["p.json::plain"]) == [("p.json", None,
                                                             "plain")]


def test_a_fourth_part_is_refused():
    with pytest.raises(ValueError):
        predict_step._load_prices(["p.json:g.json:unregistered:extra"])
