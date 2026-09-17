"""Source-only elapsed-M fit for the separately observed cached four-row domain."""
from math import isfinite
from statistics import median

from atom.compass.core.cost.composition_qualification import geometry
from atom.compass.core.cost.native_ap_work import eligible

SCHEMA = "compass.compiled_prefill_width_fit/1"
METHOD = "relative least squares on all cached-N4 source geometry medians"


def fit_scale_floor(points):
    """Minimize relative squared error for max(alpha * B, floor), with positive parameters."""
    points = sorted(points)
    if not points or any(not isfinite(v) or v <= 0 for point in points for v in point):
        raise ValueError("width execution fit needs positive finite source B and M")
    candidates = []
    # Every change in the floor/linear partition is a one-variable boundary.
    for boundary, _ in points:
        x = [max(b, boundary) / m for b, m in points]
        alpha = sum(x) / sum(v * v for v in x)
        candidates.append((alpha, alpha * boundary))
    # Interior optima separate into the floor and scale weighted means.
    for cut in range(1, len(points)):
        lower, upper = points[:cut], points[cut:]
        floor = sum(1 / m for _, m in lower) / sum(1 / (m * m) for _, m in lower)
        x = [b / m for b, m in upper]
        alpha = sum(x) / sum(v * v for v in x)
        if alpha * lower[-1][0] <= floor <= alpha * upper[0][0]:
            candidates.append((alpha, floor))
    alpha, floor = min(candidates, key=lambda p: sum(((max(p[0] * b, p[1]) - m) / m) ** 2
                                                     for b, m in points))
    return dict(alpha=alpha, floor_seconds=floor)


def source_groups(source, quotes):
    """Require every eligible cached-N4 observation, preserving its raw measured M."""
    expected = {i for i, row in enumerate(source) if eligible(row)
                and len(row["descriptor"]["q"]) == 4 and any(row["descriptor"]["history"])}
    observations = quotes["observations"]
    if (not expected or any(row.get("role") != "source" for row in source)
            or len(observations) != len(expected)
            or {row["source_row_index"] for row in observations} != expected
            or quotes.get("refused")):
        raise ValueError("width execution fit omits or duplicates eligible cached-N4 source observations")
    groups = {}
    for observation in observations:
        row = source[observation["source_row_index"]]
        key = observation["geometry_key"]
        quote = quotes["quotes"][key]
        if (row.get("normal_return") is not True or quote.get("complete") is not True
                or quote["geometry"] != geometry(row["descriptor"])
                or quote["B"] != quote["body_seconds"] + quote["head_seconds"]
                or observation["B"] != quote["B"] or observation["M"] != row["seconds"]["run_model"]
                or not isfinite(observation["B"]) or observation["B"] <= 0
                or not isfinite(observation["M"]) or observation["M"] <= 0):
            raise ValueError("width execution fit changes native source M or its complete raw B quote")
        groups.setdefault(key, []).append(observation["M"])
    if set(groups) != set(quotes["quotes"]):
        raise ValueError("width execution fit adds unused source geometries")
    return [dict(geometry_key=key, B=quotes["quotes"][key]["B"], all_source_M=values,
                 median_M=median(values)) for key, values in sorted(groups.items())]


def validate_width_fit(width, active, candidate, read, historical_inputs):
    from atom.compass.core.cost.compiled_prefill_execution import body_inputs

    if "execution_fit" not in width:
        raise ValueError("width execution parameters lack their frozen source fit")
    fit = read(width["execution_fit"], "width_execution_fit")
    if (fit.get("schema") != SCHEMA or fit.get("source_only") is not True
            or fit.get("frozen") is not True or fit.get("source_qualified") is not False
            or fit.get("heldout_rows_read") is not False or fit.get("e2e_timings_read") is not False
            or fit.get("original_parameters_unchanged") is not True
            or fit.get("frozen_before_fresh_heldouts") is not True
            or fit.get("full_forward_qualification_required") is not True
            or fit.get("method") != METHOD
            or fit.get("source") != active["source"]
            or fit["source"]["sha256"] != candidate["source_input"]["sha256"]):
        raise ValueError("width execution fit changes its separate cached-N4 source-only scope")
    source = read(fit["source"], "width_execution_source")["rows"]
    quotes = read(fit["body_quotes"], "width_execution_quotes")
    inputs = read(quotes["loaded_inputs"], "width_body_loaded_inputs")
    if (quotes.get("source_only") is not True or quotes.get("heldout_rows_read") is not False
            or quotes.get("e2e_timings_read") is not False or quotes["source"]["sha256"] != fit["source"]["sha256"]
            or body_inputs(inputs) != body_inputs(historical_inputs)):
        raise ValueError("width execution fit changes its body sources or uses target observations")
    groups = source_groups(source, quotes)
    parameters = fit_scale_floor([(g["B"], g["median_M"]) for g in groups])
    if (fit.get("source_groups") != groups or fit.get("parameters") != parameters
            or width.get("parameters") != parameters):
        raise ValueError("width execution parameters differ from the complete frozen source fit")
    return parameters
