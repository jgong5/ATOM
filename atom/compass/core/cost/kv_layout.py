"""Compare native KV view layout separately from its allocation capacity."""


def _frozen(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def kv_layout_key(layout):
    """Ignore only the leading block extent of recorded 5D native K/V views.

    Their per-block shape, strides, dtype and contiguity still must agree.
    Source allocation extent remains in the original scope as conditioning;
    this comparison does not assert an observed target capacity. Unrecognized
    layouts retain their complete identity.
    """
    layout = _frozen(layout)
    if not isinstance(layout, tuple):
        return layout
    try:
        views = dict(layout)
        if set(views) != {"k", "v"}:
            return layout
        normalized = []
        for name, description in layout:
            fields = dict(description)
            shape, stride = fields.get("shape"), fields.get("stride")
            if (not isinstance(shape, tuple) or len(shape) != 5
                    or not isinstance(stride, tuple) or len(stride) != 5
                    or type(shape[0]) is not int or shape[0] <= 0):
                return layout
            normalized.append((name, tuple((key, ("*blocks",) + value[1:]
                                           if key == "shape" else value)
                                          for key, value in description)))
        return tuple(normalized)
    except (TypeError, ValueError):
        return layout
