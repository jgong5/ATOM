"""Source-witnessed host synchronization inside otherwise opaque operators."""


def cached_prefill_sync_reason(op, scope):
    """Whether this declared native branch must wait for prior GPU work.

    AiterBackend gives PagedAttentionImpl a non-flash layout. With unified
    attention disabled, dispatch_backend selects prefill_attention, whose
    cached branch calls _gather_prefix_and_concat_kv. Its repeat_interleave
    reads GPU context lengths without output_size, synchronizing to determine
    the output allocation size. All work preceding this operator must then
    have completed before its caller can publish a deferred token.

    This does not locate the synchronization inside the opaque operator. The
    cost model uses the prefix of complete preceding operators and leaves
    that unresolved internal work in the remainder. Neither a different
    backend nor an undeclared dispatcher is silently assigned this behavior.
    """
    if op.get("name") != "aiter::unified_attention_with_output_base":
        return None
    from .families.attention import structure_of

    structure = structure_of(op)
    if structure is None or not structure.is_prefill or not structure.has_cached:
        return None
    backend = (scope or {}).get("attention_backend")
    if not isinstance(backend, (dict, list, tuple)):
        return None
    try:
        facts = dict(backend)
    except (TypeError, ValueError):
        return None
    if (facts.get("backend") != "atom.model_ops.attentions.aiter_attention.AiterBackend"
            or facts.get("impl") != "atom.model_ops.attention_mha.PagedAttentionImpl"
            or facts.get("ATOM_USE_UNIFIED_ATTN") not in (False, "False", "false", 0, "0")):
        return None
    return ("PagedAttentionImpl.prefill_attention cached branch calls "
            "_gather_prefix_and_concat_kv: repeat_interleave over GPU context "
            "lengths without output_size synchronizes before returning")
