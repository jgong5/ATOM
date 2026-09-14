# Corpus prefix identity for opt-in diagnostics

Cache-aware diagnostic rows retain `session`, `corpus_line_1based`,
`hash_id_scope: "local"`, ordered `hash_ids`, ancestry, arrivals and output
counts. A pinned `prompt_encoding: {path, sha256}` artifact and full
`cache_policy` are required in the manifest. Each row pins
`prompt_token_sha256`; case and execution identities retain the ordered token
digests. Existing selectors, workloads and replay defaults are unchanged.

`cc_traces_local_prefix_v1` admits the pinned 393-root corpus only. Its local
hash IDs have consistent positions and predecessors; they are opaque prefix
nodes, not recovered original token blocks. The codec encodes the 12-bit corpus
root ordinal and 20-bit local ID into eight hexadecimal vocabulary digits.
Two codec/phase markers and four warmup-namespace digits precede padding to
exactly 64 tokens. All distinguishing information fits in the first 14 tokens,
so differing source blocks cannot create an extra native 16-token prefix hit.
Out-of-range IDs refuse; no modulo or truncated hash assigns identities.
Root/model/actor labels do not create additional aliases: the pinned root
ordinal separates roots, while source model and actor do not split a root.

The artifact pins the codec implementation digest and complete canonical fast
tokenizer backend/special-token identity. Twenty distinct ordinary single-token
words supply the digit and marker IDs; runtime verification must reproduce
them. `--prompt-encoding PATH --prompt-encoding-sha256 SHA` sends direct token
IDs, retaining the exact source length without decode/re-tokenize round trips.
Client-local validation constructs no engine requests or future cache demand.

The only implemented cache boundary is fresh after preparation. Warmup uses
distinct per-request namespaces, preserving full prompts and any explicitly
declared diagnostic output cap. After the HTTP preparation drain, replay
requires `/compass/cache/reset` to acknowledge native quiescence, worker
completion and empty KV/state indexes before measured registration/pacing.
The modelled side must be fresh and unprepared, with zero counters, retained
outputs and virtual elapsed/advanced time. No epoch rebase occurs. The complete
receipt is `run.cache_boundary`; the final `/compass/cache` snapshot is
`run.cache_state_after`. Cumulative measurement counters are end minus reset
after-counters. Record draining alone is not a cache reset. There is no reset
between measured events and no change to the ordinary disk compile-cache policy.

This surrogate preserves declared **prompt-prefix** identities. Original output
tokens and their causal inclusion in later prompts are unavailable; Qwen output
must not be declared equivalent to later synthetic input from output counts.
Actual hits can be smaller than the declared common prefix because of arrival,
eviction, the final-block rule and recurrent-state checkpoints. Cache-policy,
state reuse, memory and timing still require fresh paired validation.
