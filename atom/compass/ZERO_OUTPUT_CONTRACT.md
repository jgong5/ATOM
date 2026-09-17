# Zero-output events in diagnostic length-and-arrival replay

A new diagnostic manifest containing any `output_tokens: 0` row must declare:

```json
"zero_output_semantics": "full_prompt_zero_retained_output_v1"
```

This contract retains the event's entire reconstructed prompt, native request
identity/ancestry and arrival offset. Replay requests `max_tokens=0`; the target
engine performs normal prompt processing and terminates with zero retained
output. The event is not converted to a no-op, cancelled at a source-derived
time, raised to one token, or removed with its root. Its completion, terminal
latency, resource/state effects and contribution to the workload's throughput
window remain part of the experiment.

TTFT applies only to requests producing at least one token; TPOT applies only
to those producing at least two. A zero-output engine record must retain its
arrival and finish timestamps, with first-token time and TTFT undefined (`null`).
The comparison reports each metric's request count and, for zero-containing
workloads, the contributing request indices. An all-zero workload has zero
output throughput and an undefined relative throughput error, not a passing
zero-percent error. Positive-output timestamp and completeness checks remain.

This is a declared **target surrogate**, not a claim about the source's unknown
failure, cancellation, requested cap or amount of model work. The pinned corpus
has 28 zero-output API leaves across 14 roots, without API termination reasons;
15 even report positive source TTFT. Source durations remain metadata, not
service costs to impose on the target. The declaration is pinned in diagnostic
case identity and carried with execution evidence.

Frozen registered workloads and their historical selection/gates are unchanged.
This contract enables new diagnostic validation; it supplies no GPU execution,
model-domain support, memory result or whole-corpus accuracy proof by itself.
