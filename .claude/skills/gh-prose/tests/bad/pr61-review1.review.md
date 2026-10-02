## Review record — IR-2, round 1, agent-authored

**Verdict: REQUEST CHANGES.** One blocking defect: **the module installs a guard**, on the one line that exists to stop it. Everything else I ran reproduces — the named result twice over, both gate figures, the `+30` split, and the effort arithmetic to the line.

Seven findings are inline. This comment carries the verdict, the re-measurements, the ruling on the four key gaps, and what the capture side has to emit.

Measured on node 18, `xiaobizh_n18_cpu`, torch 2.10.0+rocm7.2.4. Both trees staged independently with `git archive` + `docker cp` into `/tmp/xiaobizh-ir2rev/{ctl,br}` (not the shared path), `.compass-commit` and `.compass-changed` written from the same `rev-parse`, md5 verified host → node 18 → container, `PYTHONPATH` asserted by `import atom` before any figure was read.

### The blocking finding

**`hash()` is the key's only gate against a symbolic value, and on this build it is the gate that mutates the record.** [inline](https://github.com/jgong5/ATOM/pull/61#discussion_r4063774176)

```
built SymBool (tokens > 4096)                     guards = 0   []
hash(SymBool)                           -> 0      guards = 1   ['s93 <= 4096']
GuardedApplicability(key={'mode':'decode','has_cached':<SymBool>})
                                                  guards = 0 -> 1
sorted([('a', <SymBool>), ('b', 1)])              guards = 0 -> 0
```

`torch/__init__.py:813-818` says why, in its own comment: a non-constant `SymBool` hashes by `# Force specialization` → `hash(builtins.bool(self))`. The sort is clean; `hash` on line 223 is what does it. Constructing the predicate therefore narrows the tracing run it was built from.

The sibling type is why this was easy to miss, and it is the same line of code in torch: `SymInt.__hash__` **raises** — IR-1's finding (a) — which is why the `sizes` path is safe. `SymBool.__hash__` specializes. One gate, two halves of the key's type surface, opposite failures.

Two consequences follow from the same line, both measured and both filed inline: `decide` **raises rather than returning a verdict** for a `SymBool` key against a Python bool (`TypeError: BooleanAtom not allowed in this context`) and for a multi-element `torch.Tensor` key (`RuntimeError: Boolean value of Tensor ... is ambiguous`) — `torch.Tensor` passes the hash gate on identity, and `ctx` is exactly the field whose natural form is an array. A positive type list in `_as_key` — `bool`, `int`, `str`, `bytes`, `float`, `None`, `Enum`, tuples of those — closes all three at one point and needs no torch import.

Reachability, stated plainly: today's `produces_output()` is `any(is_final_chunk)` over a Python list and `has_cached` in `backends.py` is a Python loop, so on today's code both are plain bools and I have **not** shown ATOM produces a `SymBool` key. What I have shown is that the module's only defence is the leak, that capture runs under `FakeTensorMode` where a bool over a symbolic size is a `SymBool`, and that the PR body's strengthening — *"the wrong idiom cannot happen inside the module, because the value never gets there"* — is false for the key half. IR-1 is fixing the sibling gate now; this one is cheap now and not cheap once #58 writes a capture path against it.

### Did the no-guard property otherwise hold?

**Yes, everywhere else I could reach, and I went looking for the IR-1 class of miss specifically.** Every one of these was `guards 1 -> 1`:

`repr(applicability)` · `hash(applicability)` · `rec == other` · `rec in [other]` · `{rec, other}` · the `sorted(pairs)` in `_as_key` · `from_shape_env`'s two attribute reads · `guard.subs(...)` · `bool(resolved)` · `_shown`'s format string · `dict(self.key)` and every `in`/lookup on it · `_key_refusal`'s `!r` on key values · the range comparison · `decide` with an extra unnamed binding, with `sizes` as a list of pairs, with a negative size.

The tuple-comparison short-circuit that hid IR-1's defect is not load-bearing here: `_as_key` checks name uniqueness *before* sorting, so `sorted` never reaches a value, and the `!=` on line 140 is reached only by values the hash gate already admitted. Which is why the gate is the finding.

### The named result, reproduced

One tracing run per path, branching on `tokens > 128`, at `f4a4fc3ad`:

```
T=17  -> guards ['s93 <= 128']  var_to_range {s93: VR[2, 128]}
        s93=17   True
        s93=512  False  guard `s93 <= 128` is false at s93=512
        s93=1    False  the record holds s93 in [2, 128]; the step binds s93=1
        othermode False  mode was recorded as 'decode' and the step states 'other'
        guards: before=1 after_record=1 after_decide=1
        int(SymInt) instead: 1 -> 2  ['s93 <= 128', 'Eq(s93, 17)']

T=512 -> guards ['s93 > 128']   var_to_range {s93: VR[129, int_oo]}
        s93=17   False  guard `s93 > 128` is false at s93=17
        guards: before=1 after_record=1 after_decide=1
        int(SymInt) instead: 1 -> 2  ['s93 > 128', 'Eq(s93, 512)']
```

Identical to the PR body, symbol name included. Five refusal kinds checked; each names **both** the condition and what the step offered, and the two the PR body did not quote do too (`guard ... is over ['s93'], which the step does not bind; it binds ['other']`, and the unbound-range form). `decide` returns a verdict on every mismatch I could construct **except** the two raise paths above.

### Does the `SymInt` refusal lock out a legitimate caller?

**No.** A `SymInt` always carries its own resolution — `tok.node.hint` and `shape_env.size_hint(tok.node.expr)` both returned 17 with `guards: 1 -> 1`. The only caller it locks out is one holding an *unbacked* symbol, where `size_hint` raises `GuardOnDataDependentSymNode` — and that step has no size to decide against in the first place. Refusal accepted, and it is the right strengthening over the brief. The message is wrong for three non-symbolic types that also reach it ([inline](https://github.com/jgong5/ATOM/pull/61#discussion_r4063775510)) — `np.int64` in particular, which is what `ScheduledBatch`'s arrays yield.

### Does this PR depend on IR-1's two defects?

**On (a), no — and not by luck, by using a different predicate.** `_as_bindings` gates on `is_symbolic`, which is `not isinstance(dim, int)`: an `isinstance`, not a `hash`, so IR-1's `as_dim` hashability problem cannot reach it. `applicability.py` never calls `as_dim`. If IR-1 changes what the canonical symbolic-dimension form is, `is_symbolic` as written survives it — anything that is not a Python `int` is refused as a size, whatever class the capture side settles on. **But the same fix does not transfer**: IR-1 will fix a gate that *refuses too much*, and this module's gate *accepts too much*, so a fix propagated from IR-1 will not close finding 1. They have to be decided together and resolved differently.

**On (b), no.** `GuardedApplicability`'s generated `__eq__` compares `key`, `guards` and `ranges` — never a `Shape` or an `Op` — and `rec == other`, `rec in [other]` and `{rec, other}` were all guard-clean.

**The duck-typing claim checks out, and it is true rather than convenient.** `applicability.py` imports `dataclasses`, `typing`, and two relative modules; the allowlist is `{abc, collections, dataclasses, enum, math, string, typing}` and `ImportFrom` with `node.level` set is skipped, so the test genuinely needed no edit and the reason is genuinely the no-torch/no-sympy duck-typing, not an oversight. A guard is "anything with `free_symbols` and `subs`" (line 247), which is the same shape of commitment `shapes.py` makes and is independent of what IR-1 settles on. Keep it.

### The one thing IR-1's review asked this PR to close, and it did not

IR-1's round-1 review, item 3 to #55: *"you have to add the abstract method anyway; adding it now closes the `Applicability()` hole."* It is still open — measured:

```
Applicability()                                  -> <Applicability object at 0x...>
Graph(applicability=Applicability(), region=Op(...))  -> accepted
```

An `abc.ABC` with no abstract method is instantiable, so `Graph`'s `isinstance` check admits a validity statement that cannot decide anything — the exact failure the field exists to prevent, one constructor call away. This PR defines the only `decide` there is; promoting it to `@abc.abstractmethod` on the base is a two-line change in `graph.py` and belongs here. (`graph.py` is unchanged in this diff so I could not anchor this inline.)

### The four gaps in ATOM's emitted key — all four confirmed

I read `run_labels.py` rather than the PR's table, and checked `_detailed_label_suffix` too, since it is the one field that could have quietly closed a gap. It cannot: `model_runner.py:2638` emits only ` sqsq= sqsk= sk=`, and only when `batch.detailed_sqsq is not None`, which needs profiling active *and* `ATOM_ENABLE_DETAILED_ANNOTATION`.

| Gap | Confirmed | Evidence |
|---|---|---|
| 1. `has_cached` | yes | no label field. Eager path emits `ctx=`, truncated to `[a,b,c]...+N` past five seqs (`run_labels.py:88-91`); the cudagraph decode branch emits `p=`/`d=` and no `ctx` at all. `AttentionMetadata.has_cached` is set in `backends.py:471-478` and branched on in `attention_mha.py:742,808`, `triton_mla.py:153`, `aiter_attention.py:880`, `aiter_mla.py:1199,1244,1350,1388`, `attention_mla.py:2428` |
| 2. `produces_output()` | yes | no label field at all. `scheduler.py:823`; skipped on at `model_runner.py:482`, `model_runner.py:3327` (`_is_pure_middle_chunk`), `pp_engine_core.py:108,384` |
| 3. attention backend | yes | appears nowhere in `build_run_label` |
| 4. spec width | yes | `spec=` is inside the `if use_cudagraph` branch **and** behind `if batch.num_spec_step > 0`, so `spec=0` and "not speculating" are one absence, and an eager or prefill step that speculates says nothing |

**Gap 2 carries the extra weight, and the weight is quantified rather than argued.** The engine rule is `any(self.is_final_chunk)` — ANY, not ALL — and it is not geometric. On this project, a `produces_output` reconstructed from geometry (`context == query`) disagreed with the scheduler's own answer on **158 of 405 batches (39.0%)**, and getting the quantifier wrong turned 1 refused region key / 2 batches into 3 keys / 44 batches. It is also already a live region-model key: `(1, 16384, False)` is a measured cell and `(1, 16384, True)` is not, so the identical 16384-token chunk prices as a middle chunk and refuses when it samples — because a middle chunk skips `postprocess` entirely, which is zero work rather than a small measurement. "No label field at all" is therefore not a tidy gap; it is the gap with a measured 39% misfile rate behind it.

**What the recording side has to emit** — the ask, concretely:

1. `ScheduledBatch.produces_output()` itself, or `is_final_chunk` **per request**, never a geometric gloss and never the ALL quantifier.
2. `AttentionMetadata.has_cached` as a bool. Not reconstructable from `ctx=`: the eager label truncates past five sequences and the replayed decode label has no `ctx` at all.
3. The attention backend identity, from the metadata builder.
4. `batch.num_spec_step` unconditionally, on every path, including zero.

None of these has an owning issue today — #15 (P0.4/T5) is the closest and is a capture task, not a key task. The PR's "What is not here" section names the work correctly; it should be carried into #55's handoff comment so it does not live only in a PR body.

### The structural claim: right division, incomplete handoff

**Right division.** A predicate that hard-coded the eight field names would be a second, *authored* statement of ATOM's control flow living beside the produced one, which is the thing the design refuses; and symmetric exact-match converts every omission into a named refusal instead of a silent admission, which is the strongest thing a decider can do without owning the taxonomy. Keep it.

**Incomplete handoff.** Symmetry buys "nobody skipped a condition". It buys nothing against "both sides computed the same condition wrongly" — two sides agreeing on a geometric `produces_output` pass this module cleanly and are wrong on 39% of batches. That residual is the whole of gap 2 and it is not visible from inside this module, so it has to be written into the capture task's brief rather than left for its author to rediscover. That is a handoff gap, not a code change.

### The two decisions the brief did not cover

**1. Guards before ranges — correct, and the refusal really is the more useful one.** I checked both orders against both records. The order only matters when both fire: at the prefill record, `s93=1` refuses as `` guard `s93 > 128` is false at s93=1 `` (the branch the tracing run took) where ranges-first would have said `the record holds s93 in [129, int_oo]` — true, but not the branch. And the ranges are not redundant: at the decode record, `s93=1` passes the guard `s93 <= 128` and is caught only by `[2, 128]`. Both halves earn their place, in this order. Accepted.

**2a. Key field sets must match in both directions — accepted, unreserved.** It is the stricter reading and it is the one that matches refusing rather than falling back.

**2b. Every symbol the domain names must be bound — accepted in principle, and as written it refuses every step on the model this IR was cut for.** [inline](https://github.com/jgong5/ATOM/pull/61#discussion_r4063774908) `from_shape_env` copies all of `var_to_range`, which holds an entry for every symbol the tracer created — branched-on or not, backed or not. An unbacked `u0` (the MoE per-expert count) lands there as `VR[-int_oo, int_oo]` with no guard, and `decide` then refuses every step naming a symbol no step can bind, with a message indistinguishable from a legitimate domain miss. A trivially-satisfiable range that is nonetheless mandatory buys no information and costs every decision. The author's placement — narrowing belongs on the recording side — is right; "later" is wrong, because this is the designed-for case and not a hypothetical. The cheap form is a **record-time** refusal or drop in `from_shape_env`, which names the problem once at the point that can act on it, rather than once per step at the point that cannot.

### Gates and effort, re-measured

| | commit | passed | skipped | xfailed | `pytest rc` | `GATE_CPU_RC` |
|---|---|---|---|---|---|---|
| control (#60's head) | `a2b7e91d4` | 4154 | 149 | 3 | 0 | 0 |
| branch | `f4a4fc3ad` | **4184** | 149 | 3 | 0 | 0 |
| delta | | **+30** | 0 | 0 | | |

Both figures reproduce exactly, and the control reproduces IR-1's own recorded number. The `+30` split is confirmed rather than accepted: `tests/compass/test_ir_applicability.py` alone is **29 passed**, and `test_ir_data_model.py` goes **70 → 71** between the two trees — the parametrised import check gaining exactly one module. Not the one-in-four flake, which moves one test between passed and skipped and would show in the skipped column. `.compass-changed` on the branch snapshot lists exactly the seven files; `gpu: not required` on both. `ruff check` and `black --check` clean on all three files.

Effort, by the PR's own AST protocol, reproduces to the line:

| | AST | file |
|---|---|---|
| `applicability.py` | **114** | 300 |
| `__init__.py` | 4 → 5 (+1 AST, +3 file lines, all additive) | |
| production total | **115** | |
| `test_ir_applicability.py` | **142** | 276 |

115 against 150-200 is **under**, as claimed — 0.77x of the lower bound, which is not a halt event in either direction. 16 `raise` statements. The one caveat IR-1's review raised about this metric applies here too and in the same direction: `ast.unparse` collapses implicit string concatenation to one line, so a module that is mostly named refusals undercounts.

### Accepted with reservation

- **`Verdict.__post_init__` refusing an unexplained refusal.** Right, and tested. The reservation is that it converts a `_key_refusal` that returns `""` into a `ValueError` escaping `decide` — the second raise-not-verdict path in the module. I could not construct one (`nan` produces a confusing but non-empty reason), so it is a latent coupling, not a defect.
- **A key with no conditions refused at construction.** Well argued, well tested, no objection.
- **A guard that is not a boolean expression.** `guards=(Symbol('s93'),)` subs to `17`, `bool(17)` is True, admitted; `Symbol('s93') - 17` refuses as `` guard `s93 - 17` is false ``. `from_shape_env` only ever produces relationals, so this is reachable only by hand-construction. Not asking for it this round.
- **An inverted range** `(10, 2)` is accepted at construction and refuses everything. Same class; `_as_ranges` has the information to refuse it.
- **Extra bindings in `sizes` are silently allowed** while extra fields in `key` are refused by name. Defensible — a size no guard and no range mentions is a size the graph did not depend on — but it is the opposite ruling to decision 3 on the same page, and the docstrings do not say why the two differ.

### What the next task in this area should watch

1. **#56 / #57:** nothing here caches or indexes over graphs, and `decide` is a few substitutions — a loop over `decide` for ~5 structures is the right shape and needs nothing else. Do not build a selection index keyed on node identity; IR-1's finding 2 and this PR's finding 1 are the same hazard from two directions.
2. **The capture task (no issue yet):** the four fields above, plus the `SymBool` question — if the key is built inside the tracing context, finding 1 is live rather than latent.
3. **#55's handoff:** carry the "both sides can be consistently wrong" residual on gap 2 forward. It is invisible from inside this module by design.

### Not checked

- Nothing ran on a GPU; the gate reported `gpu: not required` from the stamp on both trees.
- No end-to-end `FakeTensorMode` capture of an ATOM forward, so whether `has_cached` / `produces_output` arrive as `SymBool` under capture is inferred from the type semantics, not observed. That is the one assumption behind finding 1 that the capture author could overturn — or confirm — with a real trace.
- The symbolic-range-bound path (finding 3) is a hole in the type as constructed; I could not produce a symbolic `var_to_range` bound from a real `ShapeEnv` on this build and say so inline.
- I did not re-derive the `attention_mha` / `triton_mla` / `aiter_attention` branch behaviour on `has_cached`; I confirmed the call sites exist and branch, not what each branch costs.

**No design-doc references in code or in emitted strings** — I grepped both new files for `D19`/`D20`/`T\d`/`P0.x`/`principle`/`Gate N`/doc filenames, including inside the refusal text where they would most naturally leak. Zero hits.
