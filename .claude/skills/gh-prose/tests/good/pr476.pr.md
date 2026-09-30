Closes #454. Blocked on #454: which mechanism does the `flush_pp_send` pair get? `need human` applied.

Adds `mechanism` and `mechanism_why` to every row of `sync_sites.json` so each sync site states how it waits, and adds `send_multipart` to the scanner.

The two `flush_pp_send` calls in `_downstream_busy_loop` are textually identical, so the inventory requires one answer, but the mapping gives K1 (idle flush) and K9 (shutdown flush). The only red is `test_calls_that_share_an_ordinal_are_answered_the_same_way`. Option (a) is a one-row data change; (b) is a one-line test change. Details on #454.

## Check these
- The 8 new `send_multipart` rows carry no `category`; `counts_by` skips missing fields so the old count table stays valid.
- `mechanism_why` is required only where the mechanism departs from the category.
- Two consistency rules the issue proposed cannot be stated with existing fields (no LP or bound field); checks use `peer` and the call expression instead.

## Dev record
- Found: the scanner reports 8 `send_multipart` sites, not 2.
- Decided: `mechanism_why` translated to English.
- Surprised: none beyond the blocked pair.
- Left undone: the ruling; per-mechanism counts on #450 go stale (noted there).

## Named result
<details><summary>Mechanism x category crosstab</summary>

```
category  K1 K2 K3 K4 K5 K6 K7 K8  K9 total
total     18  6  6 13 10 35  8  5 119   220
```
</details>

## Gates
- branch @ `67117f738`: 5286 passed, 1 failed (the blocked pair). ruff and black clean.
- control @ `f87413a7a`: 5281 passed, 0 failed.

<details><summary>Evidence</summary>

Revert-red table (M1-M10) and cost vs estimate go here.
</details>
