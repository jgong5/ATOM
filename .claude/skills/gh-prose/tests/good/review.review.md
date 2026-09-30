REQUEST CHANGES: 1 blocking. `_as_key` calls `hash()` on key values, and `hash(SymBool)` installs a guard, so building the predicate narrows the trace it came from.

Checked: named result, both gate counts and effort re-run on node 18; all reproduce.
Accepted with reservation: none.
Watch next: #58's capture path runs under `FakeTensorMode`, where key values can be `SymBool`.
