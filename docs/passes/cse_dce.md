# Common Subexpression Elimination and Dead Code Elimination

## CSE
- **Algorithm:** value numbering over the function's single block. Each op is keyed by (op name, operand value ids,
  attributes); an op whose key was seen before is replaced by the earlier result.
- **Status (v1):** the key ignores result types, so `constant 1.0 : f32[]` and `constant 1.0 : f32[4,4]` merge; doubles
  are written into the key with 6 significant digits, so nearby constants merge; attributes are read from an unordered
  map. All three are recorded as disabled tests in `tests/unit/test_known_bugs.cpp` (P2).
- **Course link:** available expressions and value numbering (CS6843).

## DCE
- **Algorithm:** mark ops live backwards from the return through use-def chains, then erase every unmarked op.
- **Status (v1):** erased ops stay in their operands' use lists as dangling pointers, so a later pass that walks those
  lists reads freed memory (P1).
- **Course link:** liveness and backward dataflow (CS6843).
