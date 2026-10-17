# Common Subexpression Elimination and Dead Code Elimination

## CSE
- **Algorithm:** value numbering over the function's single block. Each op is keyed by its name, its operand values
  (by identity), its result type and its attributes (doubles by their bit pattern, keys sorted); an op whose key was
  seen before is replaced by the earlier result and erased. Canonicalization runs first, so commutative operands are
  already in order.
- **Status (P2):** the v1 defects are fixed — the key includes the result type (`constant 1.0 : f32[]` and
  `constant 1.0 : f32[4,4]` stay apart), doubles are exact (`0.1000001` and `0.1000002` stay apart), attribute order
  no longer matters, and erased ops leave consistent use lists. Tested by `CSE.*` unit tests and translation
  validation on 16 exported graphs.
- **Course link:** available expressions and value numbering (CS6843).

## DCE
- **Algorithm:** mark ops live backwards from the return through use-def chains, then erase every unmarked op.
- **Status (P1):** ops are erased users-first through `Block::eraseOperation`, which unlinks each op from its
  operands' use lists and refuses to erase an op whose result is still used (`DCE.UnlinksErasedOpsFromTheirOperands`,
  `OperationHygiene.ErasingUnlinksAndRefusesLiveOps`). The verifier checks every use list. Translation-validated on 16
  exported graphs.
- **Course link:** liveness and backward dataflow (CS6843).
