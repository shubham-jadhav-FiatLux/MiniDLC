# Common Subexpression Elimination and Dead Code Elimination

## CSE
- **Algorithm:** value numbering over the function's single block. Each op is keyed by (op name, operand value ids,
  attributes); an op whose key was seen before is replaced by the earlier result.
- **Status (P1):** replaced ops are erased through `Block::eraseOperation`, so use lists stay consistent
  (`CSE.ErasedDuplicatesLeaveConsistentUseLists`), and the pass is translation-validated on 16 exported graphs. Still
  wrong (P2): the key ignores result types, so `constant 1.0 : f32[]` and `constant 1.0 : f32[4,4]` merge
  (`KnownBugs.DISABLED_CSEKeepsConstantsWithDifferentTypes`); doubles enter the key with 6 significant digits, so
  `0.1000001` and `0.1000002` merge (`KnownBugs.DISABLED_CSEKeepsNearbyConstantsApart`, and the `nearby_constants`
  translation-validation case). Attributes are read from an unordered map, which can hide a match but not cause one.
- **Course link:** available expressions and value numbering (CS6843).

## DCE
- **Algorithm:** mark ops live backwards from the return through use-def chains, then erase every unmarked op.
- **Status (P1):** ops are erased users-first through `Block::eraseOperation`, which unlinks each op from its
  operands' use lists and refuses to erase an op whose result is still used (`DCE.UnlinksErasedOpsFromTheirOperands`,
  `OperationHygiene.ErasingUnlinksAndRefusesLiveOps`). The verifier checks every use list. Translation-validated on 16
  exported graphs.
- **Course link:** liveness and backward dataflow (CS6843).
