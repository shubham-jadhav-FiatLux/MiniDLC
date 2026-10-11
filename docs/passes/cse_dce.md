# Common Subexpression Elimination (CSE) & Dead Code Elimination (DCE)

## Common Subexpression Elimination (CSE)
- **Algorithm**: Global value numbering across basic blocks.
- Uses a hash of `(OpCode, Operands, Attributes)` to identify duplicate computations.
- Replaces duplicate operation results with existing SSA value references.
- **Course Link**: CS6843 Available Expressions analysis.

## Dead Code Elimination (DCE)
- **Algorithm**: Reverse reachability liveness analysis starting from graph outputs (return values and mutated weights).
- Marks live operations backward through use-def chains.
- Drops all unreached/dead operations with no side effects.
- **Course Link**: CS6843 Liveness analysis and backwards dataflow.
