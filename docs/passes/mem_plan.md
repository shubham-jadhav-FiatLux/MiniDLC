# Memory Planning Pass

## Purpose
Assigns memory offsets within a pre-allocated contiguous execution arena for all intermediate tensors, reusing memory across non-overlapping lifetimes to minimize peak footprint.

## Algorithm
1. Compute execution schedule (topological linear order of kernels).
2. Compute liveness intervals \([start_i, end_i]\) for every intermediate value.
3. Build an interference graph where edges represent overlapping intervals.
4. Solve greedy offset assignment (largest tensor size first) minimizing total arena capacity.
5. Analogy to linear scan register allocation (Poletto & Sarkar 1999) and Chaitin-style graph coloring.
