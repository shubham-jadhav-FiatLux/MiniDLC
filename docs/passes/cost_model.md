# Cost Model

## Purpose
Count the work and the compulsory memory traffic of each op and of each fused kernel: the inputs of the roofline
report (FLOPs per byte), and the measure of what fusion saves.

## Model (`lib/passes/CostModel.cpp`)
- **FLOPs** from the types: `2·M·N·K` per matmul (per batch); one per output element for elementwise ops (transcendental
  functions count as one); per input element for reductions; GELU, softmax, LayerNorm and cross-entropy count the
  arithmetic of their formula (9, 5, 8 and 5 per element).
- **Bytes**: an op reads each operand once and writes its result once; a fused kernel reads each of its inputs once and
  writes each of its outputs once, and values inside it are free. Real traffic is higher (caches miss, tiles are
  reloaded); this is the lower bound a kernel cannot beat.

`dlc-opt --print-cost` prints the kernels with FLOPs, bytes and intensity, and the total traffic with and without
fusion.

## Status (P3)
`CostModel.CountsFlopsAndTheTrafficFusionSaves` checks the MNIST layer (2·128·256·784 + 2·128·256 FLOPs; the matmul and
add results never reach memory); `tests/lit/fusion.dlc` checks the report. Fusion does not consult the model yet (every
legal merge is profitable on the CPU); the performance report uses it for the roofline (P7).

## Course link
The roofline model (CS6400, ID5130): arithmetic intensity decides whether a kernel is bound by memory or by compute.
