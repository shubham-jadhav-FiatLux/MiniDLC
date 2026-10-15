# MiniDLC Development Log

## Milestone M0 — Foundations (Completed 2026-10-11)

- **Goals**: Repository skeleton, CMake build system, C++20 Graph IR core, Python frontend tracing, NumPy reference interpreter, test harness, lit configuration.
- **Completed**:
  - Scaffolding repository layout conforming to specification.
  - Setting up CMake configuration with modular targets for IR, passes, runtime, and backends.
  - Defining SSA Graph IR types, operations, builder, verifier, printer, and parser.
  - Providing pure-Python frontend (`minidlc`) and reference interpreter (`minidlc.reference`).
  - Configuring lit golden tests, C++ unit tests, and test suites.

---

## Milestone M1 — Graph passes + autodiff; MNIST through interpreter (Completed 2026-10-15)

- **Goals**:
  - The full op set with forward shape inference and broadcasting.
  - Canonicalization, constant folding, CSE, DCE.
  - Reverse-mode automatic differentiation (autodiff) pass with Vector-Jacobian Product (VJP) rules (§1.5).
  - Fan-out adjoint accumulation and broadcast adjoint reduction.
  - SGD and Adam optimizer lowering.
  - Double precision (`float64`) central finite-difference gradient checking suite for all ops.
  - Train MLP on MNIST through reference interpreter to ~97% test accuracy.
- **Completed & Verified**:
  - Full op set (~16 ops) implemented in C++ IR, builder, printer, parser, and verifier.
  - Shape inference pass evaluating abstract interpretation over shape lattice with broadcasting.
  - Algebraic canonicalization and constant folding (`x + 0 -> x`, `x * 1 -> x`, constant arithmetic folding).
  - Parameterized Common Subexpression Elimination (CSE) with attribute-aware value numbering, and Dead Code Elimination (DCE).
  - Reverse-mode autodiff pass in C++ and Python generating backward graphs with fan-out accumulation and broadcasting reductions.
  - All op VJP gradient checks passed against double precision central differences (`rtol=1e-4`, `atol=1e-4`).
  - Lit golden IR test suite passed (`autodiff.dlc`, `basic_relu_matmul.dlc`, `shape_infer.dlc`).
  - MLP trained on full MNIST dataset through the reference interpreter, reaching **96.93% test accuracy** on the 10,000-image test set.
