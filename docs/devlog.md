# MiniDLC Development Log

## Milestone M0 — Foundations (In Progress)

- **Goals**: Repository skeleton, CMake build system, C++20 Graph IR core, Python frontend tracing, NumPy reference interpreter, test harness, lit configuration.
- **Progress**:
  - Scaffolding repository layout conforming to specification.
  - Setting up CMake configuration with modular targets for IR, passes, runtime, and backends.
  - Defining SSA Graph IR types, operations, builder, verifier, printer, and parser.
  - Providing pure-Python frontend (`minidlc`) and reference interpreter (`minidlc.reference`).
  - Configuring lit golden tests, C++ unit tests, and pytest suite.
