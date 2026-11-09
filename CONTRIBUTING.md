# Contributing to MiniDLC

Thanks for your interest. MiniDLC is a teaching-scale compiler, so changes are welcome when they keep it small, tested
and honest about what it does.

## Getting started

Build and test as described in the README's [Quickstart](README.md#quickstart): LLVM 23 development files, CMake ≥ 3.20,
a C++20 compiler, and Python ≥ 3.11 with NumPy, nanobind, pytest and lit.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3)
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Before opening a pull request, also run the sanitizer build, which CI runs too:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DDLC_SANITIZE=ON -DDLC_BUILD_PYTHON=OFF
cmake --build build-asan -j4 && ctest --test-dir build-asan --output-on-failure
```

## What a change needs

- **A test that fails without it.** A bug fix comes with a test that reproduces the bug. A new pass or op is checked by
  what it produces, not by whether it ran:
  - a graph pass by translation validation (running the graph before and after it);
  - an op through the generated code on every target against the NumPy reference (`tests/python/`);
  - a loop pass against the interpreter, with a lit/FileCheck test for its output (`tests/lit/`).
- **Docs that match the code.** If you change what a pass does, update its page in `docs/passes/`, and the status table
  in `docs/architecture.md` if a component's status changes. Performance claims cite the script and machine that
  produced them (`docs/perf-report.md`).
- **Regenerated GPU artifacts** if you change lowering or code generation: `PYTHONPATH=python python3
  examples/gpu_artifacts.py`. `test_committed_artifacts_are_current` fails until you do.

## Style

- C++20; follow the surrounding code (4-space indentation, lines of about 120 columns). Errors are reported with
  `dlc::IRError` (or a subclass) and a message that names the op or kernel.
- Output must not depend on the compiler that builds MiniDLC. Never pass two arguments that each create an IR op
  or emit an LLVM instruction to one call (`b.createSub(b.createSoftmax(x), b.createOneHot(y))`): C++ leaves their
  evaluation order unspecified, so GCC and Clang would emit the ops in different orders, and with them the kernel
  order and the PTX. Build such operands in separate statements (braced initializer lists are evaluated in order).
  CI builds with Clang; build with GCC too if you can.
- Python: type hints on public functions, docstrings on public modules and classes, NumPy for numerics.
- No new runtime dependencies in the core: PyTorch is used only as a reference in tests and benchmarks.

## Commits and pull requests

- Keep each commit focused, and write the subject in the imperative with a
  [Conventional Commits](https://www.conventionalcommits.org/) prefix (`feat:`, `fix:`, `test:`, `docs:`, `perf:`,
  `refactor:`, `build:`, `ci:`); explain *why* in the body.
- Open pull requests against `main`. CI (Release, Debug + ASan/UBSan, Python 3.11 and 3.13) must pass.
- Describe what you tested, and include before/after numbers for performance changes.

## Reporting bugs

Open an issue with the smallest model or `.dlc` file that reproduces it, the target (`cpu`, `gpu-sim`, `gpu`,
`reference`), the command you ran and its output, and your LLVM and Python versions. For security issues, see
[SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE), and to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
