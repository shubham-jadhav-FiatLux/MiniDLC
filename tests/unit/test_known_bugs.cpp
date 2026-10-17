// Known defects, each written as a DISABLED_ test that states the correct behaviour; the phase that fixes one moves it
// into the regular test files. To see the open ones fail: dlc_unit_tests --gtest_also_run_disabled_tests
// --gtest_filter='KnownBugs.*'
//
// None is open. Of the 13 the 2026-10-17 audit found, P1 fixed 5, P2 fixed 5 and P3 fixed 2 (a fused kernel's output
// overlapping its inputs, and parameter buffers being reused; now BufferAssignment.* tests). The last one, the v1 LLVM
// emitter turning division into an addition, went with that emitter in P3; tests/lit/llvm_division.dlc tracks the
// division lowering of its P4 replacement.
#include <gtest/gtest.h>
