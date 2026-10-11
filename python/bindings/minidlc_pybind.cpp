#include "dlc/ir/Module.h"
#include "dlc/ir/Builder.h"
#include "dlc/ir/Verifier.h"
#include "dlc/ir/Printer.h"
#include "dlc/ir/Parser.h"

// Note: nanobind bindings will be linked when nanobind is built via CMake.
// This file establishes the C++ extension module interface stub.

extern "C" {
    // Export marker for shared library
    const char* minidlc_c_version() {
        return "0.1.0";
    }
}
