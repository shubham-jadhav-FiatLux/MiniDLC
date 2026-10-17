#pragma once

#include "dlc/ir/Module.h"
#include <string>
#include <vector>

namespace dlc {

struct VerificationResult {
    bool success{true};
    std::vector<std::string> errors;

    void addError(std::string err) {
        success = false;
        errors.push_back(std::move(err));
    }
};

// Checks every op against the op registry (operand count, attributes, inferred result type), definitions before
// uses, a single trailing 'return' matching the signature, and use lists that match the actual uses.
class Verifier {
public:
    static VerificationResult verify(const Module& module);
    static VerificationResult verify(const Function& func);
    static VerificationResult verify(const Operation& op);
};

} // namespace dlc
