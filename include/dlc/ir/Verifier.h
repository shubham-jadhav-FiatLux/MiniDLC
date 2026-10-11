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

class Verifier {
public:
    static VerificationResult verify(const Module& module);
    static VerificationResult verify(const Function& func);
    static VerificationResult verify(const Operation& op);
};

} // namespace dlc
