#include "dlc/passes/Pass.h"
#include "dlc/ir/Error.h"
#include "dlc/ir/Verifier.h"
#include <iostream>

namespace dlc {

bool Pass::runOnModule(Module& module) {
    bool changed = false;
    for (auto& func : module.getFunctions()) {
        if (runOnFunction(*func)) {
            changed = true;
        }
    }
    return changed;
}

bool Pass::runOnFunction(Function& /*func*/) {
    return false;
}

bool PassManager::run(Module& module) {
    bool anyChanged = false;
    for (auto& pass : passes_) {
        bool changed = false;
        try {
            changed = pass->runOnModule(module);
        } catch (const IRError& e) {
            throw IRError("pass '" + pass->getName() + "': " + e.what());
        }
        if (changed) {
            anyChanged = true;
        }
        if (printAfterAll_) {
            std::cout << "// --- After Pass: " << pass->getName() << " ---\n";
            std::cout << module.toString() << "\n";
        }
        if (verifyEach_) {
            auto res = Verifier::verify(module);
            if (!res.success) {
                std::string message = "verification failed after pass '" + pass->getName() + "':";
                for (const auto& err : res.errors) message += "\n  " + err;
                throw IRError(message);
            }
        }
    }
    return anyChanged;
}

} // namespace dlc
