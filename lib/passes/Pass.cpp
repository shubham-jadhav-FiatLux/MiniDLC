#include "dlc/passes/Pass.h"
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
        bool changed = pass->runOnModule(module);
        if (changed) {
            anyChanged = true;
        }
        if (printAfterAll_) {
            std::cout << "// --- After Pass: " << pass->getName() << " ---\n";
            std::cout << module.toString() << "\n";
        }
    }
    return anyChanged;
}

} // namespace dlc
