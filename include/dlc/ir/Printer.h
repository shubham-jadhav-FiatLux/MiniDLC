#pragma once

#include "dlc/ir/Module.h"
#include <ostream>
#include <string>

namespace dlc {

class IRPrinter {
public:
    explicit IRPrinter(std::ostream& os) : os_(os) {}

    void print(const Module& module);
    void print(const Function& func);
    void print(const Operation& op);
    void print(const Value& val);
    void print(const Type& type);

    static std::string toString(const Module& module);
    static std::string toString(const Function& func);
    static std::string toString(const Operation& op);

    // Graphviz export
    static void printDot(const Function& func, std::ostream& os);

private:
    std::ostream& os_;
};

} // namespace dlc
