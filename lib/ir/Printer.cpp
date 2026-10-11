#include "dlc/ir/Printer.h"
#include <sstream>

namespace dlc {

void IRPrinter::print(const Module& module) {
    os_ << module.toString();
}

void IRPrinter::print(const Function& func) {
    os_ << func.toString();
}

void IRPrinter::print(const Operation& op) {
    os_ << op.toString();
}

void IRPrinter::print(const Value& val) {
    os_ << val.toString();
}

void IRPrinter::print(const Type& type) {
    os_ << type.toString();
}

std::string IRPrinter::toString(const Module& module) {
    return module.toString();
}

std::string IRPrinter::toString(const Function& func) {
    return func.toString();
}

std::string IRPrinter::toString(const Operation& op) {
    return op.toString();
}

void IRPrinter::printDot(const Function& func, std::ostream& os) {
    os << "digraph \"" << func.getName() << "\" {\n";
    os << "  rankdir=TB;\n";
    os << "  node [shape=box, style=filled, fillcolor=white, fontname=\"Courier\"];\n";

    // Print arguments
    for (const auto& arg : func.getArguments()) {
        os << "  \"" << arg->toString() << "\" [label=\"" << arg->toString() << "\\n"
           << arg->getType()->toString() << "\", shape=ellipse, fillcolor=\"#e0f7fa\"];\n";
    }

    const Block* entry = func.getEntryBlock();
    if (entry) {
        size_t opIdx = 0;
        for (const auto& op : entry->getOperations()) {
            std::string opNode = "op_" + std::to_string(opIdx++);
            os << "  \"" << opNode << "\" [label=\"" << op->getOpName() << "\", fillcolor=\"#fff9c4\"];\n";

            // Edges from operands to op
            for (const auto& operand : op->getOperands()) {
                if (operand) {
                    os << "  \"" << operand->toString() << "\" -> \"" << opNode << "\";\n";
                }
            }

            // Edges from op to results
            for (const auto& result : op->getResults()) {
                if (result) {
                    os << "  \"" << result->toString() << "\" [label=\"" << result->toString() << "\\n"
                       << result->getType()->toString() << "\", shape=ellipse, fillcolor=\"#f1f8e9\"];\n";
                    os << "  \"" << opNode << "\" -> \"" << result->toString() << "\";\n";
                }
            }
        }
    }

    os << "}\n";
}

} // namespace dlc
