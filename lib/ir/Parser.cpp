#include "dlc/ir/Parser.h"
#include "dlc/ir/Builder.h"
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace dlc {

void IRParser::skipWhitespaceAndComments() {
    while (pos_ < source_.size()) {
        if (std::isspace(source_[pos_])) {
            pos_++;
        } else if (source_[pos_] == '/' && pos_ + 1 < source_.size() && source_[pos_ + 1] == '/') {
            // Line comment
            pos_ += 2;
            while (pos_ < source_.size() && source_[pos_] != '\n') {
                pos_++;
            }
        } else {
            break;
        }
    }
}

bool IRParser::match(std::string_view expected) {
    skipWhitespaceAndComments();
    if (source_.substr(pos_, expected.size()) == expected) {
        pos_ += expected.size();
        return true;
    }
    return false;
}

std::string IRParser::consumeIdentifier() {
    skipWhitespaceAndComments();
    size_t start = pos_;
    while (pos_ < source_.size() && (std::isalnum(source_[pos_]) || source_[pos_] == '_' || source_[pos_] == '@' || source_[pos_] == '%')) {
        pos_++;
    }
    return std::string(source_.substr(start, pos_ - start));
}

int64_t IRParser::consumeInteger() {
    skipWhitespaceAndComments();
    size_t start = pos_;
    if (pos_ < source_.size() && (source_[pos_] == '-' || source_[pos_] == '+')) {
        pos_++;
    }
    while (pos_ < source_.size() && std::isdigit(source_[pos_])) {
        pos_++;
    }
    return std::stoll(std::string(source_.substr(start, pos_ - start)));
}

Shape IRParser::parseShape() {
    Shape shape;
    if (match("[")) {
        if (!match("]")) {
            while (true) {
                shape.push_back(consumeInteger());
                if (match("]")) break;
                if (!match(",")) break;
            }
        }
    }
    return shape;
}

TypePtr IRParser::parseType() {
    skipWhitespaceAndComments();
    std::string typeName = consumeIdentifier();
    DataType dt = DataType::Float32;
    if (typeName == "f32") {
        dt = DataType::Float32;
    } else if (typeName == "i32") {
        dt = DataType::Int32;
    }
    Shape shape = parseShape();
    return std::make_shared<TensorType>(dt, shape);
}

ModulePtr IRParser::parseModule(std::string_view source) {
    IRParser parser(source);
    return parser.parse();
}

FunctionPtr IRParser::parseFunction(std::string_view source) {
    auto mod = parseModule(source);
    if (mod && !mod->getFunctions().empty()) {
        return mod->getFunctions()[0];
    }
    return nullptr;
}

ModulePtr IRParser::parse() {
    auto mod = std::make_shared<Module>("parsed_module");
    std::unordered_map<std::string, ValuePtr> valueSymbolTable;

    while (pos_ < source_.size()) {
        skipWhitespaceAndComments();
        if (pos_ >= source_.size()) break;

        if (match("func")) {
            std::string funcName = consumeIdentifier();
            if (!funcName.empty() && funcName[0] == '@') {
                funcName = funcName.substr(1);
            }

            match("(");
            std::vector<TypePtr> argTypes;
            std::vector<std::string> argNames;
            if (!match(")")) {
                while (true) {
                    std::string argName = consumeIdentifier();
                    match(":");
                    TypePtr argType = parseType();
                    argNames.push_back(argName);
                    argTypes.push_back(argType);
                    if (match(")")) break;
                    match(",");
                }
            }

            match("->");
            std::vector<TypePtr> resTypes;
            if (match("(")) {
                if (!match(")")) {
                    while (true) {
                        resTypes.push_back(parseType());
                        if (match(")")) break;
                        match(",");
                    }
                }
            } else {
                resTypes.push_back(parseType());
            }

            IRBuilder builder;
            auto func = builder.createFunction(funcName, argTypes, resTypes);
            for (size_t i = 0; i < argNames.size(); ++i) {
                valueSymbolTable[argNames[i]] = func->getArguments()[i];
            }

            match("{");
            while (!match("}") && pos_ < source_.size()) {
                skipWhitespaceAndComments();
                if (pos_ >= source_.size() || source_[pos_] == '}') break;

                std::string token = consumeIdentifier();
                if (token == "return") {
                    std::vector<ValuePtr> retVals;
                    while (!match("\n") && pos_ < source_.size() && source_[pos_] != '}') {
                        skipWhitespaceAndComments();
                        if (pos_ < source_.size() && source_[pos_] == '%') {
                            std::string valName = consumeIdentifier();
                            if (valueSymbolTable.count(valName)) {
                                retVals.push_back(valueSymbolTable[valName]);
                            }
                        } else {
                            break;
                        }
                    }
                    builder.createReturn(retVals);
                    continue;
                }

                // SSA Assignment: %0 = op(...) : type
                std::string resultName = token;
                match("=");
                std::string opName = consumeIdentifier();
                match("(");
                std::vector<ValuePtr> operands;
                if (!match(")")) {
                    while (true) {
                        std::string opValName = consumeIdentifier();
                        if (valueSymbolTable.count(opValName)) {
                            operands.push_back(valueSymbolTable[opValName]);
                        }
                        if (match(")")) break;
                        match(",");
                    }
                }

                TypePtr resType = nullptr;
                if (match(":")) {
                    resType = parseType();
                }

                // Skip attribute dictionary if present: { ... }
                skipWhitespaceAndComments();
                if (match("{")) {
                    int braceDepth = 1;
                    while (pos_ < source_.size() && braceDepth > 0) {
                        if (source_[pos_] == '{') braceDepth++;
                        else if (source_[pos_] == '}') braceDepth--;
                        pos_++;
                    }
                }

                ValuePtr resultVal = nullptr;
                if (opName == "matmul" && operands.size() >= 2) {
                    resultVal = builder.createMatmul(operands[0], operands[1], resType);
                } else if (opName == "add" && operands.size() >= 2) {
                    resultVal = builder.createAdd(operands[0], operands[1], resType);
                } else if (opName == "sub" && operands.size() >= 2) {
                    resultVal = builder.createSub(operands[0], operands[1], resType);
                } else if (opName == "mul" && operands.size() >= 2) {
                    resultVal = builder.createMul(operands[0], operands[1], resType);
                } else if (opName == "div" && operands.size() >= 2) {
                    resultVal = builder.createDiv(operands[0], operands[1], resType);
                } else if (opName == "relu" && operands.size() >= 1) {
                    resultVal = builder.createRelu(operands[0], resType);
                } else if (opName == "gelu" && operands.size() >= 1) {
                    resultVal = builder.createGelu(operands[0], resType);
                } else if (opName == "exp" && operands.size() >= 1) {
                    resultVal = builder.createExp(operands[0], resType);
                } else if (opName == "log" && operands.size() >= 1) {
                    resultVal = builder.createLog(operands[0], resType);
                } else if (opName == "softmax" && operands.size() >= 1) {
                    resultVal = builder.createSoftmax(operands[0], -1, resType);
                } else if (opName == "log_softmax" && operands.size() >= 1) {
                    resultVal = builder.createLogSoftmax(operands[0], -1, resType);
                } else if (opName == "cross_entropy" && operands.size() >= 2) {
                    resultVal = builder.createCrossEntropy(operands[0], operands[1], resType);
                } else if (opName == "layernorm" && operands.size() >= 3) {
                    resultVal = builder.createLayerNorm(operands[0], operands[1], operands[2], 1e-5, resType);
                } else if (opName == "sum" && operands.size() >= 1) {
                    resultVal = builder.createSum(operands[0], -1, false, resType);
                } else if (opName == "mean" && operands.size() >= 1) {
                    resultVal = builder.createMean(operands[0], -1, false, resType);
                } else if (opName == "embedding" && operands.size() >= 2) {
                    resultVal = builder.createEmbedding(operands[0], operands[1], resType);
                } else {
                    auto genericOp = std::make_shared<Operation>(opName, operands, resType ? std::vector<TypePtr>{resType} : std::vector<TypePtr>{});
                    builder.getInsertBlock()->addOperation(genericOp);
                    if (genericOp->getNumResults() > 0) {
                        resultVal = genericOp->getResult(0);
                    }
                }

                if (resultVal) {
                    resultVal->setName(resultName.starts_with("%") ? resultName.substr(1) : resultName);
                    valueSymbolTable[resultName] = resultVal;
                }
            }

            mod->addFunction(func);
        } else {
            // Unrecognized top-level token, skip one char
            pos_++;
        }
    }

    return mod;
}

} // namespace dlc
