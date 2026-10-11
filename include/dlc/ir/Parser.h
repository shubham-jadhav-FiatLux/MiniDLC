#pragma once

#include "dlc/ir/Module.h"
#include <string>
#include <string_view>
#include <memory>

namespace dlc {

class IRParser {
public:
    explicit IRParser(std::string_view source) : source_(source), pos_(0) {}

    static ModulePtr parseModule(std::string_view source);
    static FunctionPtr parseFunction(std::string_view source);

    ModulePtr parse();

private:
    std::string_view source_;
    size_t pos_{0};

    void skipWhitespaceAndComments();
    std::string nextToken();
    bool match(std::string_view expected);
    std::string consumeIdentifier();
    int64_t consumeInteger();
    TypePtr parseType();
    Shape parseShape();
};

} // namespace dlc
