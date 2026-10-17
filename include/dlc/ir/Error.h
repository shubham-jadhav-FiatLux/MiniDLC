#pragma once

#include <stdexcept>
#include <string>

namespace dlc {

// Raised for malformed IR: parse errors, invalid shapes or attributes, broken invariants.
class IRError : public std::runtime_error {
public:
    explicit IRError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace dlc
