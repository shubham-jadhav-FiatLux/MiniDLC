#include "dlc/runtime/Plan.h"
#include <cmath>
#include <cstdio>
#include <sstream>

namespace dlc::runtime {

namespace {

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
            continue;
        }
        out += c;
    }
    return out + "\"";
}

std::string number(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string slot(const Slot& s) {
    std::ostringstream os;
    os << "{\"region\": " << quote(s.region == Region::Persistent ? "persistent" : "transient") << ", \"offset\": " << s.offset
       << ", \"bytes\": " << s.bytes << "}";
    return os.str();
}

template <class T> std::string list(const T& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) os << (i ? ", " : "") << values[i];
    os << "]";
    return os.str();
}

} // namespace

std::string toJson(const Plan& plan) {
    std::ostringstream os;
    os << "{\n  \"function\": " << quote(plan.function) << ",\n  \"persistent_bytes\": " << plan.persistentBytes
       << ",\n  \"transient_bytes\": " << plan.transientBytes << ",\n  \"args\": [";
    for (size_t i = 0; i < plan.args.size(); ++i) {
        const PlanArg& a = plan.args[i];
        os << (i ? "," : "") << "\n    {\"name\": " << quote(a.name) << ", \"role\": " << quote(argRoleName(a.role))
           << ", \"dtype\": " << quote(dataTypeToString(a.dtype)) << ", \"shape\": " << list(a.shape)
           << ", \"slot\": " << slot(a.slot) << ", \"init\": " << (a.init ? number(*a.init) : "null") << "}";
    }
    os << "\n  ],\n  \"results\": [";
    for (size_t i = 0; i < plan.results.size(); ++i) {
        const PlanResult& r = plan.results[i];
        os << (i ? "," : "") << "\n    {\"kind\": " << quote(resultKindName(r.kind)) << ", \"arg\": " << quote(r.arg)
           << ", \"dtype\": " << quote(dataTypeToString(r.dtype)) << ", \"shape\": " << list(r.shape)
           << ", \"slot\": " << slot(r.slot) << "}";
    }
    os << "\n  ],\n  \"launches\": [";
    for (size_t i = 0; i < plan.launches.size(); ++i) {
        const Launch& l = plan.launches[i];
        os << (i ? "," : "") << "\n    {\"kernel\": " << l.kernel << ", \"name\": " << quote(l.name) << ", \"grid\": " << list(l.grid)
           << ", \"block\": " << list(l.block) << ", \"parallel_extent\": " << l.parallelExtent << ", \"flops\": " << number(l.flops)
           << ", \"bytes\": " << l.bytes << ", \"args\": [";
        for (size_t j = 0; j < l.args.size(); ++j) os << (j ? ", " : "") << slot(l.args[j]);
        os << "]}";
    }
    os << "\n  ],\n  \"copies\": [";
    for (size_t i = 0; i < plan.copies.size(); ++i) {
        os << (i ? "," : "") << "\n    {\"from\": " << slot(plan.copies[i].from) << ", \"to\": " << slot(plan.copies[i].to) << "}";
    }
    os << "\n  ]\n}\n";
    return os.str();
}

} // namespace dlc::runtime
