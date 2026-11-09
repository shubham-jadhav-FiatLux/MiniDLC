#include "dlc/backends/llvm/KernelCache.h"
#include "dlc/backends/llvm/Emitter.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/SHA256.h>
#include <random>
#include <sstream>
#include <unistd.h>

namespace dlc::backends::llvm_emit {

namespace fs = std::filesystem;

namespace {

// Only files named like entries (kind-<64 hex digits>) or their temporaries belong to the cache: the directory may be
// one the user chose, with other files in it.
bool isEntry(const std::string& name) {
    auto dash = name.find('-');
    if (dash == std::string::npos || name.size() != dash + 1 + 64) return false;
    return name.find_first_not_of("0123456789abcdef", dash + 1) == std::string::npos;
}

bool isTemporary(const std::string& name) {
    auto tmp = name.find(".tmp.");
    return name.starts_with(".") && tmp != std::string::npos && isEntry(name.substr(1, tmp - 1));
}

} // namespace

KernelCache& KernelCache::instance() {
    static KernelCache cache;
    return cache;
}

KernelCache::KernelCache() { configure(); }

void KernelCache::configure() {
    const char* off = std::getenv("MINIDLC_CACHE");
    enabled_ = !(off && (std::string(off) == "off" || std::string(off) == "0"));
    if (const char* dir = std::getenv("MINIDLC_CACHE_DIR"); dir && *dir) {
        dir_ = dir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        dir_ = std::string(xdg) + "/minidlc";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
        dir_ = std::string(home) + "/.cache/minidlc";
    } else {
        enabled_ = false;
    }
}

std::string KernelCache::key(const std::string& kind, const std::string& content) const {
    llvm::SHA256 sha;
    std::string header = kind + "\n" + codegenBuildId() + "\nLLVM " LLVM_VERSION_STRING "\n";
    sha.update(header);
    sha.update(content);
    auto digest = sha.final();
    return kind + "-" + llvm::toHex(digest, /*LowerCase=*/true);
}

std::optional<std::string> KernelCache::load(const std::string& key) {
    if (!enabled_) return std::nullopt;
    std::ifstream in(dir_ + "/" + key, std::ios::binary);
    if (!in) {
        ++misses_;
        return std::nullopt;
    }
    std::ostringstream bytes;
    bytes << in.rdbuf();
    ++hits_;
    return bytes.str();
}

void KernelCache::store(const std::string& key, const std::string& bytes) {
    if (!enabled_) return;
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) return; // an unwritable cache is not an error: the code is compiled anyway
    std::random_device rd;
    std::string tmp = dir_ + "/." + key + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(rd());
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) return;
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            fs::remove(tmp, ec);
            return;
        }
    }
    fs::rename(tmp, dir_ + "/" + key, ec);
    if (ec) fs::remove(tmp, ec);
}

KernelCache::Stats KernelCache::stats() const {
    Stats s;
    s.hits = hits_;
    s.misses = misses_;
    std::error_code ec;
    if (!fs::is_directory(dir_, ec)) return s;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (!entry.is_regular_file() || !isEntry(entry.path().filename().string())) continue;
        ++s.entries;
        s.bytes += static_cast<int64_t>(entry.file_size());
    }
    return s;
}

void KernelCache::clear() {
    std::error_code ec;
    if (!fs::is_directory(dir_, ec)) return;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && (isEntry(name) || isTemporary(name))) fs::remove(entry.path(), ec);
    }
}

} // namespace dlc::backends::llvm_emit
