#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

namespace dlc::backends::llvm_emit {

// The kernel cache: compiled code on disk, keyed by a SHA-256 of everything that determines it — the loop IR of the
// kernels (schedules, shapes, buffer layout, launch configuration), the target (triple, CPU, features or GPU
// architecture), the LLVM version and the build of the code generator. A hit skips LLVM's O3 pipeline and code
// generation: CPU executables load the cached object file into the JIT; GPU artifacts take the cached PTX.
//
// Location: $MINIDLC_CACHE_DIR, else $XDG_CACHE_HOME/minidlc, else ~/.cache/minidlc. MINIDLC_CACHE=off disables it.
// Entries are written to a temporary file and renamed, so concurrent compilers never read half an entry. Nothing is
// evicted: delete the directory (or call clear()) to reclaim the space.
class KernelCache {
public:
    static KernelCache& instance();

    bool enabled() const { return enabled_; }
    const std::string& directory() const { return dir_; }

    // The key of `content` (the code generator's build and LLVM's version are mixed in).
    std::string key(const std::string& kind, const std::string& content) const;
    std::optional<std::string> load(const std::string& key);
    void store(const std::string& key, const std::string& bytes);

    struct Stats {
        int64_t hits{0};
        int64_t misses{0};
        int64_t entries{0};
        int64_t bytes{0};
    };
    Stats stats() const; // hits and misses of this process; entries and bytes on disk
    void clear();        // deletes every entry

    // Re-reads the environment (for tests).
    void configure();

private:
    KernelCache();

    bool enabled_{true};
    std::string dir_;
    std::atomic<int64_t> hits_{0};
    std::atomic<int64_t> misses_{0};
};

} // namespace dlc::backends::llvm_emit
