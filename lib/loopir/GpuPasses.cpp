#include "dlc/loopir/GpuPasses.h"
#include "dlc/ir/Error.h"
#include "dlc/loopir/Interpreter.h"
#include "dlc/loopir/Transforms.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>

namespace dlc::loopir {

namespace {

constexpr int kBanks = 32;
constexpr int64_t kSectorBytes = 32;

// n-way conflict of one shared-memory request: the most distinct words that fall in one bank.
int bankWays(const std::vector<int64_t>& words) {
    std::map<int64_t, std::set<int64_t>> perBank;
    for (int64_t w : words) perBank[((w % kBanks) + kBanks) % kBanks].insert(w);
    int ways = 1;
    for (const auto& [bank, distinct] : perBank) ways = std::max(ways, static_cast<int>(distinct.size()));
    return ways;
}

Stmt* findLet(Body& stmts, const std::string& name) {
    for (auto& s : stmts) {
        if (s->kind == StmtKind::Let && s->name == name) return s.get();
        if (Stmt* inner = findLet(s->body, name)) return inner;
        if (Stmt* inner = findLet(s->elseBody, name)) return inner;
    }
    return nullptr;
}

std::string siteIndex(const WarpRequest& r, const Kernel& k) {
    if (r.store) return toString(static_cast<const Stmt*>(r.site)->index, k);
    return toString(static_cast<const Expr*>(r.site)->args[0], k);
}

} // namespace

GpuResources checkResources(const Kernel& k, const GpuLimits& limits) {
    if (!k.gpu) throw IRError("resource check: " + k.name + " is not a GPU kernel");
    GpuResources r;
    r.threads = k.block[0] * k.block[1] * k.block[2];
    for (const BufferDecl& b : k.buffers) {
        if (b.scope == MemScope::Shared) r.sharedBytes += b.elements() * 4;
        if (b.scope == MemScope::Register) r.registerWords += b.elements();
    }
    auto fail = [&](const std::string& what) { throw IRError("resource check: " + k.name + ": " + what); };
    static const char* axes = "xyz";
    for (int d = 0; d < 3; ++d) {
        if (k.block[d] < 1 || k.block[d] > limits.maxBlock[d]) {
            fail(std::string("block.") + axes[d] + " = " + std::to_string(k.block[d]) + " is outside [1, " +
                 std::to_string(limits.maxBlock[d]) + "]");
        }
        if (k.grid[d] < 1 || k.grid[d] > limits.maxGrid[d]) {
            fail(std::string("grid.") + axes[d] + " = " + std::to_string(k.grid[d]) + " is outside [1, " +
                 std::to_string(limits.maxGrid[d]) + "]");
        }
    }
    if (r.threads > limits.maxThreadsPerBlock) {
        fail(std::to_string(r.threads) + " threads per block, more than " + std::to_string(limits.maxThreadsPerBlock));
    }
    if (r.sharedBytes > limits.maxSharedBytes) {
        fail(std::to_string(r.sharedBytes) + " bytes of shared memory per block, more than " + std::to_string(limits.maxSharedBytes));
    }
    if (r.registerWords > limits.maxRegistersPerThread) {
        fail(std::to_string(r.registerWords) + " words of register buffers per thread, more than the " +
             std::to_string(limits.maxRegistersPerThread) + " registers a thread can have");
    }
    if (r.registerWords * r.threads > limits.registersPerBlock) {
        fail(std::to_string(r.registerWords) + " register words for each of " + std::to_string(r.threads) +
             " threads exceed the " + std::to_string(limits.registersPerBlock) + " registers of a block");
    }
    return r;
}

std::vector<AccessStats> analyzeAccesses(const Kernel& k) {
    std::vector<WarpRequest> requests = probeWarpAccesses(k);
    std::map<const void*, AccessStats> bySite;
    std::vector<const void*> order;
    std::map<const void*, double> waysSum, sectorSum, efficiencySum;
    for (const WarpRequest& r : requests) {
        auto [it, inserted] = bySite.try_emplace(r.site);
        AccessStats& s = it->second;
        if (inserted) {
            s.buffer = r.buffer;
            s.store = r.store;
            s.index = siteIndex(r, k);
            order.push_back(r.site);
        }
        ++s.requests;
        if (k.buffers[r.buffer].scope == MemScope::Shared) {
            int ways = bankWays(r.indices);
            s.maxWays = std::max(s.maxWays, ways);
            waysSum[r.site] += ways;
        } else {
            std::set<int64_t> sectors, words(r.indices.begin(), r.indices.end());
            for (int64_t i : r.indices) sectors.insert(i * 4 / kSectorBytes);
            sectorSum[r.site] += static_cast<double>(sectors.size());
            efficiencySum[r.site] += static_cast<double>(words.size() * 4) / static_cast<double>(sectors.size() * kSectorBytes);
        }
    }
    std::vector<AccessStats> out;
    for (const void* site : order) {
        AccessStats s = bySite[site];
        if (k.buffers[s.buffer].scope == MemScope::Shared) {
            s.meanWays = waysSum[site] / static_cast<double>(s.requests);
        } else {
            s.sectors = sectorSum[site] / static_cast<double>(s.requests);
            s.efficiency = efficiencySum[site] / static_cast<double>(s.requests);
        }
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<PaddingDecision> padSharedBuffers(Kernel& k, const GpuLimits& limits) {
    std::vector<PaddingDecision> out;
    for (size_t b = 0; b < k.buffers.size(); ++b) {
        if (k.buffers[b].scope != MemScope::Shared || k.buffers[b].shape.size() != 2) continue;
        const std::string name = k.buffers[b].name;
        const std::string pitchName = name + "_pitch";
        Stmt* pitchLet = findLet(k.body, pitchName);
        if (!pitchLet) continue;
        const int64_t rows = k.buffers[b].shape[0], cols = k.buffers[b].shape[1];
        int64_t otherShared = 0;
        for (size_t o = 0; o < k.buffers.size(); ++o) {
            if (o != b && k.buffers[o].scope == MemScope::Shared) otherShared += k.buffers[o].elements() * 4;
        }
        // The worst conflict over every request to the buffer, with pitch p.
        auto waysAt = [&](int64_t p) {
            pitchLet->value = intConst(p);
            k.buffers[b].shape = {rows, p};
            int worst = 1;
            for (const WarpRequest& r : probeWarpAccesses(k)) {
                if (r.buffer == static_cast<int>(b)) worst = std::max(worst, bankWays(r.indices));
            }
            return worst;
        };
        PaddingDecision d{name, cols, cols, waysAt(cols), 0};
        d.waysAfter = d.waysBefore;
        // Pitches modulo 32 words cover every bank mapping; stop at the first conflict-free one.
        for (int64_t p = cols + 1; p < cols + kBanks && d.waysAfter > 1; ++p) {
            if (otherShared + rows * p * 4 > limits.maxSharedBytes) break;
            int ways = waysAt(p);
            if (ways < d.waysAfter) {
                d.to = p;
                d.waysAfter = ways;
            }
        }
        k.buffers[b].shape = {rows, d.to};
        removeLets(k.body, pitchName);
        substitute(k.body, pitchName, intConst(d.to));
        out.push_back(d);
    }
    return out;
}

std::string toString(const GpuReport& report, const Kernel& k) {
    std::ostringstream os;
    const GpuResources& r = report.resources;
    os << "    resources: " << r.threads << " threads, " << r.sharedBytes << " bytes of shared memory, " << r.registerWords
       << " register words per thread\n";
    for (const PaddingDecision& d : report.padding) {
        os << "    padding: " << d.buffer << " rows " << d.from << " -> " << d.to << " words (bank conflicts " << d.waysBefore
           << "-way -> " << d.waysAfter << "-way)\n";
    }
    char line[512];
    for (const AccessStats& a : report.accesses) {
        const BufferDecl& b = k.buffers[a.buffer];
        if (b.scope == MemScope::Shared) {
            std::snprintf(line, sizeof line, "    shared %-5s %-10s %-48s %5lld requests, %d-way worst, %.2f-way mean\n",
                          a.store ? "store" : "load", b.name.c_str(), a.index.c_str(), static_cast<long long>(a.requests),
                          a.maxWays, a.meanWays);
        } else {
            std::snprintf(line, sizeof line, "    global %-5s %-10s %-48s %5lld requests, %.1f sectors, %3.0f%% of fetched bytes used\n",
                          a.store ? "store" : "load", b.name.c_str(), a.index.c_str(), static_cast<long long>(a.requests),
                          a.sectors, 100.0 * a.efficiency);
        }
        os << line;
    }
    return os.str();
}

} // namespace dlc::loopir
