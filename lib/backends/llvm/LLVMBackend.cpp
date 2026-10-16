#include "dlc/backends/llvm/LLVMBackend.h"
#include "dlc/backends/llvm/LLVMCodeGen.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

namespace dlc::backends::llvm {

using KernelFn = void (*)(void* arena, int64_t start, int64_t end);

LLVMBackend::LLVMBackend(size_t threads) : threadPool_(threads) {}

LLVMBackend::~LLVMBackend() {
    for (void* handle : dlHandles_) {
        if (handle) dlclose(handle);
    }
    for (const auto& f : tempSoFiles_) {
        unlink(f.c_str());
    }
}

runtime::ExecutionPlan LLVMBackend::compile(const std::vector<loopir::KernelNest>& kernels) {
    runtime::ExecutionPlan plan;
    if (kernels.empty()) return plan;

    // 1. Emit LLVM IR Module
    lastGeneratedIR_ = LLVMCodeGen::generateModuleIR(kernels);

    // 2. Write to temp .ll file
    char llPattern[] = "/tmp/dlc_llvm_XXXXXX.ll";
    int fd = mkstemps(llPattern, 3);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary LLVM IR file");
    }
    std::string llPath = llPattern;
    close(fd);

    std::ofstream llFile(llPath);
    llFile << lastGeneratedIR_;
    llFile.close();

    std::string soPath = llPath.substr(0, llPath.length() - 3) + ".so";
    tempSoFiles_.push_back(llPath);
    tempSoFiles_.push_back(soPath);

    // 3. Compile with Clang -O3 -mavx2 -mfma -shared -fPIC
    std::string cmd = "clang -O3 -shared -mavx2 -mfma -fPIC " + llPath + " -o " + soPath + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to invoke clang compiler");
    }

    char buffer[256];
    std::string compilerOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        compilerOutput += buffer;
    }
    int ret = pclose(pipe);
    if (ret != 0) {
        std::cerr << "LLVM Compilation failed:\n" << compilerOutput << "\n";
        throw std::runtime_error("Clang LLVM JIT compilation failed");
    }

    // 4. Load compiled shared library
    void* handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        throw std::runtime_error(std::string("dlopen failed: ") + (err ? err : "unknown"));
    }
    dlHandles_.push_back(handle);

    // 5. Create kernel dispatches
    for (const auto& k : kernels) {
        void* sym = dlsym(handle, k.getName().c_str());
        if (!sym) {
            std::cerr << "Warning: Symbol @" << k.getName() << " not found in compiled module\n";
            continue;
        }

        KernelFn fn = reinterpret_cast<KernelFn>(sym);

        // Determine loop bounds and parallelization
        int64_t loopStart = 0;
        int64_t loopEnd = 1;
        bool isParallel = false;

        if (!k.getRootStmts().empty()) {
            auto forLoop = std::dynamic_pointer_cast<loopir::ForLoop>(k.getRootStmts()[0]);
            if (forLoop) {
                loopStart = forLoop->getStart();
                loopEnd = forLoop->getEnd();
                isParallel = forLoop->isParallel();
            }
        }

        runtime::KernelDispatch kd;
        kd.name = k.getName();
        kd.totalFlops = k.getEstimatedFlops();
        kd.bytesMoved = k.getEstimatedBytesMoved();

        kd.launchFn = [fn, loopStart, loopEnd, isParallel, this](runtime::ArenaAllocator& arena) {
            void* rawArena = arena.getRawPointer();
            if (isParallel && (loopEnd - loopStart) > 1) {
                int64_t total = loopEnd - loopStart;
                int64_t numThreads = 4;
                int64_t chunkSize = (total + numThreads - 1) / numThreads;
                std::vector<std::future<void>> futures;

                for (int64_t t = 0; t < numThreads; ++t) {
                    int64_t s = loopStart + t * chunkSize;
                    int64_t e = std::min(loopEnd, s + chunkSize);
                    if (s < e) {
                        futures.push_back(threadPool_.enqueue([fn, rawArena, s, e]() {
                            fn(rawArena, s, e);
                        }));
                    }
                }
                for (auto& fut : futures) {
                    fut.get();
                }
            } else {
                fn(rawArena, loopStart, loopEnd);
            }
        };

        plan.addKernel(std::move(kd));
    }

    return plan;
}

} // namespace dlc::backends::llvm

