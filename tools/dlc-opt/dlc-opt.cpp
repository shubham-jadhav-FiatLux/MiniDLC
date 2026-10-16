#include "dlc/ir/Parser.h"
#include "dlc/ir/Printer.h"
#include "dlc/ir/Verifier.h"
#include "dlc/passes/Pass.h"
#include "dlc/passes/ShapeInference.h"
#include "dlc/passes/Canonicalize.h"
#include "dlc/passes/CSE.h"
#include "dlc/passes/DCE.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/Fusion.h"
#include "dlc/passes/MemoryPlanning.h"
#include "dlc/loopir/Lowering.h"
#include "dlc/loopir/LoopPasses.h"
#include "dlc/backends/llvm/LLVMCodeGen.h"
#include "dlc/backends/cuda/CUDACodeGen.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <input.dlc>\n"
              << "Options:\n"
              << "  --passes=<p1,p2,...>   Comma-separated list of passes to run\n"
              << "                         Available: shape-infer, canonicalize, cse, dce, autodiff, fusion, mem-plan\n"
              << "  --emit-loopir          Lower optimized graph to Loop IR and print\n"
              << "  --emit-llvm            Generate and print LLVM IR (CPU)\n"
              << "  --emit-cuda            Generate and print CUDA C++ kernels (GPU)\n"
              << "  --print-mem-report     Print memory planning and arena reuse report\n"
              << "  --print-after-all      Print IR after each pass\n"
              << "  --dot                  Export Graphviz .dot representation\n"
              << "  --verify               Run verifier before and after passes (default on)\n"
              << "  --help                 Show this help message\n";
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile;
    std::string passesArg;
    bool printAfterAll = false;
    bool exportDot = false;
    bool verifyIR = true;
    bool emitLoopIR = false;
    bool emitLLVM = false;
    bool emitCUDA = false;
    bool printMemReport = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg.starts_with("--passes=")) {
            passesArg = arg.substr(9);
        } else if (arg == "--emit-loopir") {
            emitLoopIR = true;
        } else if (arg == "--emit-llvm") {
            emitLLVM = true;
        } else if (arg == "--emit-cuda") {
            emitCUDA = true;
        } else if (arg == "--print-mem-report") {
            printMemReport = true;
        } else if (arg == "--print-after-all") {
            printAfterAll = true;
        } else if (arg == "--dot") {
            exportDot = true;
        } else if (arg == "--no-verify") {
            verifyIR = false;
        } else if (!arg.starts_with("-")) {
            inputFile = arg;
        }
    }

    if (inputFile.empty()) {
        std::cerr << "Error: No input file specified.\n";
        return 1;
    }

    std::ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file " << inputFile << "\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();
    std::string source = buffer.str();

    auto module = dlc::IRParser::parseModule(source);
    if (!module) {
        std::cerr << "Error: Failed to parse IR from " << inputFile << "\n";
        return 1;
    }

    if (verifyIR) {
        auto res = dlc::Verifier::verify(*module);
        if (!res.success) {
            std::cerr << "Initial verification failed:\n";
            for (const auto& err : res.errors) {
                std::cerr << "  " << err << "\n";
            }
            return 1;
        }
    }

    dlc::PassManager pm;
    pm.setPrintAfterAll(printAfterAll);

    std::unique_ptr<dlc::FusionPass> fusionPassPtr;
    std::unique_ptr<dlc::MemoryPlanningPass> memPlanPassPtr;

    if (!passesArg.empty()) {
        auto passList = split(passesArg, ',');
        for (const auto& p : passList) {
            if (p == "shape-infer") {
                pm.addPass(std::make_unique<dlc::ShapeInferencePass>());
            } else if (p == "canonicalize") {
                pm.addPass(std::make_unique<dlc::CanonicalizePass>());
            } else if (p == "cse") {
                pm.addPass(std::make_unique<dlc::CSEPass>());
            } else if (p == "dce") {
                pm.addPass(std::make_unique<dlc::DCEPass>());
            } else if (p == "autodiff") {
                pm.addPass(std::make_unique<dlc::AutodiffPass>());
            } else if (p == "fusion") {
                auto pass = std::make_unique<dlc::FusionPass>();
                fusionPassPtr = std::make_unique<dlc::FusionPass>();
                pm.addPass(std::move(pass));
            } else if (p == "mem-plan") {
                auto pass = std::make_unique<dlc::MemoryPlanningPass>();
                memPlanPassPtr = std::make_unique<dlc::MemoryPlanningPass>();
                pm.addPass(std::move(pass));
            } else {
                std::cerr << "Warning: Unknown pass '" << p << "' ignored.\n";
            }
        }
    }

    pm.run(*module);

    if (verifyIR) {
        auto res = dlc::Verifier::verify(*module);
        if (!res.success) {
            std::cerr << "Post-optimization verification failed:\n";
            for (const auto& err : res.errors) {
                std::cerr << "  " << err << "\n";
            }
            return 1;
        }
    }

    // Run standalone fusion / mem-plan for lowering if requested
    dlc::FusionPass standaloneFusion;
    dlc::MemoryPlanningPass standaloneMemPlan;
    for (const auto& func : module->getFunctions()) {
        standaloneFusion.runOnFunction(*func);
        standaloneMemPlan.runOnFunction(*func);
    }

    if (printMemReport) {
        std::cout << standaloneMemPlan.getMemoryReport();
    }

    if (emitLoopIR || emitLLVM || emitCUDA) {
        std::vector<dlc::loopir::KernelNest> allKernels;
        for (const auto& func : module->getFunctions()) {
            auto kernels = dlc::loopir::Lowering::lowerFunction(*func, &standaloneFusion, &standaloneMemPlan);
            for (auto& k : kernels) {
                dlc::loopir::LoopTilingPass::run(k);
                dlc::loopir::LoopVectorizationPass::run(k, 8);
                dlc::loopir::LoopParallelizationPass::run(k);
                dlc::loopir::BoundsCheckEliminationPass::run(k);
                allKernels.push_back(std::move(k));
            }
        }

        if (emitCUDA) {
            std::cout << dlc::backends::cuda::CUDACodeGen::generateModuleCode(allKernels);
        } else if (emitLLVM) {
            std::cout << dlc::backends::llvm::LLVMCodeGen::generateModuleIR(allKernels);
        } else {
            for (const auto& k : allKernels) {
                std::cout << k.toString();
            }
        }
        return 0;
    }

    if (exportDot) {
        for (const auto& func : module->getFunctions()) {
            dlc::IRPrinter::printDot(*func, std::cout);
        }
    } else {
        std::cout << module->toString();
    }

    return 0;
}
