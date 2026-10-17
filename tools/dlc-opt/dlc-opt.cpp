#include "dlc/ir/Error.h"
#include "dlc/ir/OpRegistry.h"
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

namespace {

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [input.dlc | -]\n"
              << "Reads textual graph IR from a file, or from stdin when no file (or '-') is given.\n"
              << "Options:\n"
              << "  --passes=<p1,p2,...>   Comma-separated list of passes to run\n"
              << "                         Available: shape-infer, canonicalize, cse, dce, autodiff, fusion, mem-plan\n"
              << "  --emit-loopir          Lower optimized graph to Loop IR and print\n"
              << "  --emit-llvm            Generate and print LLVM IR (CPU)\n"
              << "  --emit-cuda            Generate and print CUDA C++ kernels (GPU)\n"
              << "  --print-mem-report     Print memory planning and arena reuse report\n"
              << "  --print-after-all      Print IR after each pass\n"
              << "  --dot                  Export Graphviz .dot representation\n"
              << "  --print-op-registry    Print the registered ops and exit\n"
              << "  --no-verify-input      Skip verifying the input (the module is still verified after each pass)\n"
              << "  --no-verify            Skip all verification\n"
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

void printOpRegistry() {
    for (const auto& def : dlc::OpRegistry::instance().ops()) {
        std::cout << def.name << " operands="
                  << (def.numOperands < 0 ? std::string("*") : std::to_string(def.numOperands))
                  << " results=" << (def.hasResult ? 1 : 0) << " class=" << dlc::opClassName(def.opClass)
                  << " attrs=";
        for (size_t i = 0; i < def.attrs.size(); ++i) {
            const auto& spec = def.attrs[i];
            std::cout << (i ? "," : "") << spec.name << ":" << dlc::attrKindName(spec.kind)
                      << (spec.required ? "!" : "");
        }
        std::cout << "\n";
    }
}

void verifyOrThrow(const dlc::Module& module, const std::string& what) {
    auto res = dlc::Verifier::verify(module);
    if (res.success) return;
    std::string message = what + " failed verification:";
    for (const auto& err : res.errors) message += "\n  " + err;
    throw dlc::IRError(message);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile;
    std::string passesArg;
    bool printAfterAll = false;
    bool exportDot = false;
    bool verifyInput = true;
    bool verifyEach = true;
    bool emitLoopIR = false;
    bool emitLLVM = false;
    bool emitCUDA = false;
    bool printMemReport = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--print-op-registry") {
            printOpRegistry();
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
        } else if (arg == "--verify") {
            // verification is on by default
        } else if (arg == "--no-verify-input") {
            verifyInput = false;
        } else if (arg == "--no-verify") {
            verifyInput = false;
            verifyEach = false;
        } else if (arg == "-" || !arg.starts_with("-")) {
            inputFile = arg;
        } else {
            std::cerr << "error: unknown option '" << arg << "' (see --help)\n";
            return 1;
        }
    }

    std::stringstream buffer;
    if (inputFile.empty() || inputFile == "-") {
        buffer << std::cin.rdbuf();
    } else {
        std::ifstream inFile(inputFile);
        if (!inFile.is_open()) {
            std::cerr << "error: could not open file " << inputFile << "\n";
            return 1;
        }
        buffer << inFile.rdbuf();
    }

    try {
        auto module = dlc::IRParser::parseModule(buffer.str());
        if (verifyInput) verifyOrThrow(*module, "input");

        dlc::PassManager pm;
        pm.setPrintAfterAll(printAfterAll);
        pm.setVerifyEach(verifyEach);
        for (const auto& p : split(passesArg, ',')) {
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
                pm.addPass(std::make_unique<dlc::FusionPass>());
            } else if (p == "mem-plan") {
                pm.addPass(std::make_unique<dlc::MemoryPlanningPass>());
            } else {
                std::cerr << "error: unknown pass '" << p << "'\n";
                return 1;
            }
        }
        pm.run(*module);

        // Lowering and the memory report plan each function on their own, without annotating the module.
        if (printMemReport) {
            for (const auto& func : module->getFunctions()) {
                dlc::MemoryPlanningPass memPlan(/*annotate=*/false);
                memPlan.runOnFunction(*func);
                std::cout << memPlan.getMemoryReport();
            }
        }

        if (emitLoopIR || emitLLVM || emitCUDA) {
            std::vector<dlc::loopir::KernelNest> allKernels;
            for (const auto& func : module->getFunctions()) {
                dlc::FusionPass fusion(/*annotate=*/false);
                dlc::MemoryPlanningPass memPlan(/*annotate=*/false);
                fusion.runOnFunction(*func);
                memPlan.runOnFunction(*func);
                for (auto& k : dlc::loopir::Lowering::lowerFunction(*func, &fusion, &memPlan)) {
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

        if (printMemReport) return 0;

        if (exportDot) {
            for (const auto& func : module->getFunctions()) {
                dlc::IRPrinter::printDot(*func, std::cout);
            }
        } else {
            std::cout << module->toString();
        }
    } catch (const dlc::IRError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
