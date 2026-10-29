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
#include "dlc/passes/Activity.h"
#include "dlc/passes/Autodiff.h"
#include "dlc/passes/Optimizer.h"
#include "dlc/backends/llvm/CpuJit.h"
#include "dlc/compiler/Compiler.h"
#include "dlc/passes/BufferAssignment.h"
#include "dlc/passes/CostModel.h"
#include "dlc/passes/Fusion.h"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [input.dlc | -]\n"
              << "Reads textual graph IR from a file, or from stdin when no file (or '-') is given.\n"
              << "Options:\n"
              << "  --passes=<p1,p2,...>   Comma-separated list of passes to run\n"
              << "                         Available: shape-infer, canonicalize, cse, dce, activity, autodiff,\n"
              << "                         sgd{lr=,momentum=}, adam{lr=,beta1=,beta2=,eps=}, fusion, buffer-assign\n"
              << "  --emit-loopir          Fuse, assign buffers and lower to loop IR with the CPU schedules; print it\n"
              << "  --emit-llvm            ... then emit the LLVM IR for the host CPU (before optimization)\n"
              << "  --emit-llvm-opt        ... after LLVM's O3 pipeline\n"
              << "  --print-loop-report    Per loop kernel: accesses proven in bounds, guarded (indexed by data), guards\n"
              << "                         eliminated; then the parallel extent\n"
              << "  --print-cost           Print the kernels with their FLOPs, bytes and arithmetic intensity\n"
              << "  --print-mem-report     Print the buffer assignment: regions, offsets, live ranges, arena reuse\n"
              << "  --print-after-all      Print IR after each pass\n"
              << "  --dot                  Export Graphviz .dot representation\n"
              << "  --print-op-registry    Print the registered ops and exit\n"
              << "  --no-verify-input      Skip verifying the input (the module is still verified after each pass)\n"
              << "  --no-verify            Skip all verification\n"
              << "  --help                 Show this help message\n";
}

// Splits at the delimiter, except inside braces (pass options such as adam{lr=0.001,beta1=0.9}).
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    int depth = 0;
    for (char c : s) {
        if (c == '{') ++depth;
        if (c == '}') --depth;
        if (c == delimiter && depth == 0) {
            if (!token.empty()) tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

// "adam{lr=0.001}" → name "adam", options {lr: 0.001}.
struct PassSpec {
    std::string name;
    std::map<std::string, double> options;
};

PassSpec parsePassSpec(const std::string& text) {
    PassSpec spec;
    size_t brace = text.find('{');
    spec.name = text.substr(0, brace);
    if (brace == std::string::npos) return spec;
    if (text.back() != '}') throw dlc::IRError("malformed pass options in '" + text + "'");
    for (const auto& item : split(text.substr(brace + 1, text.size() - brace - 2), ',')) {
        size_t eq = item.find('=');
        if (eq == std::string::npos) throw dlc::IRError("pass option '" + item + "' needs a value");
        try {
            spec.options[item.substr(0, eq)] = std::stod(item.substr(eq + 1));
        } catch (const std::exception&) {
            throw dlc::IRError("pass option '" + item + "' needs a number");
        }
    }
    return spec;
}

dlc::OptimizerConfig optimizerConfig(const PassSpec& spec) {
    dlc::OptimizerConfig config;
    config.kind = spec.name == "adam" ? dlc::OptimizerConfig::Kind::Adam : dlc::OptimizerConfig::Kind::SGD;
    for (const auto& [key, value] : spec.options) {
        if (key == "lr") config.lr = value;
        else if (key == "momentum" && spec.name == "sgd") config.momentum = value;
        else if (key == "beta1" && spec.name == "adam") config.beta1 = value;
        else if (key == "beta2" && spec.name == "adam") config.beta2 = value;
        else if (key == "eps" && spec.name == "adam") config.eps = value;
        else throw dlc::IRError("unknown option '" + key + "' for pass '" + spec.name + "'");
    }
    return config;
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
    bool printCost = false;
    bool printMemReport = false;
    std::string emit; // "loopir", "llvm" or "llvm-opt"
    bool printLoopReport = false;

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
        } else if (arg == "--emit-loopir" || arg == "--emit-llvm" || arg == "--emit-llvm-opt") {
            emit = arg.substr(7);
        } else if (arg == "--print-loop-report") {
            printLoopReport = true;
        } else if (arg == "--print-cost") {
            printCost = true;
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
        for (const auto& text : split(passesArg, ',')) {
            PassSpec spec = parsePassSpec(text);
            const std::string& p = spec.name;
            if (!spec.options.empty() && p != "adam" && p != "sgd") {
                throw dlc::IRError("pass '" + p + "' takes no options");
            }
            if (p == "shape-infer") {
                pm.addPass(std::make_unique<dlc::ShapeInferencePass>());
            } else if (p == "canonicalize") {
                pm.addPass(std::make_unique<dlc::CanonicalizePass>());
            } else if (p == "cse") {
                pm.addPass(std::make_unique<dlc::CSEPass>());
            } else if (p == "dce") {
                pm.addPass(std::make_unique<dlc::DCEPass>());
            } else if (p == "activity") {
                pm.addPass(std::make_unique<dlc::ActivityPass>());
            } else if (p == "autodiff") {
                pm.addPass(std::make_unique<dlc::AutodiffPass>());
            } else if (p == "adam" || p == "sgd") {
                pm.addPass(std::make_unique<dlc::OptimizerPass>(optimizerConfig(spec)));
            } else if (p == "fusion") {
                pm.addPass(std::make_unique<dlc::FusionPass>());
            } else if (p == "buffer-assign") {
                pm.addPass(std::make_unique<dlc::BufferAssignmentPass>());
            } else {
                std::cerr << "error: unknown pass '" << p << "'\n";
                return 1;
            }
        }
        pm.run(*module);

        // The reports group and plan each function on their own, without annotating the module.
        for (const auto& func : module->getFunctions()) {
            if (printCost) {
                dlc::FusionPass fusion(/*annotate=*/false);
                fusion.runOnFunction(*func);
                std::cout << dlc::costReport(*func, fusion.getKernels());
            }
            if (printMemReport) {
                dlc::BufferAssignmentPass assign(/*annotate=*/false);
                assign.runOnFunction(*func);
                std::cout << assign.getReport();
            }
        }
        if (printCost || printMemReport) return 0;

        if (!emit.empty() || printLoopReport) {
            std::vector<dlc::loopir::Kernel> all;
            for (const auto& func : module->getFunctions()) {
                for (auto& l : dlc::compiler::lowerForInspection(*func, dlc::loopir::Target::CPU)) {
                    if (printLoopReport) {
                        std::cout << l.kernel.name << ": " << l.bounds.accesses << " accesses, " << l.bounds.proven
                                  << " proven in bounds, " << l.bounds.guarded << " guarded, " << l.bounds.eliminated
                                  << " guards eliminated; parallel extent " << l.kernel.parallelExtent() << "\n";
                    }
                    all.push_back(std::move(l.kernel));
                }
            }
            if (emit == "loopir") {
                for (const auto& k : all) std::cout << dlc::loopir::toString(k);
            } else if (!emit.empty()) {
                std::cout << dlc::backends::llvm_emit::cpuModuleText(all, emit == "llvm-opt");
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
    } catch (const dlc::IRError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
