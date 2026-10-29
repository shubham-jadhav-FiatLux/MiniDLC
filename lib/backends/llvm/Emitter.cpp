#include "dlc/backends/llvm/Emitter.h"
#include "dlc/ir/Error.h"
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <unordered_map>

namespace dlc::backends::llvm_emit {

using namespace dlc::loopir;

namespace {

// A name in scope: a stack slot (loop counters, Let locals) or an SSA value (the constant of an unrolled iteration).
struct Binding {
    llvm::Value* slot{nullptr};
    llvm::Value* value{nullptr};
    llvm::Type* type{nullptr};
};

class KernelEmitter {
public:
    KernelEmitter(const Kernel& kernel, llvm::Module& module)
        : k_(kernel), module_(module), ctx_(module.getContext()), b_(ctx_) {}

    void emit() {
        llvm::Type* ptr = llvm::PointerType::getUnqual(ctx_);
        llvm::Type* i64 = llvm::Type::getInt64Ty(ctx_);
        std::vector<int> globals = k_.globalBuffers();

        // The body: buffers as noalias pointers.
        std::vector<llvm::Type*> params(globals.size(), ptr);
        params.push_back(i64);
        params.push_back(i64);
        auto* bodyType = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), params, false);
        auto* body = llvm::Function::Create(bodyType, llvm::Function::InternalLinkage, k_.name + ".body", module_);
        body->addFnAttr(llvm::Attribute::AlwaysInline);
        body->addFnAttr(llvm::Attribute::NoUnwind);
        for (size_t i = 0; i < globals.size(); ++i) body->addParamAttr(static_cast<unsigned>(i), llvm::Attribute::NoAlias);
        func_ = body;
        auto* entry = llvm::BasicBlock::Create(ctx_, "entry", body);
        allocaBlock_ = entry;
        b_.SetInsertPoint(entry);

        buffers_.assign(k_.buffers.size(), nullptr);
        for (size_t i = 0; i < globals.size(); ++i) buffers_[globals[i]] = body->getArg(static_cast<unsigned>(i));
        for (size_t i = 0; i < k_.buffers.size(); ++i) {
            if (k_.buffers[i].scope == MemScope::Global) continue;
            auto* arrayType = llvm::ArrayType::get(elemType(k_.buffers[i].type), k_.buffers[i].elements());
            buffers_[i] = b_.CreateAlloca(arrayType, nullptr, k_.buffers[i].name);
        }
        begin_ = body->getArg(static_cast<unsigned>(globals.size()));
        end_ = body->getArg(static_cast<unsigned>(globals.size() + 1));

        scopes_.emplace_back();
        bool parallel = k_.parallelExtent() > 1 ||
                        (k_.body.size() == 1 && k_.body[0]->kind == StmtKind::For && k_.body[0]->loop == LoopKind::Parallel);
        if (parallel) loop(*k_.body[0], true);
        else emitBody(k_.body);
        scopes_.pop_back();
        b_.CreateRetVoid();

        // The entry point: unpack the argument array.
        auto* entryType = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {ptr, i64, i64}, false);
        auto* entryFn = llvm::Function::Create(entryType, llvm::Function::ExternalLinkage, k_.name, module_);
        entryFn->addFnAttr(llvm::Attribute::NoUnwind);
        auto* bb = llvm::BasicBlock::Create(ctx_, "entry", entryFn);
        llvm::IRBuilder<> eb(bb);
        std::vector<llvm::Value*> callArgs;
        for (size_t i = 0; i < globals.size(); ++i) {
            llvm::Value* addr = eb.CreateConstInBoundsGEP1_64(ptr, entryFn->getArg(0), i);
            callArgs.push_back(eb.CreateLoad(ptr, addr, k_.buffers[globals[i]].name));
        }
        callArgs.push_back(entryFn->getArg(1));
        callArgs.push_back(entryFn->getArg(2));
        eb.CreateCall(body, callArgs);
        eb.CreateRetVoid();
    }

private:
    llvm::Type* elemType(ScalarType t) {
        switch (t) {
            case ScalarType::F32: return llvm::Type::getFloatTy(ctx_);
            case ScalarType::I32: return llvm::Type::getInt32Ty(ctx_);
            case ScalarType::I64: return llvm::Type::getInt64Ty(ctx_);
            case ScalarType::Bool: return llvm::Type::getInt1Ty(ctx_);
        }
        return nullptr;
    }

    llvm::AllocaInst* createAlloca(llvm::Type* type, const std::string& name) {
        llvm::IRBuilder<> ab(allocaBlock_, allocaBlock_->begin());
        return ab.CreateAlloca(type, nullptr, name);
    }

    Binding& lookup(const std::string& name) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        throw IRError("LLVM emitter: undefined variable '" + name + "' in " + k_.name);
    }

    void emitBody(const Body& stmts) {
        for (const auto& s : stmts) stmt(*s);
    }

    llvm::Value* address(int buffer, llvm::Value* index) {
        const BufferDecl& decl = k_.buffers[buffer];
        llvm::Type* elem = elemType(decl.type);
        if (decl.scope == MemScope::Global) return b_.CreateInBoundsGEP(elem, buffers_[buffer], index);
        auto* arrayType = llvm::ArrayType::get(elem, decl.elements());
        return b_.CreateInBoundsGEP(arrayType, buffers_[buffer], {b_.getInt64(0), index});
    }

    void loop(const Stmt& s, bool topLevelParallel = false) {
        if (s.loop == LoopKind::Unrolled) {
            for (int64_t i = 0; i < s.extent; ++i) {
                scopes_.emplace_back();
                scopes_.back()[s.var] = {nullptr, b_.getInt64(i), b_.getInt64Ty()};
                emitBody(s.body);
                scopes_.pop_back();
            }
            return;
        }
        llvm::Type* i64 = b_.getInt64Ty();
        llvm::Value* lo = topLevelParallel ? begin_ : b_.getInt64(0);
        llvm::Value* hi = b_.getInt64(s.extent);
        if (topLevelParallel) hi = b_.CreateSelect(b_.CreateICmpSLT(end_, hi), end_, hi);
        llvm::AllocaInst* iv = createAlloca(i64, s.var);
        b_.CreateStore(lo, iv);
        auto* header = llvm::BasicBlock::Create(ctx_, s.var + ".header", func_);
        auto* bodyBlock = llvm::BasicBlock::Create(ctx_, s.var + ".body", func_);
        auto* latch = llvm::BasicBlock::Create(ctx_, s.var + ".latch", func_);
        auto* exit = llvm::BasicBlock::Create(ctx_, s.var + ".exit", func_);
        b_.CreateBr(header);
        b_.SetInsertPoint(header);
        llvm::Value* v = b_.CreateLoad(i64, iv);
        b_.CreateCondBr(b_.CreateICmpSLT(v, hi), bodyBlock, exit);
        b_.SetInsertPoint(bodyBlock);
        scopes_.emplace_back();
        scopes_.back()[s.var] = {iv, nullptr, i64};
        bool vector = s.loop == LoopKind::Vectorized;
        vectorDepth_ += vector ? 1 : 0;
        emitBody(s.body);
        vectorDepth_ -= vector ? 1 : 0;
        scopes_.pop_back();
        b_.CreateBr(latch);
        b_.SetInsertPoint(latch);
        llvm::Value* next = b_.CreateAdd(b_.CreateLoad(i64, iv), b_.getInt64(1), "", false, true);
        b_.CreateStore(next, iv);
        llvm::BranchInst* back = b_.CreateBr(header);
        if (vector) back->setMetadata(llvm::LLVMContext::MD_loop, vectorizeMetadata(s.width));
        b_.SetInsertPoint(exit);
    }

    llvm::MDNode* vectorizeMetadata(int width) {
        llvm::Metadata* enable[] = {llvm::MDString::get(ctx_, "llvm.loop.vectorize.enable"),
                                    llvm::ConstantAsMetadata::get(b_.getTrue())};
        llvm::Metadata* w[] = {llvm::MDString::get(ctx_, "llvm.loop.vectorize.width"),
                               llvm::ConstantAsMetadata::get(b_.getInt32(width))};
        llvm::Metadata* ops[] = {nullptr, llvm::MDNode::get(ctx_, enable), llvm::MDNode::get(ctx_, w)};
        llvm::MDNode* node = llvm::MDNode::getDistinct(ctx_, ops);
        node->replaceOperandWith(0, node);
        return node;
    }

    // A reduction update in a vectorized loop (an accumulator, or a register buffer) may be reassociated, which is
    // what lets LLVM vectorize it; everything else keeps IEEE semantics.
    struct Reassociate {
        Reassociate(llvm::IRBuilder<>& b, bool enable) : b_(b), saved_(b.getFastMathFlags()) {
            if (!enable) return;
            llvm::FastMathFlags fmf = saved_;
            fmf.setAllowReassoc();
            fmf.setAllowContract();
            b_.setFastMathFlags(fmf);
        }
        ~Reassociate() { b_.setFastMathFlags(saved_); }
        llvm::IRBuilder<>& b_;
        llvm::FastMathFlags saved_;
    };

    void stmt(const Stmt& s) {
        bool reduction = vectorDepth_ > 0 && (s.kind == StmtKind::Assign ||
                                              (s.kind == StmtKind::Store && k_.buffers[s.buffer].scope != MemScope::Global));
        Reassociate scope(b_, reduction);
        switch (s.kind) {
            case StmtKind::For: loop(s); break;
            case StmtKind::Let: {
                llvm::Value* value = expr(*s.value);
                llvm::AllocaInst* slot = createAlloca(elemType(s.type), s.name);
                b_.CreateStore(value, slot);
                scopes_.back()[s.name] = {slot, nullptr, elemType(s.type)};
                break;
            }
            case StmtKind::Assign: {
                llvm::Value* value = expr(*s.value);
                Binding& binding = lookup(s.name);
                if (!binding.slot) throw IRError("LLVM emitter: assignment to a loop variable");
                b_.CreateStore(value, binding.slot);
                break;
            }
            case StmtKind::Store:
                b_.CreateAlignedStore(expr(*s.value), address(s.buffer, expr(*s.index)), llvm::MaybeAlign(4));
                break;
            case StmtKind::AtomicAdd:
                b_.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, address(s.buffer, expr(*s.index)), expr(*s.value),
                                   llvm::MaybeAlign(4), llvm::AtomicOrdering::Monotonic);
                break;
            case StmtKind::If: {
                auto* thenBlock = llvm::BasicBlock::Create(ctx_, "then", func_);
                auto* elseBlock = llvm::BasicBlock::Create(ctx_, "else", func_);
                auto* merge = llvm::BasicBlock::Create(ctx_, "endif", func_);
                b_.CreateCondBr(expr(*s.cond), thenBlock, elseBlock);
                for (auto [block, stmts] : {std::pair{thenBlock, &s.body}, std::pair{elseBlock, &s.elseBody}}) {
                    b_.SetInsertPoint(block);
                    scopes_.emplace_back();
                    emitBody(*stmts);
                    scopes_.pop_back();
                    b_.CreateBr(merge);
                }
                b_.SetInsertPoint(merge);
                break;
            }
            case StmtKind::Barrier: break; // one CPU thread runs a whole iteration space
        }
    }

    llvm::Value* intrinsic(llvm::Intrinsic::ID id, std::vector<llvm::Value*> args) {
        return b_.CreateIntrinsic(id, {args[0]->getType()}, args);
    }

    llvm::Value* expr(const Expr& e) {
        switch (e.kind) {
            case ExprKind::Int: return b_.getInt64(e.intValue);
            case ExprKind::Float: return llvm::ConstantFP::get(b_.getFloatTy(), e.floatValue);
            case ExprKind::Var: {
                Binding& binding = lookup(e.name);
                return binding.value ? binding.value : b_.CreateLoad(binding.type, binding.slot, e.name);
            }
            case ExprKind::Load: {
                llvm::Type* type = elemType(k_.buffers[e.buffer].type);
                return b_.CreateAlignedLoad(type, address(e.buffer, expr(*e.args[0])), llvm::MaybeAlign(4));
            }
            case ExprKind::Binary: return binary(e);
            case ExprKind::Unary: {
                llvm::Value* a = expr(*e.args[0]);
                switch (e.unary) {
                    case UnaryOp::Neg: return e.type == ScalarType::F32 ? b_.CreateFNeg(a) : b_.CreateNeg(a);
                    case UnaryOp::Exp: return intrinsic(llvm::Intrinsic::exp, {a});
                    case UnaryOp::Log: return intrinsic(llvm::Intrinsic::log, {a});
                    case UnaryOp::Tanh: return intrinsic(llvm::Intrinsic::tanh, {a});
                    case UnaryOp::Sqrt: return intrinsic(llvm::Intrinsic::sqrt, {a});
                    case UnaryOp::Rsqrt:
                        return b_.CreateFDiv(llvm::ConstantFP::get(b_.getFloatTy(), 1.0), intrinsic(llvm::Intrinsic::sqrt, {a}));
                }
                break;
            }
            case ExprKind::Select: return b_.CreateSelect(expr(*e.args[0]), expr(*e.args[1]), expr(*e.args[2]));
            case ExprKind::Compare: {
                llvm::Value* a = expr(*e.args[0]);
                llvm::Value* c = expr(*e.args[1]);
                if (e.args[0]->type == ScalarType::F32) {
                    switch (e.compare) {
                        case CompareOp::LT: return b_.CreateFCmpOLT(a, c);
                        case CompareOp::LE: return b_.CreateFCmpOLE(a, c);
                        case CompareOp::GT: return b_.CreateFCmpOGT(a, c);
                        case CompareOp::GE: return b_.CreateFCmpOGE(a, c);
                        case CompareOp::EQ: return b_.CreateFCmpOEQ(a, c);
                        case CompareOp::NE: return b_.CreateFCmpUNE(a, c);
                    }
                }
                switch (e.compare) {
                    case CompareOp::LT: return b_.CreateICmpSLT(a, c);
                    case CompareOp::LE: return b_.CreateICmpSLE(a, c);
                    case CompareOp::GT: return b_.CreateICmpSGT(a, c);
                    case CompareOp::GE: return b_.CreateICmpSGE(a, c);
                    case CompareOp::EQ: return b_.CreateICmpEQ(a, c);
                    case CompareOp::NE: return b_.CreateICmpNE(a, c);
                }
                break;
            }
            case ExprKind::Cast: {
                llvm::Value* a = expr(*e.args[0]);
                ScalarType from = e.args[0]->type;
                llvm::Type* to = elemType(e.type);
                if (e.type == ScalarType::F32) return from == ScalarType::Bool ? b_.CreateUIToFP(a, to) : b_.CreateSIToFP(a, to);
                if (from == ScalarType::F32) return b_.CreateFPToSI(a, to);
                if (from == ScalarType::Bool) return b_.CreateZExt(a, to);
                return b_.CreateSExtOrTrunc(a, to);
            }
            case ExprKind::Shuffle: throw IRError("LLVM emitter: warp shuffles are GPU-only");
        }
        throw IRError("LLVM emitter: unknown expression");
    }

    llvm::Value* binary(const Expr& e) {
        if (e.binary == BinaryOp::Fma) {
            return intrinsic(llvm::Intrinsic::fmuladd, {expr(*e.args[0]), expr(*e.args[1]), expr(*e.args[2])});
        }
        llvm::Value* a = expr(*e.args[0]);
        llvm::Value* c = expr(*e.args[1]);
        if (e.type == ScalarType::F32) {
            switch (e.binary) {
                case BinaryOp::Add: return b_.CreateFAdd(a, c);
                case BinaryOp::Sub: return b_.CreateFSub(a, c);
                case BinaryOp::Mul: return b_.CreateFMul(a, c);
                case BinaryOp::Div: return b_.CreateFDiv(a, c);
                case BinaryOp::Min: return intrinsic(llvm::Intrinsic::minnum, {a, c});
                case BinaryOp::Max: return intrinsic(llvm::Intrinsic::maxnum, {a, c});
                case BinaryOp::MaxNaN: return intrinsic(llvm::Intrinsic::maximum, {a, c});
                default: break;
            }
            throw IRError("LLVM emitter: bad float op");
        }
        switch (e.binary) {
            case BinaryOp::Add: return b_.CreateAdd(a, c, "", false, true);
            case BinaryOp::Sub: return b_.CreateSub(a, c, "", false, true);
            case BinaryOp::Mul: return b_.CreateMul(a, c, "", false, true);
            case BinaryOp::Div: return b_.CreateSDiv(a, c);
            case BinaryOp::Rem: return b_.CreateSRem(a, c);
            case BinaryOp::Min: return b_.CreateSelect(b_.CreateICmpSLT(a, c), a, c);
            case BinaryOp::Max:
            case BinaryOp::MaxNaN: return b_.CreateSelect(b_.CreateICmpSGT(a, c), a, c);
            default: break;
        }
        throw IRError("LLVM emitter: bad integer op");
    }

    const Kernel& k_;
    llvm::Module& module_;
    llvm::LLVMContext& ctx_;
    llvm::IRBuilder<> b_;
    llvm::Function* func_{nullptr};
    llvm::BasicBlock* allocaBlock_{nullptr};
    std::vector<llvm::Value*> buffers_;
    llvm::Value* begin_{nullptr};
    llvm::Value* end_{nullptr};
    std::vector<std::unordered_map<std::string, Binding>> scopes_;
    int vectorDepth_{0};
};

} // namespace

std::unique_ptr<llvm::Module> emitCpuModule(const std::vector<Kernel>& kernels, llvm::LLVMContext& context) {
    auto module = std::make_unique<llvm::Module>("dlc_cpu", context);
    for (const Kernel& k : kernels) KernelEmitter(k, *module).emit();
    std::string errors;
    llvm::raw_string_ostream os(errors);
    if (llvm::verifyModule(*module, &os)) throw IRError("LLVM emitter produced invalid IR:\n" + errors);
    return module;
}

void optimizeModule(llvm::Module& module, llvm::TargetMachine& machine) {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PipelineTuningOptions options;
    options.LoopVectorization = true;
    options.SLPVectorization = true;
    llvm::PassBuilder pb(&machine, options);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(module, mam);
}

std::string moduleText(const llvm::Module& module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os, nullptr);
    return text;
}

} // namespace dlc::backends::llvm_emit
