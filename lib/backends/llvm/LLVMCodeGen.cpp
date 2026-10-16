#include "dlc/backends/llvm/LLVMCodeGen.h"
#include <sstream>
#include <iostream>

namespace dlc::backends::llvm {

std::string LLVMCodeGen::generateModuleIR(const std::vector<loopir::KernelNest>& kernels) {
    std::ostringstream ss;
    ss << "; ModuleID = 'dlc_cpu_module'\n";
    ss << "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n";
    ss << "target triple = \"x86_64-pc-linux-gnu\"\n\n";

    // Standard LLVM vector intrinsics
    ss << "declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>)\n";
    ss << "declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>)\n";
    ss << "declare float @llvm.fmuladd.f32(float, float, float)\n";
    ss << "declare float @llvm.maximum.f32(float, float)\n";
    ss << "declare float @llvm.exp.f32(float)\n";
    ss << "declare float @llvm.log.f32(float)\n";
    ss << "declare float @llvm.sqrt.f32(float)\n\n";

    for (const auto& k : kernels) {
        ss << generateKernelIR(k) << "\n";
    }

    return ss.str();
}

static bool isMatmulKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("matmul") != std::string::npos;
}

std::string LLVMCodeGen::generateKernelIR(const loopir::KernelNest& kernel) {
    std::ostringstream ss;
    const std::string& name = kernel.getName();
    const auto& buffers = kernel.getBuffers();

    ss << "define void @" << name << "(ptr %arena, i64 %start, i64 %end) #0 {\n";
    ss << "entry:\n";

    // 1. Resolve buffer base pointers in arena
    for (const auto& b : buffers) {
        ss << "  %ptr_" << b.name << " = getelementptr inbounds i8, ptr %arena, i64 "
           << b.arenaOffsetBytes << "\n";
    }

    if (isMatmulKernel(kernel)) {
        // Matmul or Fused Matmul + Bias + ReLU
        // Find A, B, Bias (if any), Out
        const loopir::Buffer* bufA = nullptr;
        const loopir::Buffer* bufB = nullptr;
        const loopir::Buffer* bufBias = nullptr;
        const loopir::Buffer* bufOut = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) {
                bufOut = &b;
            } else if (!bufA) {
                bufA = &b;
            } else if (!bufB) {
                bufB = &b;
            } else if (!bufBias) {
                bufBias = &b;
            }
        }

        if (!bufA || !bufB || !bufOut || bufA->shape.size() < 2 || bufB->shape.size() < 2) {
            // Fallback empty return
            ss << "  ret void\n}\n";
            return ss.str();
        }

        int64_t M = bufA->shape[0];
        int64_t K = bufA->shape[1];
        int64_t N = bufB->shape[1];
        bool hasBias = (bufBias != nullptr);
        bool hasRelu = (name.find("relu") != std::string::npos);

        ss << "  %has_work = icmp slt i64 %start, %end\n";
        ss << "  br i1 %has_work, label %loop_i_start, label %exit\n\n";

        ss << "loop_i_start:\n";
        ss << "  br label %loop_i\n\n";

        ss << "loop_i:\n";
        ss << "  %i = phi i64 [ %start, %loop_i_start ], [ %i_next, %loop_j_end ]\n";
        ss << "  br label %loop_j\n\n";

        ss << "loop_j:\n";
        ss << "  %j = phi i64 [ 0, %loop_i ], [ %j_next, %loop_k_done ]\n";
        ss << "  br label %loop_k\n\n";


        ss << "loop_k:\n";
        ss << "  %k = phi i64 [ 0, %loop_j ], [ %k_next, %loop_k ]\n";
        ss << "  %acc = phi <8 x float> [ zeroinitializer, %loop_j ], [ %acc_fma, %loop_k ]\n";

        // Load A[i, k] and broadcast to vector
        ss << "  %ik = mul i64 %i, " << K << "\n";
        ss << "  %idx_a = add i64 %ik, %k\n";
        ss << "  %ptr_a_elem = getelementptr inbounds float, ptr %ptr_" << bufA->name << ", i64 %idx_a\n";
        ss << "  %val_a = load float, ptr %ptr_a_elem, align 4\n";
        ss << "  %vec_a_0 = insertelement <8 x float> poison, float %val_a, i32 0\n";
        ss << "  %splat_a = shufflevector <8 x float> %vec_a_0, <8 x float> poison, <8 x i32> zeroinitializer\n";

        // Load B[k, j..j+7]
        ss << "  %kn = mul i64 %k, " << N << "\n";
        ss << "  %idx_b = add i64 %kn, %j\n";
        ss << "  %ptr_b_elem = getelementptr inbounds float, ptr %ptr_" << bufB->name << ", i64 %idx_b\n";
        ss << "  %vec_b = load <8 x float>, ptr %ptr_b_elem, align 4\n";

        // FMA
        ss << "  %acc_fma = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %splat_a, <8 x float> %vec_b, <8 x float> %acc)\n";

        // k-loop test
        ss << "  %k_next = add i64 %k, 1\n";
        ss << "  %cond_k = icmp slt i64 %k_next, " << K << "\n";
        ss << "  br i1 %cond_k, label %loop_k, label %loop_k_done\n\n";

        ss << "loop_k_done:\n";
        std::string curAcc = "%acc_fma";

        if (hasBias) {
            ss << "  %ptr_bias_elem = getelementptr inbounds float, ptr %ptr_" << bufBias->name << ", i64 %j\n";
            ss << "  %vec_bias = load <8 x float>, ptr %ptr_bias_elem, align 4\n";
            ss << "  %acc_with_bias = fadd <8 x float> " << curAcc << ", %vec_bias\n";
            curAcc = "%acc_with_bias";
        }

        if (hasRelu) {
            ss << "  %acc_relu = call <8 x float> @llvm.maximum.v8f32(<8 x float> " << curAcc
               << ", <8 x float> zeroinitializer)\n";
            curAcc = "%acc_relu";
        }

        // Store C[i, j..j+7]
        ss << "  %in = mul i64 %i, " << N << "\n";
        ss << "  %idx_c = add i64 %in, %j\n";
        ss << "  %ptr_c_elem = getelementptr inbounds float, ptr %ptr_" << bufOut->name << ", i64 %idx_c\n";
        ss << "  store <8 x float> " << curAcc << ", ptr %ptr_c_elem, align 4\n";

        // j-loop test
        ss << "  %j_next = add i64 %j, 8\n";
        ss << "  %cond_j = icmp slt i64 %j_next, " << N << "\n";
        ss << "  br i1 %cond_j, label %loop_j, label %loop_j_end\n\n";

        ss << "loop_j_end:\n";
        ss << "  %i_next = add i64 %i, 1\n";
        ss << "  %cond_i = icmp slt i64 %i_next, %end\n";
        ss << "  br i1 %cond_i, label %loop_i, label %exit\n\n";

    } else {
        // Elementwise or Generic Kernel
        const loopir::Buffer* bufOut = nullptr;
        std::vector<const loopir::Buffer*> inBufs;
        for (const auto& b : buffers) {
            if (b.isOutput) bufOut = &b;
            else inBufs.push_back(&b);
        }

        if (!bufOut) {
            ss << "  ret void\n}\n";
            return ss.str();
        }

        int64_t totalElements = bufOut->getNumElements();
        bool hasRelu = (name.find("relu") != std::string::npos);
        bool hasMul = (name.find("mul") != std::string::npos);
        bool hasSub = (name.find("sub") != std::string::npos);

        ss << "  %range_valid = icmp slt i64 %start, %end\n";
        ss << "  br i1 %range_valid, label %vec_check, label %exit\n\n";

        ss << "vec_check:\n";
        ss << "  br label %loop_vec\n\n";

        ss << "loop_vec:\n";
        ss << "  %idx = phi i64 [ %start, %vec_check ], [ %idx_next, %loop_vec ]\n";

        if (inBufs.size() >= 2) {
            // Binary op
            ss << "  %ptr_in0 = getelementptr inbounds float, ptr %ptr_" << inBufs[0]->name << ", i64 %idx\n";
            ss << "  %val_in0 = load <8 x float>, ptr %ptr_in0, align 4\n";

            // If second input is broadcasted (smaller shape), load scalar or vector
            if (inBufs[1]->getNumElements() == 1) {
                ss << "  %val_in1_s = load float, ptr %ptr_" << inBufs[1]->name << ", align 4\n";
                ss << "  %vec_in1_0 = insertelement <8 x float> poison, float %val_in1_s, i32 0\n";
                ss << "  %val_in1 = shufflevector <8 x float> %vec_in1_0, <8 x float> poison, <8 x i32> zeroinitializer\n";
            } else {
                ss << "  %ptr_in1 = getelementptr inbounds float, ptr %ptr_" << inBufs[1]->name << ", i64 %idx\n";
                ss << "  %val_in1 = load <8 x float>, ptr %ptr_in1, align 4\n";
            }

            if (hasSub) {
                ss << "  %res_op = fsub <8 x float> %val_in0, %val_in1\n";
            } else if (hasMul) {
                ss << "  %res_op = fmul <8 x float> %val_in0, %val_in1\n";
            } else {
                ss << "  %res_op = fadd <8 x float> %val_in0, %val_in1\n";
            }
        } else if (inBufs.size() == 1) {
            // Unary op
            ss << "  %ptr_in = getelementptr inbounds float, ptr %ptr_" << inBufs[0]->name << ", i64 %idx\n";
            ss << "  %val_in = load <8 x float>, ptr %ptr_in, align 4\n";
            ss << "  %res_op = fadd <8 x float> %val_in, zeroinitializer\n";
        } else {
            ss << "  %res_op = fadd <8 x float> zeroinitializer, zeroinitializer\n";
        }

        std::string finalVal = "%res_op";
        if (hasRelu) {
            ss << "  %res_relu = call <8 x float> @llvm.maximum.v8f32(<8 x float> %res_op, <8 x float> zeroinitializer)\n";
            finalVal = "%res_relu";
        }

        ss << "  %ptr_out = getelementptr inbounds float, ptr %ptr_" << bufOut->name << ", i64 %idx\n";
        ss << "  store <8 x float> " << finalVal << ", ptr %ptr_out, align 4\n";

        ss << "  %idx_next = add i64 %idx, 8\n";
        ss << "  %cond_vec = icmp slt i64 %idx_next, %end\n";
        ss << "  br i1 %cond_vec, label %loop_vec, label %exit\n\n";
    }

    ss << "exit:\n";
    ss << "  ret void\n";
    ss << "}\n";

    return ss.str();
}

} // namespace dlc::backends::llvm
