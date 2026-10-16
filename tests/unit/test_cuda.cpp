#include "dlc/backends/cuda/CUDACodeGen.h"
#include "dlc/loopir/LoopIR.h"
#include <iostream>
#include <cassert>

using namespace dlc;
using namespace dlc::backends::cuda;

void testCUDATiledMatmul() {
    loopir::KernelNest matmulKernel("fused_matmul_bias_relu");
    loopir::Buffer bA{"A", DataType::Float32, {128, 784}, {784, 1}, 0, 401408, true, false, false};
    loopir::Buffer bB{"B", DataType::Float32, {784, 256}, {256, 1}, 401408, 802816, true, false, false};
    loopir::Buffer bBias{"bias", DataType::Float32, {256}, {1}, 1204224, 1024, true, false, false};
    loopir::Buffer bOut{"C", DataType::Float32, {128, 256}, {256, 1}, 1205248, 131072, false, true, false};
    matmulKernel.addBuffer(bA);
    matmulKernel.addBuffer(bB);
    matmulKernel.addBuffer(bBias);
    matmulKernel.addBuffer(bOut);

    std::string code = CUDACodeGen::generateKernelCode(matmulKernel);

    // Verify key CS6023 GPU concepts in generated code
    assert(code.find("__global__ void fused_matmul_bias_relu") != std::string::npos);
    assert(code.find("__shared__ float sA[TILE_DIM][TILE_DIM]") != std::string::npos);
    assert(code.find("__shared__ float sB[TILE_DIM][TILE_DIM]") != std::string::npos);
    assert(code.find("__syncthreads()") != std::string::npos);
    assert(code.find("#pragma unroll") != std::string::npos);
    assert(code.find("cuda_relu") != std::string::npos);

    std::cout << "[PASS] testCUDATiledMatmul\n";
}

void testCUDALaunchConfig() {
    loopir::KernelNest matmulKernel("matmul");
    CUDALaunchConfig cfg = CUDACodeGen::calculateLaunchConfig(matmulKernel);

    // 2D 16x16 thread block
    assert(cfg.blockX == 16);
    assert(cfg.blockY == 16);
    assert(cfg.blockZ == 1);
    assert(cfg.sharedMemBytes == 2 * 16 * 16 * sizeof(float));
    (void)cfg;

    loopir::KernelNest elemKernel("fused_elementwise_relu");
    CUDALaunchConfig elemCfg = CUDACodeGen::calculateLaunchConfig(elemKernel);
    assert(elemCfg.blockX == 256);
    assert(elemCfg.blockY == 1);
    (void)elemCfg;

    std::cout << "[PASS] testCUDALaunchConfig\n";
}

void testCUDAModuleGeneration() {
    std::vector<loopir::KernelNest> kernels;
    loopir::KernelNest k1("fused_matmul_bias_relu");
    loopir::KernelNest k2("softmax_kernel");
    kernels.push_back(k1);
    kernels.push_back(k2);

    std::string modCode = CUDACodeGen::generateModuleCode(kernels);
    assert(modCode.find("#include <cuda_runtime.h>") != std::string::npos);
    assert(modCode.find("warpReduceSum") != std::string::npos);
    assert(modCode.find("warpReduceMax") != std::string::npos);
    assert(modCode.find("__shfl_down_sync") != std::string::npos);
    assert(modCode.find("extern \"C\"") != std::string::npos);

    std::cout << "[PASS] testCUDAModuleGeneration\n";
}

void testCUDABackwardKernels() {
    // 1. matmul_transA
    loopir::KernelNest transAKernel("matmul_transA");
    loopir::Buffer bA{"A", DataType::Float32, {128, 784}, {784, 1}, 0, 401408, true, false, false};
    loopir::Buffer bB{"B", DataType::Float32, {128, 256}, {256, 1}, 401408, 131072, true, false, false};
    loopir::Buffer bOut{"dW", DataType::Float32, {784, 256}, {256, 1}, 532480, 802816, false, true, false};
    transAKernel.addBuffer(bA);
    transAKernel.addBuffer(bB);
    transAKernel.addBuffer(bOut);
    std::string codeTransA = CUDACodeGen::generateKernelCode(transAKernel);
    assert(codeTransA.find("__global__ void matmul_transA") != std::string::npos);
    assert(codeTransA.find("sA[threadIdx.x][threadIdx.y]") != std::string::npos);
    assert(codeTransA.find("acc += sA[threadIdx.y][k] * sB[k][threadIdx.x]") != std::string::npos);

    // 2. matmul_transB
    loopir::KernelNest transBKernel("matmul_transB");
    loopir::Buffer bDLogits{"dlogits", DataType::Float32, {128, 10}, {10, 1}, 0, 5120, true, false, false};
    loopir::Buffer bW2{"W2", DataType::Float32, {256, 10}, {10, 1}, 5120, 10240, true, false, false};
    loopir::Buffer bDH{"dH", DataType::Float32, {128, 256}, {256, 1}, 15360, 131072, false, true, false};
    transBKernel.addBuffer(bDLogits);
    transBKernel.addBuffer(bW2);
    transBKernel.addBuffer(bDH);
    std::string codeTransB = CUDACodeGen::generateKernelCode(transBKernel);
    assert(codeTransB.find("__global__ void matmul_transB") != std::string::npos);
    assert(codeTransB.find("sB[threadIdx.x][threadIdx.y]") != std::string::npos);
    assert(codeTransB.find("acc += sA[threadIdx.y][k] * sB[k][threadIdx.x]") != std::string::npos);

    // 3. bias_backward
    loopir::KernelNest biasKernel("bias_backward");
    loopir::Buffer bD{"dOut", DataType::Float32, {128, 256}, {256, 1}, 0, 131072, true, false, false};
    loopir::Buffer bDb{"dbias", DataType::Float32, {256}, {1}, 131072, 1024, false, true, false};
    biasKernel.addBuffer(bD);
    biasKernel.addBuffer(bDb);
    std::string codeBias = CUDACodeGen::generateKernelCode(biasKernel);
    assert(codeBias.find("__global__ void bias_backward") != std::string::npos);
    assert(codeBias.find("dBias[col] = acc") != std::string::npos);

    // 4. relu_backward
    loopir::KernelNest reluBwdKernel("relu_backward");
    loopir::Buffer bDOut{"dOut", DataType::Float32, {128, 256}, {256, 1}, 0, 131072, true, false, false};
    loopir::Buffer bAct{"act", DataType::Float32, {128, 256}, {256, 1}, 131072, 131072, true, false, false};
    loopir::Buffer bDIn{"dIn", DataType::Float32, {128, 256}, {256, 1}, 262144, 131072, false, true, false};
    reluBwdKernel.addBuffer(bDOut);
    reluBwdKernel.addBuffer(bAct);
    reluBwdKernel.addBuffer(bDIn);
    std::string codeRelu = CUDACodeGen::generateKernelCode(reluBwdKernel);
    assert(codeRelu.find("__global__ void relu_backward") != std::string::npos);
    assert(codeRelu.find("dIn[i] = (h > 0.0f) ? dOut[i] : 0.0f") != std::string::npos);

    // 5. adam_update
    loopir::KernelNest adamKernel("adam_update");
    loopir::Buffer bP{"p", DataType::Float32, {256, 10}, {10, 1}, 0, 10240, false, true, false};
    loopir::Buffer bG{"g", DataType::Float32, {256, 10}, {10, 1}, 10240, 10240, true, false, false};
    loopir::Buffer bM{"m", DataType::Float32, {256, 10}, {10, 1}, 20480, 10240, true, false, false};
    loopir::Buffer bV{"v", DataType::Float32, {256, 10}, {10, 1}, 30720, 10240, true, false, false};
    adamKernel.addBuffer(bP);
    adamKernel.addBuffer(bG);
    adamKernel.addBuffer(bM);
    adamKernel.addBuffer(bV);
    std::string codeAdam = CUDACodeGen::generateKernelCode(adamKernel);
    assert(codeAdam.find("__global__ void adam_update") != std::string::npos);
    assert(codeAdam.find("p[i] -= lr * m_hat / (sqrtf(v_hat) + eps)") != std::string::npos);

    // 6. embedding_backward
    loopir::KernelNest embKernel("embedding_backward");
    loopir::Buffer bTable{"table", DataType::Float32, {1000, 64}, {64, 1}, 0, 256000, false, true, false};
    loopir::Buffer bDGrad{"dOut", DataType::Float32, {128, 64}, {64, 1}, 256000, 32768, true, false, false};
    embKernel.addBuffer(bTable);
    embKernel.addBuffer(bDGrad);
    std::string codeEmb = CUDACodeGen::generateKernelCode(embKernel);
    assert(codeEmb.find("__global__ void embedding_backward") != std::string::npos);
    assert(codeEmb.find("atomicAdd") != std::string::npos);

    // 7. cross_entropy_backward
    loopir::KernelNest ceKernel("cross_entropy_backward");
    loopir::Buffer bProbs{"probs", DataType::Float32, {128, 10}, {10, 1}, 0, 5120, true, false, false};
    loopir::Buffer bLogits{"dlogits", DataType::Float32, {128, 10}, {10, 1}, 5120, 5120, false, true, false};
    ceKernel.addBuffer(bProbs);
    ceKernel.addBuffer(bLogits);
    std::string codeCE = CUDACodeGen::generateKernelCode(ceKernel);
    assert(codeCE.find("__global__ void cross_entropy_backward") != std::string::npos);
    assert(codeCE.find("dLogits[idx] = (probs[idx] - target_val) / (float)batch_sz") != std::string::npos);

    std::cout << "[PASS] testCUDABackwardKernels\n";
}

int main_cuda_tests() {
    testCUDATiledMatmul();
    testCUDALaunchConfig();
    testCUDAModuleGeneration();
    testCUDABackwardKernels();
    return 0;
}
