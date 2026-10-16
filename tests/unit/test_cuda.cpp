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

int main_cuda_tests() {
    testCUDATiledMatmul();
    testCUDALaunchConfig();
    testCUDAModuleGeneration();
    return 0;
}
