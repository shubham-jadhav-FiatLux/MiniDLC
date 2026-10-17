#include "dlc/backends/cuda/CUDACodeGen.h"
#include "dlc/loopir/LoopIR.h"
#include <gtest/gtest.h>

// These tests check the text of the v1 CUDA templates only. Nothing here is compiled with NVRTC or run on a GPU,
// and the templates are selected by kernel name rather than generated from the loop IR (replaced in P5).

namespace {

using namespace dlc;
using namespace dlc::backends::cuda;

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

TEST(CUDACodeGen, TiledMatmulTemplateUsesSharedMemoryTiles) {
    loopir::KernelNest matmulKernel("fused_matmul_bias_relu");
    matmulKernel.addBuffer({"A", DataType::Float32, {128, 784}, {784, 1}, 0, 401408, true, false, false});
    matmulKernel.addBuffer({"B", DataType::Float32, {784, 256}, {256, 1}, 401408, 802816, true, false, false});
    matmulKernel.addBuffer({"bias", DataType::Float32, {256}, {1}, 1204224, 1024, true, false, false});
    matmulKernel.addBuffer({"C", DataType::Float32, {128, 256}, {256, 1}, 1205248, 131072, false, true, false});

    std::string code = CUDACodeGen::generateKernelCode(matmulKernel);
    EXPECT_TRUE(contains(code, "__global__ void fused_matmul_bias_relu"));
    EXPECT_TRUE(contains(code, "__shared__ float sA[TILE_DIM][TILE_DIM]"));
    EXPECT_TRUE(contains(code, "__shared__ float sB[TILE_DIM][TILE_DIM]"));
    EXPECT_TRUE(contains(code, "__syncthreads()"));
    EXPECT_TRUE(contains(code, "#pragma unroll"));
    EXPECT_TRUE(contains(code, "cuda_relu"));
}

TEST(CUDACodeGen, LaunchConfigUses16x16BlocksForMatmul) {
    CUDALaunchConfig cfg = CUDACodeGen::calculateLaunchConfig(loopir::KernelNest("matmul"));
    EXPECT_EQ(cfg.blockX, 16);
    EXPECT_EQ(cfg.blockY, 16);
    EXPECT_EQ(cfg.blockZ, 1);
    EXPECT_EQ(cfg.sharedMemBytes, static_cast<int>(2 * 16 * 16 * sizeof(float)));

    CUDALaunchConfig elemCfg = CUDACodeGen::calculateLaunchConfig(loopir::KernelNest("fused_elementwise_relu"));
    EXPECT_EQ(elemCfg.blockX, 256);
    EXPECT_EQ(elemCfg.blockY, 1);
}

TEST(CUDACodeGen, ModuleContainsWarpShuffleReductions) {
    std::vector<loopir::KernelNest> kernels;
    kernels.emplace_back("fused_matmul_bias_relu");
    kernels.emplace_back("softmax_kernel");

    std::string modCode = CUDACodeGen::generateModuleCode(kernels);
    EXPECT_TRUE(contains(modCode, "#include <cuda_runtime.h>"));
    EXPECT_TRUE(contains(modCode, "warpReduceSum"));
    EXPECT_TRUE(contains(modCode, "warpReduceMax"));
    EXPECT_TRUE(contains(modCode, "__shfl_down_sync"));
    EXPECT_TRUE(contains(modCode, "extern \"C\""));
}

TEST(CUDACodeGen, BackwardKernelTemplates) {
    loopir::KernelNest transAKernel("matmul_transA");
    transAKernel.addBuffer({"A", DataType::Float32, {128, 784}, {784, 1}, 0, 401408, true, false, false});
    transAKernel.addBuffer({"B", DataType::Float32, {128, 256}, {256, 1}, 401408, 131072, true, false, false});
    transAKernel.addBuffer({"dW", DataType::Float32, {784, 256}, {256, 1}, 532480, 802816, false, true, false});
    std::string codeTransA = CUDACodeGen::generateKernelCode(transAKernel);
    EXPECT_TRUE(contains(codeTransA, "__global__ void matmul_transA"));
    EXPECT_TRUE(contains(codeTransA, "sA[threadIdx.x][threadIdx.y]"));
    EXPECT_TRUE(contains(codeTransA, "acc += sA[threadIdx.y][k] * sB[k][threadIdx.x]"));

    loopir::KernelNest transBKernel("matmul_transB");
    transBKernel.addBuffer({"dlogits", DataType::Float32, {128, 10}, {10, 1}, 0, 5120, true, false, false});
    transBKernel.addBuffer({"W2", DataType::Float32, {256, 10}, {10, 1}, 5120, 10240, true, false, false});
    transBKernel.addBuffer({"dH", DataType::Float32, {128, 256}, {256, 1}, 15360, 131072, false, true, false});
    std::string codeTransB = CUDACodeGen::generateKernelCode(transBKernel);
    EXPECT_TRUE(contains(codeTransB, "__global__ void matmul_transB"));
    EXPECT_TRUE(contains(codeTransB, "sB[threadIdx.x][threadIdx.y]"));

    loopir::KernelNest biasKernel("bias_backward");
    biasKernel.addBuffer({"dOut", DataType::Float32, {128, 256}, {256, 1}, 0, 131072, true, false, false});
    biasKernel.addBuffer({"dbias", DataType::Float32, {256}, {1}, 131072, 1024, false, true, false});
    EXPECT_TRUE(contains(CUDACodeGen::generateKernelCode(biasKernel), "dBias[col] = acc"));

    loopir::KernelNest reluBwdKernel("relu_backward");
    reluBwdKernel.addBuffer({"dOut", DataType::Float32, {128, 256}, {256, 1}, 0, 131072, true, false, false});
    reluBwdKernel.addBuffer({"act", DataType::Float32, {128, 256}, {256, 1}, 131072, 131072, true, false, false});
    reluBwdKernel.addBuffer({"dIn", DataType::Float32, {128, 256}, {256, 1}, 262144, 131072, false, true, false});
    EXPECT_TRUE(contains(CUDACodeGen::generateKernelCode(reluBwdKernel), "dIn[i] = (h > 0.0f) ? dOut[i] : 0.0f"));

    loopir::KernelNest adamKernel("adam_update");
    adamKernel.addBuffer({"p", DataType::Float32, {256, 10}, {10, 1}, 0, 10240, false, true, false});
    adamKernel.addBuffer({"g", DataType::Float32, {256, 10}, {10, 1}, 10240, 10240, true, false, false});
    adamKernel.addBuffer({"m", DataType::Float32, {256, 10}, {10, 1}, 20480, 10240, true, false, false});
    adamKernel.addBuffer({"v", DataType::Float32, {256, 10}, {10, 1}, 30720, 10240, true, false, false});
    EXPECT_TRUE(contains(CUDACodeGen::generateKernelCode(adamKernel), "p[i] -= lr * m_hat / (sqrtf(v_hat) + eps)"));

    loopir::KernelNest embKernel("embedding_backward");
    embKernel.addBuffer({"table", DataType::Float32, {1000, 64}, {64, 1}, 0, 256000, false, true, false});
    embKernel.addBuffer({"dOut", DataType::Float32, {128, 64}, {64, 1}, 256000, 32768, true, false, false});
    EXPECT_TRUE(contains(CUDACodeGen::generateKernelCode(embKernel), "atomicAdd"));

    loopir::KernelNest ceKernel("cross_entropy_backward");
    ceKernel.addBuffer({"probs", DataType::Float32, {128, 10}, {10, 1}, 0, 5120, true, false, false});
    ceKernel.addBuffer({"dlogits", DataType::Float32, {128, 10}, {10, 1}, 5120, 5120, false, true, false});
    EXPECT_TRUE(contains(CUDACodeGen::generateKernelCode(ceKernel),
                         "dLogits[idx] = (probs[idx] - target_val) / (float)batch_sz"));
}

} // namespace
