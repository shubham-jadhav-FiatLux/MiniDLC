#include "dlc/backends/cuda/CUDACodeGen.h"
#include <sstream>
#include <iomanip>

namespace dlc::backends::cuda {

static bool isMatmulKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("matmul") != std::string::npos;
}

static bool isSoftmaxKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("softmax") != std::string::npos;
}

static bool isEmbeddingKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("embedding") != std::string::npos;
}

std::string CUDACodeGen::generateModuleCode(const std::vector<loopir::KernelNest>& kernels) {
    std::ostringstream ss;
    ss << "// ============================================================================\n";
    ss << "// MiniDLC Generated CUDA GPU Module\n";
    ss << "// Generated for NVIDIA CUDA architecture (shared memory tiling + warp shuffles)\n";
    ss << "// ============================================================================\n\n";

    ss << "#include <cuda_runtime.h>\n";
    ss << "#include <math.h>\n\n";

    ss << "#define WARP_SIZE 32\n\n";

    // Warp-level parallel reduction tree via __shfl_down_sync (CS6023)
    ss << "// Warp-level parallel reduction tree via __shfl_down_sync (CS6023)\n";
    ss << "__device__ inline float warpReduceSum(float val) {\n";
    ss << "    #pragma unroll\n";
    ss << "    for (int offset = 16; offset > 0; offset /= 2) {\n";
    ss << "        val += __shfl_down_sync(0xffffffff, val, offset);\n";
    ss << "    }\n";
    ss << "    return val;\n";
    ss << "}\n\n";

    // Warp-level parallel reduction max
    ss << "// Warp-level parallel reduction max via __shfl_down_sync (CS6023)\n";
    ss << "__device__ inline float warpReduceMax(float val) {\n";
    ss << "    #pragma unroll\n";
    ss << "    for (int offset = 16; offset > 0; offset /= 2) {\n";
    ss << "        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));\n";
    ss << "    }\n";
    ss << "    return val;\n";
    ss << "}\n\n";

    // NaN-propagating ReLU matching PyTorch / IEEE-754 semantics
    ss << "// NaN-propagating ReLU matching PyTorch semantics\n";
    ss << "__device__ inline float cuda_relu(float x) {\n";
    ss << "    return (x != x) ? x : (x > 0.0f ? x : 0.0f);\n";
    ss << "}\n\n";

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ss << "// GELU activation function\n";
    ss << "__device__ inline float cuda_gelu(float x) {\n";
    ss << "    const float k0 = 0.7978845608028654f; // sqrt(2/pi)\n";
    ss << "    const float k1 = 0.044715f;\n";
    ss << "    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));\n";
    ss << "}\n\n";

    ss << "extern \"C\" {\n\n";

    for (const auto& k : kernels) {
        ss << generateKernelCode(k) << "\n\n";
    }

    ss << "} // extern \"C\"\n";
    return ss.str();
}

std::string CUDACodeGen::generateKernelCode(const loopir::KernelNest& kernel) {
    std::ostringstream ss;
    const std::string& name = kernel.getName();
    const auto& buffers = kernel.getBuffers();

    // 1. Shared-memory Tiled Matmul with Fused Epilogues
    if (isMatmulKernel(kernel)) {
        bool hasBias = name.find("bias") != std::string::npos;
        bool hasRelu = name.find("relu") != std::string::npos;

        const loopir::Buffer* bufA = nullptr;
        const loopir::Buffer* bufB = nullptr;
        const loopir::Buffer* bufBias = nullptr;
        const loopir::Buffer* bufOut = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufOut = &b;
            else if (!bufA) bufA = &b;
            else if (!bufB) bufB = &b;
            else if (!bufBias) bufBias = &b;
        }

        ss << "// Shared-memory tiled matrix multiplication kernel with fused epilogue (CS6023)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int M, int K, int N) {\n";
        ss << "    const int TILE_DIM = 16;\n";
        ss << "    __shared__ float sA[TILE_DIM][TILE_DIM];\n";
        ss << "    __shared__ float sB[TILE_DIM][TILE_DIM];\n\n";

        // Buffer pointers
        if (bufA) ss << "    const float* __restrict__ A = (const float*)(arena + " << bufA->arenaOffsetBytes << ");\n";
        if (bufB) ss << "    const float* __restrict__ B = (const float*)(arena + " << bufB->arenaOffsetBytes << ");\n";
        if (bufBias && hasBias) ss << "    const float* __restrict__ bias = (const float*)(arena + " << bufBias->arenaOffsetBytes << ");\n";
        if (bufOut) ss << "    float* __restrict__ out = (float*)(arena + " << bufOut->arenaOffsetBytes << ");\n\n";

        ss << "    int row = blockIdx.y * TILE_DIM + threadIdx.y;\n";
        ss << "    int col = blockIdx.x * TILE_DIM + threadIdx.x;\n";
        ss << "    float acc = 0.0f;\n\n";

        ss << "    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;\n";
        ss << "    for (int t = 0; t < numTiles; ++t) {\n";
        ss << "        int tCol = t * TILE_DIM + threadIdx.x;\n";
        ss << "        int tRow = t * TILE_DIM + threadIdx.y;\n";
        ss << "        sA[threadIdx.y][threadIdx.x] = (row < M && tCol < K) ? A[row * K + tCol] : 0.0f;\n";
        ss << "        sB[threadIdx.y][threadIdx.x] = (tRow < K && col < N) ? B[tRow * N + col] : 0.0f;\n";
        ss << "        __syncthreads();\n\n";
        ss << "        #pragma unroll\n";
        ss << "        for (int k = 0; k < TILE_DIM; ++k) {\n";
        ss << "            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n\n";

        ss << "    if (row < M && col < N) {\n";
        if (hasBias && hasRelu) {
            ss << "        acc += bias[col];\n";
            ss << "        out[row * N + col] = cuda_relu(acc);\n";
        } else if (hasBias) {
            ss << "        acc += bias[col];\n";
            ss << "        out[row * N + col] = acc;\n";
        } else if (hasRelu) {
            ss << "        out[row * N + col] = cuda_relu(acc);\n";
        } else {
            ss << "        out[row * N + col] = acc;\n";
        }
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 2. Row-wise Softmax / Cross-Entropy
    if (isSoftmaxKernel(kernel)) {
        const loopir::Buffer* bufIn = nullptr;
        const loopir::Buffer* bufOut = nullptr;
        for (const auto& b : buffers) {
            if (b.isOutput) bufOut = &b;
            else if (!bufIn) bufIn = &b;
        }

        ss << "// Row-wise softmax reduction kernel (one block per row, warp shuffle tree)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int rows, int cols) {\n";
        ss << "    int row = blockIdx.x;\n";
        ss << "    if (row >= rows) return;\n\n";

        if (bufIn) ss << "    const float* __restrict__ in_row = (const float*)(arena + " << bufIn->arenaOffsetBytes << ") + row * cols;\n";
        if (bufOut) ss << "    float* __restrict__ out_row = (float*)(arena + " << bufOut->arenaOffsetBytes << ") + row * cols;\n\n";

        ss << "    __shared__ float s_max;\n";
        ss << "    __shared__ float s_sum;\n\n";

        // Step 1: Max reduction
        ss << "    float local_max = -1e30f;\n";
        ss << "    for (int c = threadIdx.x; c < cols; c += blockDim.x) {\n";
        ss << "        local_max = fmaxf(local_max, in_row[c]);\n";
        ss << "    }\n";
        ss << "    local_max = warpReduceMax(local_max);\n";
        ss << "    if (threadIdx.x == 0) s_max = local_max;\n";
        ss << "    __syncthreads();\n\n";

        // Step 2: Sum of exponentials
        ss << "    float local_sum = 0.0f;\n";
        ss << "    for (int c = threadIdx.x; c < cols; c += blockDim.x) {\n";
        ss << "        local_sum += expf(in_row[c] - s_max);\n";
        ss << "    }\n";
        ss << "    local_sum = warpReduceSum(local_sum);\n";
        ss << "    if (threadIdx.x == 0) s_sum = local_sum;\n";
        ss << "    __syncthreads();\n\n";

        // Step 3: Write normalized probability
        ss << "    float inv_sum = 1.0f / s_sum;\n";
        ss << "    for (int c = threadIdx.x; c < cols; c += blockDim.x) {\n";
        ss << "        out_row[c] = expf(in_row[c] - s_max) * inv_sum;\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 3. Embedding Gather / Scatter
    if (isEmbeddingKernel(kernel)) {
        ss << "// Embedding lookup and scatter-add atomic kernel (CS6023)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, const int* __restrict__ idx, int seq_len, int d_model) {\n";
        ss << "    int i = blockIdx.y * blockDim.y + threadIdx.y;\n";
        ss << "    int d = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (i < seq_len && d < d_model) {\n";
        const loopir::Buffer* bufTable = buffers.size() > 0 ? &buffers[0] : nullptr;
        const loopir::Buffer* bufOut = buffers.size() > 1 ? &buffers[1] : nullptr;
        if (bufTable && bufOut) {
            ss << "        const float* __restrict__ table = (const float*)(arena + " << bufTable->arenaOffsetBytes << ");\n";
            ss << "        float* __restrict__ out = (float*)(arena + " << bufOut->arenaOffsetBytes << ");\n";
            ss << "        out[i * d_model + d] = table[idx[i] * d_model + d];\n";
        }
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 4. Elementwise / Vectorized / Adam Optimizer Kernel
    int64_t totalElements = 0;
    const loopir::Buffer* bufOut = nullptr;
    std::vector<const loopir::Buffer*> inBufs;
    for (const auto& b : buffers) {
        if (b.isOutput) bufOut = &b;
        else inBufs.push_back(&b);
        if (b.sizeBytes > 0) totalElements = std::max(totalElements, b.sizeBytes / 4);
    }

    ss << "// Elementwise grid-stride vector kernel\n";
    ss << "__global__ void " << name << "(char* __restrict__ arena, int numElements) {\n";
    ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    int stride = blockDim.x * gridDim.x;\n\n";

    if (bufOut) {
        ss << "    float* __restrict__ out = (float*)(arena + " << bufOut->arenaOffsetBytes << ");\n";
    }
    for (size_t i = 0; i < inBufs.size(); ++i) {
        ss << "    const float* __restrict__ in" << i << " = (const float*)(arena + " << inBufs[i]->arenaOffsetBytes << ");\n";
    }

    ss << "\n    for (int i = idx; i < numElements; i += stride) {\n";

    if (name.find("relu") != std::string::npos && !inBufs.empty() && bufOut) {
        ss << "        out[i] = cuda_relu(in0[i]);\n";
    } else if (name.find("gelu") != std::string::npos && !inBufs.empty() && bufOut) {
        ss << "        out[i] = cuda_gelu(in0[i]);\n";
    } else if (name.find("add") != std::string::npos && inBufs.size() >= 2 && bufOut) {
        ss << "        out[i] = in0[i] + in1[i];\n";
    } else if (name.find("mul") != std::string::npos && inBufs.size() >= 2 && bufOut) {
        ss << "        out[i] = in0[i] * in1[i];\n";
    } else if (bufOut && !inBufs.empty()) {
        ss << "        out[i] = in0[i];\n";
    }
    ss << "    }\n";
    ss << "}";
    return ss.str();
}

CUDALaunchConfig CUDACodeGen::calculateLaunchConfig(const loopir::KernelNest& kernel) {
    CUDALaunchConfig cfg;

    if (isMatmulKernel(kernel)) {
        // 16x16 2D thread block
        cfg.blockX = 16;
        cfg.blockY = 16;
        cfg.blockZ = 1;
        // Defaults for batch 128 x 256 output
        cfg.gridX = (256 + 15) / 16;
        cfg.gridY = (128 + 15) / 16;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 2 * 16 * 16 * sizeof(float); // sA + sB
    } else if (isSoftmaxKernel(kernel)) {
        // One block per row, 256 threads per row
        cfg.blockX = 256;
        cfg.blockY = 1;
        cfg.blockZ = 1;
        cfg.gridX = 128; // num rows
        cfg.gridY = 1;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 64 * sizeof(float);
    } else {
        // 1D grid-stride loop
        cfg.blockX = 256;
        cfg.blockY = 1;
        cfg.blockZ = 1;
        cfg.gridX = 128;
        cfg.gridY = 1;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 0;
    }

    return cfg;
}

} // namespace dlc::backends::cuda
