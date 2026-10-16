#include "dlc/backends/cuda/CUDACodeGen.h"
#include <sstream>
#include <iomanip>

namespace dlc::backends::cuda {

static bool isMatmulTransAKernel(const loopir::KernelNest& kernel) {
    const auto& name = kernel.getName();
    return name.find("matmul_transA") != std::string::npos || name.find("transA") != std::string::npos;
}

static bool isMatmulTransBKernel(const loopir::KernelNest& kernel) {
    const auto& name = kernel.getName();
    return name.find("matmul_transB") != std::string::npos || name.find("transB") != std::string::npos;
}

static bool isMatmulKernel(const loopir::KernelNest& kernel) {
    const auto& name = kernel.getName();
    return name.find("matmul") != std::string::npos &&
           !isMatmulTransAKernel(kernel) &&
           !isMatmulTransBKernel(kernel);
}

static bool isBiasBackwardKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("bias_backward") != std::string::npos;
}

static bool isReluBackwardKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("relu_backward") != std::string::npos;
}

static bool isAdamKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("adam") != std::string::npos;
}

static bool isCrossEntropyBackwardKernel(const loopir::KernelNest& kernel) {
    const auto& name = kernel.getName();
    return name.find("cross_entropy_backward") != std::string::npos ||
           name.find("cross_entropy_grad") != std::string::npos;
}

static bool isEmbeddingBackwardKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("embedding_backward") != std::string::npos;
}

static bool isSoftmaxKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("softmax") != std::string::npos;
}

static bool isEmbeddingKernel(const loopir::KernelNest& kernel) {
    return kernel.getName().find("embedding") != std::string::npos &&
           !isEmbeddingBackwardKernel(kernel);
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

    // 1. Matmul Transpose A: C[M, N] = A.T[M, K] @ B[K, N] (A is [K, M], B is [K, N])
    if (isMatmulTransAKernel(kernel)) {
        const loopir::Buffer* bufA = nullptr;
        const loopir::Buffer* bufB = nullptr;
        const loopir::Buffer* bufOut = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufOut = &b;
            else if (!bufA) bufA = &b;
            else if (!bufB) bufB = &b;
        }

        int64_t offsetA = bufA ? bufA->arenaOffsetBytes : 0;
        int64_t offsetB = bufB ? bufB->arenaOffsetBytes : 0;
        int64_t offsetOut = bufOut ? bufOut->arenaOffsetBytes : 0;

        ss << "// Shared-memory tiled matrix multiplication kernel (Transposed A: C = A^T @ B)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int M, int K, int N) {\n";
        ss << "    const int TILE_DIM = 16;\n";
        ss << "    __shared__ float sA[TILE_DIM][TILE_DIM];\n";
        ss << "    __shared__ float sB[TILE_DIM][TILE_DIM];\n\n";

        ss << "    const float* __restrict__ A = (const float*)(arena + " << offsetA << ");\n";
        ss << "    const float* __restrict__ B = (const float*)(arena + " << offsetB << ");\n";
        ss << "    float* __restrict__ out = (float*)(arena + " << offsetOut << ");\n\n";

        ss << "    int row = blockIdx.y * TILE_DIM + threadIdx.y;\n";
        ss << "    int col = blockIdx.x * TILE_DIM + threadIdx.x;\n";
        ss << "    float acc = 0.0f;\n\n";

        ss << "    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;\n";
        ss << "    for (int t = 0; t < numTiles; ++t) {\n";
        ss << "        // Coalesced global load for A with transposition into shared memory\n";
        ss << "        int k_idx_A = t * TILE_DIM + threadIdx.y;\n";
        ss << "        int m_idx_A = blockIdx.y * TILE_DIM + threadIdx.x;\n";
        ss << "        sA[threadIdx.x][threadIdx.y] = (k_idx_A < K && m_idx_A < M) ? A[k_idx_A * M + m_idx_A] : 0.0f;\n\n";

        ss << "        // Coalesced global load for B\n";
        ss << "        int k_idx_B = t * TILE_DIM + threadIdx.y;\n";
        ss << "        int n_idx_B = blockIdx.x * TILE_DIM + threadIdx.x;\n";
        ss << "        sB[threadIdx.y][threadIdx.x] = (k_idx_B < K && n_idx_B < N) ? B[k_idx_B * N + n_idx_B] : 0.0f;\n";
        ss << "        __syncthreads();\n\n";

        ss << "        #pragma unroll\n";
        ss << "        for (int k = 0; k < TILE_DIM; ++k) {\n";
        ss << "            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n\n";

        ss << "    if (row < M && col < N) {\n";
        ss << "        out[row * N + col] = acc;\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 2. Matmul Transpose B: C[M, N] = A[M, K] @ B.T[K, N] (A is [M, K], B is [N, K])
    if (isMatmulTransBKernel(kernel)) {
        const loopir::Buffer* bufA = nullptr;
        const loopir::Buffer* bufB = nullptr;
        const loopir::Buffer* bufOut = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufOut = &b;
            else if (!bufA) bufA = &b;
            else if (!bufB) bufB = &b;
        }

        int64_t offsetA = bufA ? bufA->arenaOffsetBytes : 0;
        int64_t offsetB = bufB ? bufB->arenaOffsetBytes : 0;
        int64_t offsetOut = bufOut ? bufOut->arenaOffsetBytes : 0;

        ss << "// Shared-memory tiled matrix multiplication kernel (Transposed B: C = A @ B^T)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int M, int K, int N) {\n";
        ss << "    const int TILE_DIM = 16;\n";
        ss << "    __shared__ float sA[TILE_DIM][TILE_DIM];\n";
        ss << "    __shared__ float sB[TILE_DIM][TILE_DIM];\n\n";

        ss << "    const float* __restrict__ A = (const float*)(arena + " << offsetA << ");\n";
        ss << "    const float* __restrict__ B = (const float*)(arena + " << offsetB << ");\n";
        ss << "    float* __restrict__ out = (float*)(arena + " << offsetOut << ");\n\n";

        ss << "    int row = blockIdx.y * TILE_DIM + threadIdx.y;\n";
        ss << "    int col = blockIdx.x * TILE_DIM + threadIdx.x;\n";
        ss << "    float acc = 0.0f;\n\n";

        ss << "    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;\n";
        ss << "    for (int t = 0; t < numTiles; ++t) {\n";
        ss << "        // Coalesced global load for A\n";
        ss << "        int k_idx_A = t * TILE_DIM + threadIdx.x;\n";
        ss << "        sA[threadIdx.y][threadIdx.x] = (row < M && k_idx_A < K) ? A[row * K + k_idx_A] : 0.0f;\n\n";

        ss << "        // Coalesced global load for B with transposition into shared memory\n";
        ss << "        int n_idx_B = blockIdx.x * TILE_DIM + threadIdx.y;\n";
        ss << "        int k_idx_B = t * TILE_DIM + threadIdx.x;\n";
        ss << "        sB[threadIdx.x][threadIdx.y] = (n_idx_B < N && k_idx_B < K) ? B[n_idx_B * K + k_idx_B] : 0.0f;\n";
        ss << "        __syncthreads();\n\n";

        ss << "        #pragma unroll\n";
        ss << "        for (int k = 0; k < TILE_DIM; ++k) {\n";
        ss << "            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n\n";

        ss << "    if (row < M && col < N) {\n";
        ss << "        out[row * N + col] = acc;\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 3. Shared-memory Tiled Matmul with Fused Epilogues (Forward)
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

        int64_t offsetA = bufA ? bufA->arenaOffsetBytes : 0;
        int64_t offsetB = bufB ? bufB->arenaOffsetBytes : 0;
        int64_t offsetOut = bufOut ? bufOut->arenaOffsetBytes : 0;

        ss << "// Shared-memory tiled matrix multiplication kernel with fused epilogue (CS6023)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int M, int K, int N) {\n";
        ss << "    const int TILE_DIM = 16;\n";
        ss << "    __shared__ float sA[TILE_DIM][TILE_DIM];\n";
        ss << "    __shared__ float sB[TILE_DIM][TILE_DIM];\n\n";

        // Buffer pointers
        ss << "    const float* __restrict__ A = (const float*)(arena + " << offsetA << ");\n";
        ss << "    const float* __restrict__ B = (const float*)(arena + " << offsetB << ");\n";
        if (bufBias && hasBias) ss << "    const float* __restrict__ bias = (const float*)(arena + " << bufBias->arenaOffsetBytes << ");\n";
        ss << "    float* __restrict__ out = (float*)(arena + " << offsetOut << ");\n\n";

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

    // 4. Bias Backward (column-wise reduction over batch)
    if (isBiasBackwardKernel(kernel)) {
        const loopir::Buffer* bufDOut = nullptr;
        const loopir::Buffer* bufDBias = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufDBias = &b;
            else if (!bufDOut) bufDOut = &b;
        }

        int64_t offsetDOut = bufDOut ? bufDOut->arenaOffsetBytes : 0;
        int64_t offsetDBias = bufDBias ? bufDBias->arenaOffsetBytes : 0;

        ss << "// Fused column-wise bias gradient reduction kernel (CS6023)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int M, int N) {\n";
        ss << "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (col >= N) return;\n\n";
        ss << "    const float* __restrict__ dOut = (const float*)(arena + " << offsetDOut << ");\n";
        ss << "    float* __restrict__ dBias = (float*)(arena + " << offsetDBias << ");\n\n";
        ss << "    float acc = 0.0f;\n";
        ss << "    #pragma unroll 4\n";
        ss << "    for (int m = 0; m < M; ++m) {\n";
        ss << "        acc += dOut[m * N + col];\n";
        ss << "    }\n";
        ss << "    dBias[col] = acc;\n";
        ss << "}";
        return ss.str();
    }

    // 5. ReLU Backward
    if (isReluBackwardKernel(kernel)) {
        const loopir::Buffer* bufDOut = nullptr;
        const loopir::Buffer* bufAct = nullptr;
        const loopir::Buffer* bufDIn = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufDIn = &b;
            else if (!bufDOut) bufDOut = &b;
            else if (!bufAct) bufAct = &b;
        }

        int64_t offsetDOut = bufDOut ? bufDOut->arenaOffsetBytes : 0;
        int64_t offsetAct = bufAct ? bufAct->arenaOffsetBytes : 0;
        int64_t offsetDIn = bufDIn ? bufDIn->arenaOffsetBytes : 0;

        ss << "// Fused vectorized ReLU gradient kernel with PyTorch NaN safety\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, int totalElements) {\n";
        ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    int stride = blockDim.x * gridDim.x;\n\n";
        ss << "    const float* __restrict__ dOut = (const float*)(arena + " << offsetDOut << ");\n";
        ss << "    const float* __restrict__ act = (const float*)(arena + " << offsetAct << ");\n";
        ss << "    float* __restrict__ dIn = (float*)(arena + " << offsetDIn << ");\n\n";
        ss << "    for (int i = idx; i < totalElements; i += stride) {\n";
        ss << "        float h = act[i];\n";
        ss << "        dIn[i] = (h > 0.0f) ? dOut[i] : 0.0f;\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 6. Adam Optimizer Update
    if (isAdamKernel(kernel)) {
        const loopir::Buffer* bufP = nullptr;
        const loopir::Buffer* bufG = nullptr;
        const loopir::Buffer* bufM = nullptr;
        const loopir::Buffer* bufV = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput && !bufP) bufP = &b;
            else if (!bufG) bufG = &b;
            else if (!bufM) bufM = &b;
            else if (!bufV) bufV = &b;
        }
        if (!bufP && !buffers.empty()) bufP = &buffers[0];

        int64_t offsetP = bufP ? bufP->arenaOffsetBytes : 0;
        int64_t offsetG = bufG ? bufG->arenaOffsetBytes : 0;
        int64_t offsetM = bufM ? bufM->arenaOffsetBytes : 0;
        int64_t offsetV = bufV ? bufV->arenaOffsetBytes : 0;

        ss << "// Fused GPU Adam optimizer update kernel (CS6023 register reuse)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, float lr, float beta1, float beta2, float eps, float bc1, float bc2, int totalElements) {\n";
        ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    int stride = blockDim.x * gridDim.x;\n";
        ss << "    const float one_minus_b1 = 1.0f - beta1;\n";
        ss << "    const float one_minus_b2 = 1.0f - beta2;\n\n";
        ss << "    float* __restrict__ p = (float*)(arena + " << offsetP << ");\n";
        ss << "    const float* __restrict__ g = (const float*)(arena + " << offsetG << ");\n";
        ss << "    float* __restrict__ m = (float*)(arena + " << offsetM << ");\n";
        ss << "    float* __restrict__ v = (float*)(arena + " << offsetV << ");\n\n";
        ss << "    for (int i = idx; i < totalElements; i += stride) {\n";
        ss << "        float grad = g[i];\n";
        ss << "        float m_val = beta1 * m[i] + one_minus_b1 * grad;\n";
        ss << "        float v_val = beta2 * v[i] + one_minus_b2 * (grad * grad);\n";
        ss << "        m[i] = m_val;\n";
        ss << "        v[i] = v_val;\n\n";
        ss << "        float m_hat = m_val / bc1;\n";
        ss << "        float v_hat = v_val / bc2;\n";
        ss << "        p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 7. Embedding Backward (Scatter-add with atomicAdd)
    if (isEmbeddingBackwardKernel(kernel)) {
        const loopir::Buffer* bufDTable = nullptr;
        const loopir::Buffer* bufDOut = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufDTable = &b;
            else if (!bufDOut) bufDOut = &b;
        }
        if (!bufDTable && !buffers.empty()) bufDTable = &buffers[0];

        int64_t offsetTable = bufDTable ? bufDTable->arenaOffsetBytes : 0;
        int64_t offsetOut = bufDOut ? bufDOut->arenaOffsetBytes : 0;

        ss << "// Fused embedding scatter-add gradient kernel via atomicAdd (CS6023)\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, const int* __restrict__ idx, int seq_len, int d_model) {\n";
        ss << "    int i = blockIdx.y * blockDim.y + threadIdx.y;\n";
        ss << "    int d = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (i < seq_len && d < d_model) {\n";
        ss << "        int token_id = idx[i];\n";
        ss << "        float* __restrict__ dTable = (float*)(arena + " << offsetTable << ");\n";
        ss << "        const float* __restrict__ dOut = (const float*)(arena + " << offsetOut << ");\n";
        ss << "        atomicAdd(&dTable[token_id * d_model + d], dOut[i * d_model + d]);\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 8. Cross-Entropy Backward
    if (isCrossEntropyBackwardKernel(kernel)) {
        const loopir::Buffer* bufProbs = nullptr;
        const loopir::Buffer* bufDLogits = nullptr;

        for (const auto& b : buffers) {
            if (b.isOutput) bufDLogits = &b;
            else if (!bufProbs) bufProbs = &b;
        }

        int64_t offsetProbs = bufProbs ? bufProbs->arenaOffsetBytes : 0;
        int64_t offsetDLogits = bufDLogits ? bufDLogits->arenaOffsetBytes : 0;

        ss << "// Cross-entropy softmax gradient kernel: dLogits = (probs - one_hot(y)) / N\n";
        ss << "__global__ void " << name << "(char* __restrict__ arena, const int* __restrict__ targets, int batch_sz, int num_classes) {\n";
        ss << "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n";
        ss << "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    if (row < batch_sz && col < num_classes) {\n";
        ss << "        const float* __restrict__ probs = (const float*)(arena + " << offsetProbs << ");\n";
        ss << "        float* __restrict__ dLogits = (float*)(arena + " << offsetDLogits << ");\n";
        ss << "        int idx = row * num_classes + col;\n";
        ss << "        float target_val = (targets[row] == col) ? 1.0f : 0.0f;\n";
        ss << "        dLogits[idx] = (probs[idx] - target_val) / (float)batch_sz;\n";
        ss << "    }\n";
        ss << "}";
        return ss.str();
    }

    // 9. Row-wise Softmax
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

    // 10. Embedding Gather / Lookup
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

    // 11. Elementwise / Vectorized Kernel
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

    if (isMatmulTransAKernel(kernel) || isMatmulTransBKernel(kernel) || isMatmulKernel(kernel)) {
        // 16x16 2D thread block for shared-memory GEMM
        cfg.blockX = 16;
        cfg.blockY = 16;
        cfg.blockZ = 1;
        cfg.gridX = (256 + 15) / 16;
        cfg.gridY = (128 + 15) / 16;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 2 * 16 * 16 * sizeof(float); // sA + sB
    } else if (isBiasBackwardKernel(kernel)) {
        // 1D block across columns
        cfg.blockX = 256;
        cfg.blockY = 1;
        cfg.blockZ = 1;
        cfg.gridX = (256 + 255) / 256;
        cfg.gridY = 1;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 0;
    } else if (isSoftmaxKernel(kernel)) {
        // One block per row, 256 threads per row
        cfg.blockX = 256;
        cfg.blockY = 1;
        cfg.blockZ = 1;
        cfg.gridX = 128; // num rows
        cfg.gridY = 1;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 64 * sizeof(float);
    } else if (isEmbeddingBackwardKernel(kernel) || isEmbeddingKernel(kernel) || isCrossEntropyBackwardKernel(kernel)) {
        // 2D grid for sequence / embedding dimensions
        cfg.blockX = 16;
        cfg.blockY = 16;
        cfg.blockZ = 1;
        cfg.gridX = (256 + 15) / 16;
        cfg.gridY = (128 + 15) / 16;
        cfg.gridZ = 1;
        cfg.sharedMemBytes = 0;
    } else {
        // 1D grid-stride loop for relu_backward, adam_update, and elementwise
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
