#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>

void fma_worker(long long iterations, double& gflops) {
    __m256 a0 = _mm256_set1_ps(1.0001f);
    __m256 a1 = _mm256_set1_ps(1.0002f);
    __m256 a2 = _mm256_set1_ps(1.0003f);
    __m256 a3 = _mm256_set1_ps(1.0004f);
    __m256 a4 = _mm256_set1_ps(1.0005f);
    __m256 a5 = _mm256_set1_ps(1.0006f);
    __m256 a6 = _mm256_set1_ps(1.0007f);
    __m256 a7 = _mm256_set1_ps(1.0008f);
    __m256 a8 = _mm256_set1_ps(1.0009f);
    __m256 a9 = _mm256_set1_ps(1.0010f);
    __m256 b0 = _mm256_set1_ps(1.00001f);
    __m256 c0 = _mm256_set1_ps(0.00001f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < iterations; ++i) {
        a0 = _mm256_fmadd_ps(a0, b0, c0);
        a1 = _mm256_fmadd_ps(a1, b0, c0);
        a2 = _mm256_fmadd_ps(a2, b0, c0);
        a3 = _mm256_fmadd_ps(a3, b0, c0);
        a4 = _mm256_fmadd_ps(a4, b0, c0);
        a5 = _mm256_fmadd_ps(a5, b0, c0);
        a6 = _mm256_fmadd_ps(a6, b0, c0);
        a7 = _mm256_fmadd_ps(a7, b0, c0);
        a8 = _mm256_fmadd_ps(a8, b0, c0);
        a9 = _mm256_fmadd_ps(a9, b0, c0);
    }
    // Prevent dead-code elimination
    __m256 sum = _mm256_add_ps(a0, a1);
    sum = _mm256_add_ps(sum, a2);
    sum = _mm256_add_ps(sum, a3);
    sum = _mm256_add_ps(sum, a4);
    sum = _mm256_add_ps(sum, a5);
    sum = _mm256_add_ps(sum, a6);
    sum = _mm256_add_ps(sum, a7);
    sum = _mm256_add_ps(sum, a8);
    sum = _mm256_add_ps(sum, a9);
    float buf[8];
    _mm256_storeu_ps(buf, sum);
    asm volatile("" : : "m"(buf[0]) : "memory");

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    // 10 FMAs per loop * 16 FLOPs per AVX2 256-bit FMA (8 multiplies + 8 adds)
    double flops = (double)iterations * 10.0 * 16.0;
    gflops = (flops / 1e9) / sec;
}

int main(int argc, char** argv) {
    int num_threads = (argc > 1) ? std::atoi(argv[1]) : 1;
    long long iters = 100000000LL;
    std::vector<std::thread> threads;
    std::vector<double> results(num_threads, 0.0);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(fma_worker, iters, std::ref(results[i]));
    }
    for (auto& t : threads) {
        t.join();
    }
    double total_gflops = 0.0;
    for (double g : results) {
        total_gflops += g;
    }
    std::cout << total_gflops << std::endl;
    return 0;
}
