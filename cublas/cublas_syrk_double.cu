#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 1
#define NUM_REPEAT 20

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int n = 32768, k = 32768;

    if (argc >= 3) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    std::vector<double> A(n * k, 0);
    std::vector<double> C(n * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    double *d_A = nullptr;
    double *d_C = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    int lda = n, ldc = n;

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * k));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * ldc * n));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(),
                          cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasDsyrk(
        cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, d_A, lda,
        &zero, d_C, ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
	    CUBLAS_CHECK(cublasDsyrk(
		cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, d_A, lda,
		&zero, d_C, ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaEventRecord(start, stream));

    CUBLAS_CHECK(cublasDsyrk(
        cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, d_A, lda,
        &zero, d_C, ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas dsyrk] " << "n: " << n << ", k: " << k << ", "
              << "latency: " << time << " ms, "
              << "Effective TFLOPS: " << (long)n * n * k / time / 1e9 << " TFLOPS, "
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
