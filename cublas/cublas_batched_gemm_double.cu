#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_REPEAT 1

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int n = 32768, k = 32768, nb = 512;

    if (argc >= 3) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        nb = atoi(argv[3]);
    }

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

    generateUniformMatrixDouble(d_A, lda, k);

    cudaEvent_t start, stop;
    float time0 = 0, time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));

        for (int j{0}; j < n / nb; ++j) {
            cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k, &one,
                         d_A + j * nb, lda, d_A + j * nb,
                         lda, &zero, d_C + j * nb + j * nb * ldc,
                         ldc);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time0 += temp_time;
    }
    time0 /= NUM_REPEAT;

    std::cout << "[cublas        gemm] " << "n: " << n << ", k: " << k
              << ", "
              << "latency: " << time0 << " ms, "
              << (long)n / nb * 2 * nb * nb * k / time0 / 1e9
              << " TFLOPS" << std::endl;

    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));

        cublasDgemmStridedBatched(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k, &one, d_A,
            lda, nb, d_A, lda, nb, &zero, d_C, ldc,
            nb + nb * ldc, n / nb);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas bached gemm] " << "n: " << n << ", k: " << k
              << ", "
              << "latency: " << time1 << " ms, "
              << (long)n / nb * 2 * nb * nb * k / time1 / 1e9
              << " TFLOPS" << std::endl;


    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
