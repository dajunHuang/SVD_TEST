#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>

#include "utils.h"

#define NUM_WARPUP 5
#define NUM_REPEAT 10

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 32768, n = 32768, k = 32768;
    int nm = 1024, nn = 1024;

    if (argc >= 6) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        nm = atoi(argv[4]);
        nn = atoi(argv[5]);
    }

    assert(m % nm == 0 && n % nn == 0);

    float *d_A = nullptr;
    float *d_B = nullptr;
    // float *d_C1 = nullptr;
    float *d_C2 = nullptr;

    float one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    int lda = m, ldb = k, ldc = m;

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * lda * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * ldb * n));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C1), sizeof(float) * ldc * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C2), sizeof(float) * ldc * n));

    generateUniformMatrix(d_A, lda, k);
    generateUniformMatrix(d_B, ldb, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // for (int i{0}; i < NUM_WARPUP; ++i) {
    //     CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, d_A, lda, d_B,
    //                              ldb, &zero, d_C1,
    //                              ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP
    // }
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // for (int i{0}; i < NUM_REPEAT; ++i) {
    //     CUDA_CHECK(cudaStreamSynchronize(stream));
    //     CUDA_CHECK(cudaEventRecord(start, stream));

    //     CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, d_A, lda, d_B,
    //                              ldb, &zero, d_C1,
    //                              ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP

    //     CUDA_CHECK(cudaStreamSynchronize(stream));
    //     CUDA_CHECK(cudaEventRecord(stop, stream));
    //     CUDA_CHECK(cudaEventSynchronize(stop));
    //     CUDA_CHECK_LAST_ERROR();
    //     CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    // }
    // time /= NUM_REPEAT;

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::cout << "[cublas sgemm] " << "m: " << m << ", n: " << n << ", k: " << k << ", "
    //           << "latency: " << time << " ms, "
    //           << "Effective TFLOPS: " << 2.0 * m * n * k / time / 1e9 << " TFLOPS, " << std::endl;

    int loop_m = (m + nm - 1) / nm;
    int loop_n = (n + nn - 1) / nn;
    int loop_num = loop_m * loop_n;

    std::vector<float *> h_Aarray(loop_num);
    std::vector<float *> h_Barray(loop_num);
    std::vector<float *> h_Carray(loop_num);
    for (int i = 0; i < loop_m; ++i) {
        for (int j = 0; j < loop_n; ++j) {
            h_Aarray[i * loop_n + j] = d_A + i * nm;
            h_Barray[i * loop_n + j] = d_B + j * nn * ldc;
            h_Carray[i * loop_n + j] = d_C2 + i * nm + j * nn * ldc;
        }
    }

    float **d_Aarray, **d_Barray, **d_Carray;
    CUDA_CHECK(cudaMalloc(&d_Aarray, sizeof(float *) * loop_num));
    CUDA_CHECK(cudaMalloc(&d_Barray, sizeof(float *) * loop_num));
    CUDA_CHECK(cudaMalloc(&d_Carray, sizeof(float *) * loop_num));
    CUDA_CHECK(
        cudaMemcpy(d_Aarray, h_Aarray.data(), sizeof(float *) * loop_num, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_Barray, h_Barray.data(), sizeof(float *) * loop_num, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_Carray, h_Carray.data(), sizeof(float *) * loop_num, cudaMemcpyHostToDevice));

    time = 0, temp_time = 0;
    for (int i{0}; i < NUM_WARPUP; ++i) {
        CUBLAS_CHECK(cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, nm, nn, k, &one,
                                        d_Aarray, lda, d_Barray, ldb, &zero, d_Carray, ldc,
                                        loop_num));  // CUBLAS_GEMM_ALGO0_TENSOR_OP
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUBLAS_CHECK(cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, nm, nn, k, &one,
                                        d_Aarray, lda, d_Barray, ldb, &zero, d_Carray, ldc,
                                        loop_num));  // CUBLAS_GEMM_ALGO0_TENSOR_OP

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas batched sgemm] " << "m: " << m << ", n: " << n << ", k: " << k << ", "
              << "nm: " << nm << ", nn: " << nn << ", "
              << "latency: " << time << " ms, "
              << "Effective TFLOPS: " << 2.0 * m * n * k / time / 1e9 << " TFLOPS, " << std::endl;

    // thrust::device_ptr<float> thrust_d_C1(d_C1);
    // thrust::device_ptr<float> thrust_d_C2(d_C2);
    // bool are_equal = thrust::equal(thrust_d_C1, thrust_d_C1 + loop_num, thrust_d_C2);
    // if (are_equal) {
    //     std::cout << "C1 C2 equal" << std::endl;
    // } else {
    //     std::cout << "C1 C2 not equal" << std::endl;
    // }

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_C2));
    CUDA_CHECK(cudaFree(d_Aarray));
    CUDA_CHECK(cudaFree(d_Barray));
    CUDA_CHECK(cudaFree(d_Carray));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
