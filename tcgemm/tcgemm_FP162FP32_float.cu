/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

#include <algorithm>
#include <iostream>

#define NUM_WARPUP 5
#define NUM_REPEAT 10

template <typename T>
void random_initialize_matrix(T* A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U)
{
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t j{0U}; j < n; ++j)
    {
        for (size_t i{0U}; i < m; ++i)
        {
            A[i + j * lda] = static_cast<T>(rand());
        }
    }
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int n = 8192, k = 16;

    if(argc >= 3)
    {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    const int ldc = n;
    const int lda = n;
    const int ldb = n;

    std::vector<__half> A(n * k, 0);
    std::vector<__half> B(n * k, 0);
    std::vector<float> C(n * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);
    std::generate(B.begin(), B.end(), rand);

    // print_matrix(n, k, A.data(), lda);

    __half *d_A = nullptr;
    __half *d_B = nullptr;
    float *d_C = nullptr;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(__half) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(__half) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * C.size()));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(__half) * A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(__half) * B.size(), cudaMemcpyHostToDevice));

    float alpha = 1, beta = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for(int i{0}; i < NUM_WARPUP; ++i)
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * C.size()));
        CUBLAS_CHECK(cublasGemmEx(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_A, CUDA_R_16F, lda,
                d_A, CUDA_R_16F, lda, &beta, d_C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUBLAS_CHECK(cublasGemmEx(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_B, CUDA_R_16F, ldb,
                d_B, CUDA_R_16F, ldb, &beta, d_C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float time = 0, temp_time = 0;
    for(int i{0}; i < NUM_REPEAT; ++i)
    {
        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * C.size()));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUBLAS_CHECK(cublasGemmEx(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_A, CUDA_R_16F, lda,
                d_A, CUDA_R_16F, lda, &beta, d_C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CUBLAS_CHECK(cublasGemmEx(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_B, CUDA_R_16F, ldb,
                d_B, CUDA_R_16F, ldb, &beta, d_C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();    
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(
        cudaMemcpyAsync(C.data(), d_C, sizeof(float) * C.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float effective_tflops = (4.0 * n * k * n) / (time * 1e-3) / 1e12;
    std::cout << "TCGEMM (float) Latency: " << time << " ms" << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
