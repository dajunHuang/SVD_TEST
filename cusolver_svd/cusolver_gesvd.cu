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

#define NUM_WARPUP 2
#define NUM_REPEAT 5

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
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 0, n = 0;

    if(argc < 3)
    {
        m = 1024;
        n = 1024;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    const int lda = m;  // lda >= m
    const int ldu = m;  // ldu >= m
    const int ldvt = n; // ldvt >= n if jobu = 'A'

    std::vector<double> A(m * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    std::vector<double> U(ldu * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
    std::vector<double> VT(ldvt * n, 0); /* n-by-n unitary matrix, right singular vectors */
    std::vector<double> S(n, 0);        /* numerical singular value */
    int info_gpu = 0;                                  /* host copy of error info */

    double *d_A = nullptr;
    double *d_S = nullptr;  /* singular values */
    double *d_U = nullptr;  /* left singular vectors */
    double *d_VT = nullptr; /* right singular vectors */

    int *devInfo = nullptr;

    int lwork = 0; /* size of workspace */
    double *d_work = nullptr;
    double *d_rwork = nullptr;

    // std::printf("A = (matlab base-1)\n");
    // print_matrix(m, n, A.data(), lda);
    // std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: compute SVD */
    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n rows of VT

    cudaEvent_t start, stop;
    float time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for(int i{0}; i < NUM_WARPUP; ++i)
    {
        CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda,
                                        d_S, d_U, ldu, d_VT, ldvt,
                                        d_work, lwork, d_rwork, devInfo));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for(int i{0}; i < NUM_REPEAT; ++i)
    {
        CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda,
                                        d_S, d_U, ldu, d_VT, ldvt,
                                        d_work, lwork, d_rwork, devInfo));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    time /= NUM_REPEAT;

    CUDA_CHECK(
        cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::printf("after gesvd: info_gpu = %d\n", info_gpu);
    // if (0 == info_gpu) {
    //     std::printf("gesvd converges \n");
    // } else if (0 > info_gpu) {
    //     std::printf("%d-th parameter is wrong \n", -info_gpu);
    //     exit(1);
    // } else {
    //     std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
    // }

    // std::printf("S = singular values (matlab base-1)\n");
    // print_matrix(n, 1, S.data(), n);
    // std::printf("=====\n");

    // std::printf("U = left singular vectors (matlab base-1)\n");
    // print_matrix(m, m, U.data(), ldu);
    // std::printf("=====\n");

    // std::printf("VT = right singular vectors (matlab base-1)\n");
    // print_matrix(n, n, VT.data(), ldvt);
    // std::printf("=====\n");

    std::cout << "Cusolver SVD (Double) Latency: " << time << " ms" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
