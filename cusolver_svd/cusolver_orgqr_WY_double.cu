#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <kernelOther.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cusolver_utils.h"

#define NUM_WARPUP 2
#define NUM_REPEAT 5

void init_identity_matrix(std::vector<double> &A, size_t m, size_t n,
                          size_t lda) {
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{0U}; i < m; ++i) {
            if (i == j)
                A[i + j * lda] = 1;
            else
                A[i + j * lda] = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 1024, n = 32;

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    const int minmn = m >= n ? m : n;
    const int lda = m;  // lda >= m
    const int ldy = m;
    const int ldr = n;
    const int ldq = m;
    int lwork = 0; /* size of workspace */
    int ldworkw = 0;

    std::vector<double> A(m * n, 0);
    std::vector<double> Q(m * m, 0);
    std::vector<double> TUA_from_gpu(minmn, 0);
    std::vector<double> Y_from_gpu(m * n, 0);
    std::vector<double> R_from_gpu(m * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    init_identity_matrix(Q, m, m, ldq);

    int info_gpu = 0; /* host copy of error info */
    int info_gpu1 = 0; /* host copy of error info */

    double *d_A = nullptr;
    double *d_Q = nullptr;
    double *d_Y = nullptr;
    double *d_R = nullptr;
    double *d_TAU = nullptr;
    double *d_work = nullptr;
    double *d_work_w = nullptr;
    int *devInfo = nullptr;
    int *devInfo1 = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(double) * m * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Y), sizeof(double) * m * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(double) * n * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_TAU), sizeof(double) * minmn));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo1), sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of BRD */
    CUSOLVER_CHECK(
        cusolverDnDgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &lwork));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
    CUSOLVER_CHECK(
        cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_A, lda, d_TAU, &ldworkw));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work_w), sizeof(double) * ldworkw));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), sizeof(double) * Q.size(),
                          cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work,
                                    lwork, devInfo));
    launchKernel_moveU({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                       d_A, lda, d_R, ldr);
    launchKernel_copyLower({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                           d_A, lda, d_Y, ldy);
    CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w,
                                    ldworkw, devInfo));
                                
    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for(int i{0}; i < NUM_WARPUP; ++i)
    {
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(),
                            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), sizeof(double) * Q.size(),
                            cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work,
                                        lwork, devInfo));
        launchKernel_moveU({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                        d_A, lda, d_R, ldr);
        launchKernel_copyLower({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                            d_A, lda, d_Y, ldy);
        CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w,
                                        ldworkw, devInfo1));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int i{0}; i < NUM_REPEAT; ++i)
    {
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(),
                            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), sizeof(double) * Q.size(),
                            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work,
                                        lwork, devInfo));
        launchKernel_moveU({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                        d_A, lda, d_R, ldr);
        launchKernel_copyLower({(m + 31) / 32, (n + 31) / 32, 1}, {32, 32, 1}, m, n,
                            d_A, lda, d_Y, ldy);                                       
        CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w,
                                        ldworkw, devInfo1));

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu1, devInfo1, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::printf("after geqrf: info_gpu = %d\n", info_gpu);
    // if (0 == info_gpu) {
    //     std::printf("geqrf converges \n");
    // } else if (0 > info_gpu) {
    //     std::printf("%d-th parameter is wrong \n", -info_gpu);
    //     exit(1);
    // } else {
    //     std::printf("WARNING: info = %d : geqrf does not converge \n",
    //     info_gpu);
    // }

    // std::printf("after orgqr: info_gpu1 = %d\n", info_gpu1);
    // if (0 == info_gpu1) {
    //     std::printf("orgqr converges \n");
    // } else if (0 > info_gpu1) {
    //     std::printf("%d-th parameter is wrong \n", -info_gpu1);
    //     exit(1);
    // } else {
    //     std::printf("WARNING: info = %d : orgqr does not converge \n",
    //     info_gpu1);
    // }

    std::cout << "Cusolver QRF + ORGQR (double) Latency: " << time << " ms"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_TAU));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_work_w));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
