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
    // const int ldq = m;
    int lwork = 0; /* size of workspace */
    int ldworkw = 0;

    int info_gpu = 0;  /* host copy of error info */
    int info_gpu1 = 0; /* host copy of error info */

    float *d_A = nullptr;
    // float *d_Q = nullptr;
    float *d_Y = nullptr;
    float *d_R = nullptr;
    float *d_TAU = nullptr;
    float *d_work = nullptr;
    float *d_work_w = nullptr;
    int *devInfo = nullptr;
    int *devInfo1 = nullptr;

    dim3 blockDim{32, 32, 1};
    dim3 gridDim{static_cast<unsigned int>((m + 31) / 32), static_cast<unsigned int>((n + 31) / 32),
                 1};

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * n));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(float) * m * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Y), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * n * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_TAU), sizeof(float) * minmn));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo1), sizeof(int)));

    // CUDA_CHECK(
    //     cudaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of BRD */
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
    CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolverH, m, n, n, d_A, lda, d_TAU, &ldworkw));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work_w), sizeof(float) * ldworkw));

    generateUniformMatrix(d_A, lda, n);
    // launchKernel_Identity(gridDim, blockDim, m, n, d_Q, ldq);
    CUSOLVER_CHECK(cusolverDnSgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));
    launchKernel_moveU(gridDim, blockDim, m, n, d_A, lda, d_R, ldr);
    launchKernel_copyLower(gridDim, blockDim, m, n, d_A, lda, d_Y, ldy);
    CUSOLVER_CHECK(
        cusolverDnSorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w, ldworkw, devInfo));

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        generateUniformMatrix(d_A, lda, n);
        // launchKernel_Identity(gridDim, blockDim, m, n, d_Q, ldq);
        CUSOLVER_CHECK(cusolverDnSgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));
        launchKernel_moveU(gridDim, blockDim, m, n, d_A, lda, d_R, ldr);
        launchKernel_copyLower(gridDim, blockDim, m, n, d_A, lda, d_Y, ldy);
        CUSOLVER_CHECK(
            cusolverDnSorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w, ldworkw, devInfo1));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        generateUniformMatrix(d_A, lda, n);
        // launchKernel_Identity(gridDim, blockDim, m, n, d_Q, ldq);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUSOLVER_CHECK(cusolverDnSgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));
        launchKernel_moveU(gridDim, blockDim, m, n, d_A, lda, d_R, ldr);
        launchKernel_copyLower(gridDim, blockDim, m, n, d_A, lda, d_Y, ldy);
        CUSOLVER_CHECK(
            cusolverDnSorgqr(cusolverH, m, n, n, d_A, lda, d_TAU, d_work_w, ldworkw, devInfo1));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu1, devInfo1, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after geqrf: info_gpu = %d\n", info_gpu);
    if (0 == info_gpu) {
        std::printf("geqrf converges \n");
    } else if (0 > info_gpu) {
        std::printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else {
        std::printf("WARNING: info = %d : geqrf does not converge \n",
        info_gpu);
    }

    std::printf("after orgqr: info_gpu1 = %d\n", info_gpu1);
    if (0 == info_gpu1) {
        std::printf("orgqr converges \n");
    } else if (0 > info_gpu1) {
        std::printf("%d-th parameter is wrong \n", -info_gpu1);
        exit(1);
    } else {
        std::printf("WARNING: info = %d : orgqr does not converge \n",
        info_gpu1);
    }

    std::cout << "m: " << m << ", n: " << n << ", Cusolver QRF + ORGQR (float) Latency: " << time
              << " ms, " << (float(2) * m * n * n - float(2) / 3 * n * n * n) / time / 1e9
              << " TFLOPS" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_Q));
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
