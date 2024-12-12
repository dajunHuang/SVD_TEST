#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "TallShinnyQR.h"
#include "TallShinnyQR_ori.h"

#define NUM_WARPUP 20
#define NUM_REPEAT 50

template <typename T>
void test_hou_tsqr_panel(int block_size, int m, int n) {
    // cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int lda = m;
    const int ldr = n;
    double kOne = 1, kZero = 0;

    std::vector<T> A(m * n, 0);
    std::vector<T> A_from_gpu(m * n, 0);
    std::vector<T> R_from_gpu(n * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<T> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    T *d_A = nullptr;
    T *d_R = nullptr;
    T *d_work = nullptr;
    T *d_work_ori = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    // CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(T) * n * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work_ori), sizeof(T) * m * m));

    const int ldwork = m;

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * m * n));
    // # origianl
    // CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
    //                       cudaMemcpyHostToDevice));
    // printf("\nhou_tsqr_panel_ori\n");
    // hou_tsqr_panel_ori<T, 128, 32>(cublasH, m, n, d_A, lda, d_R, ldr,
    //                                d_work_ori);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // printDeviceMatrixV2(d_R, ldr, 32, 32);
    // printf("Q\n");
    // printDeviceMatrixV2(d_A, lda, m < 169 ? m : 169, 32);

    // # new
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    // printf("A\n");
    // printDeviceMatrixV2(d_A, lda, m < 169 ? m : 169, 32);
    // printf("hou_tsqr_panel\n");
    hou_tsqr_panel<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr, d_work,
                      ldwork);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST_ERROR();
    // printf("R\n");
    // printDeviceMatrixV2(d_R, ldr, 32, 32);
    // printf("Q\n");
    // printDeviceMatrixV2(d_A, lda, m < 32 ? m : 32, n < 32 ? n : 32);

    // T *d_Q = nullptr;
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(T) * n *
    // n)); cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &kOne, d_A,
    //             lda, d_A, lda, &kZero, d_Q, n);
    // printDeviceMatrixV2(d_Q, n, n, n);

    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &kOne, d_A, lda,
                d_R, ldr, &kZero, d_A, lda);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!all_close(A_from_gpu.data(), A.data(), m, n, lda, 1.0e-4, 1.0e-5)) {
        std::cout << "Error: hou_tsqr_panel" << std::endl;
        exit(-1);
    }

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("warmup %d\n", i);
        hou_tsqr_panel<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr, d_work,
                          ldwork);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("repeat %d\n", i);
        CUDA_CHECK(cudaEventRecord(start, stream));

        hou_tsqr_panel<T>(cublasH, block_size, m, n, d_A, lda, d_R, ldr, d_work,
                          ldwork);

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(R_from_gpu.data(), d_R,
                               sizeof(T) * R_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "hou_tsqr_panel Latency: " << time << " ms" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_work_ori));

    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}

// template void test_hou_tsqr_panel<float>(int m, int n);
template void test_hou_tsqr_panel<double>(int block_size, int m, int n);

int main(int argc, char *argv[]) {
    int m = 13824, n = 32;
    int block_size = 128;
    int dataType = 2;

    print_device_info();

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        dataType = atoi(argv[3]);
    }

    if (0 == dataType) {
        // test_hou_tsqr_panel<half>(m, n);
    } else if (1 == dataType) {
        // test_hou_tsqr_panel<float>(m, n);
    } else if (2 == dataType) {
        test_hou_tsqr_panel<double>(block_size, m, n);
    }

    return 0;
}
