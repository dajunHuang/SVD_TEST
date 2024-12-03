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
void test_hou_tsqr_panel(int m, int n) {
    // cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int lda = m;
    const int ldr = n;

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

    const int blockNum = (m + 128 - 1) / 128;
    const int ldwork{32 * blockNum};

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work),
                          sizeof(T) * ldwork * 32));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * A.size(),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    printf("hou_tsqr_panel\n");
    hou_tsqr_panel<T, 128, 32>(cublasH, m, n, d_A, lda, d_R, ldr, d_work,
                               ldwork);
    printDeviceMatrixV2(d_A, lda, 16, 16);
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
                          cudaMemcpyHostToDevice));
    printf("\nhou_tsqr_panel_ori\n");
    hou_tsqr_panel_ori<T, 128, 32>(cublasH, m, n, d_A, lda, d_R, ldr,
                                   d_work_ori);
    printDeviceMatrixV2(d_A, lda, 16, 16);
    CUDA_CHECK_LAST_ERROR();

    // cudaEvent_t start, stop;
    // float time = 0, temp_time = 0;

    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));
    // for (int i{0}; i < NUM_WARPUP; ++i) {
    //     cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
    //     cudaMemcpyHostToDevice); hou_tsqr_panel<T, 128, 32>(cublasH, m, n,
    //     d_A, lda, d_R, ldr, d_work,
    //                             ldwork);
    // }
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // for (int i{0}; i < NUM_REPEAT; ++i) {
    //     cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(),
    //     cudaMemcpyHostToDevice); CUDA_CHECK(cudaEventRecord(start, stream));

    //     hou_tsqr_panel<T, 128, 32>(cublasH, m, n, d_A, lda, d_R, ldr, d_work,
    //                             ldwork);

    //     CUDA_CHECK(cudaEventRecord(stop, stream));
    //     CUDA_CHECK(cudaEventSynchronize(stop));
    //     CUDA_CHECK_LAST_ERROR();
    //     CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    // }
    // time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A,
                               sizeof(T) * A_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(R_from_gpu.data(), d_R,
                               sizeof(T) * R_from_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::cout << "hou_tsqr_panel Latency: " << time << " ms" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_work));

    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}

template void test_hou_tsqr_panel<float>(int m, int n);
// template void test_hou_tsqr_panel<double>(int m, int n);

int main(int argc, char *argv[]) {
    int m = 2048, n = 32;
    int dataType = 1;

    // print_device_info();

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        dataType = atoi(argv[3]);
    }

    if (0 == dataType) {
        // test_hou_tsqr_panel<half>(m, n);
    } else if (1 == dataType) {
        test_hou_tsqr_panel<float>(m, n);
    } else if (2 == dataType) {
        // test_hou_tsqr_panel<double>(m, n);
    }

    return 0;
}
