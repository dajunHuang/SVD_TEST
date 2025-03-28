#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "utils.h"

void generateUniqueRandomArray(int* arr, int nb, int max) {
    for (int i = 0; i < nb; ++i) {
        arr[i] = rand() % max; // 生成 0 到 max-1 之间的随机数
    }
}

template <typename T>
__global__ void swap_kernel(int n, T *x, int incx, T *y, int incy) {}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 16384, n = 16384, nb = 1024;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        nb = atoi(argv[3]);
    }

    double *d_A_ori = nullptr;
    double *d_A_cublas = nullptr;
    double *d_A_custom = nullptr;
    int *hdevIpiv = (int *)malloc(sizeof(int) * nb);

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    int lda = m;

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_ori), sizeof(double) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_cublas), sizeof(double) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_custom), sizeof(double) * lda * n));

    generateUniformMatrixDouble(d_A_ori, lda, n);
    generateUniqueRandomArray(hdevIpiv, nb, m);
    // for(int i = 0; i < nb; ++i) {
    //     printf("%d: %d\n", i, hdevIpiv[i]);
    // }


    // printf("d_A_ori:\n");
    // print_device_matrix(d_A_ori, lda, 16, 16);

    CUDA_CHECK(cudaMemcpy(d_A_cublas, d_A_ori, lda * n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_custom, d_A_ori, lda * n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUBLAS_CHECK(cublasDswap(cublasH, n, d_A_cublas + r1, lda, d_A_cublas + r2, lda));

    // printf("d_A_cublas:\n");
    // print_device_matrix(d_A_cublas, lda, 16, 16);

    cudaEvent_t start, stop;
    float time = 0;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for(int i = 0; i < nb; ++i) {
        cublasDswap(cublasH, n, d_A_cublas + i, lda, d_A_cublas + hdevIpiv[i], lda);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[cublas dswap] " << "m: " << m << ", n: " << n << ", " << "nb: " << nb << ", "
              << "latency: " << time << " ms" << std::endl;


    /* free resources */
    CUDA_CHECK(cudaFree(d_A_ori));
    CUDA_CHECK(cudaFree(d_A_cublas));
    CUDA_CHECK(cudaFree(d_A_custom));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
