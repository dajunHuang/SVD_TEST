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
    // std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
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

    int m = 1024, n = 32;

    if(argc >= 3)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    const int minmn = m >= n ? m : n;
    const int lda = m;  // lda >= m

    std::vector<double> A(m * n, 0);
    std::vector<double> A_from_gpu(m * n, 0);
    std::vector<double> TAU_from_gpu(minmn, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    std::generate(A.begin(), A.end(), rand);

    int info_gpu = 0;                                  /* host copy of error info */

    double *d_A = nullptr;
    double *d_TAU = nullptr;
    int *devInfo = nullptr;

    int lwork = 0; /* size of workspace */
    double *d_work = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_TAU), sizeof(double) * minmn));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of BRD */
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &lwork));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice);
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for(int i{0}; i < NUM_WARPUP; ++i)
    {
        cudaMemcpy(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice);
        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int i{0}; i < NUM_REPEAT; ++i)
    {
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_TAU, d_work, lwork, devInfo));

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(A_from_gpu.data(), d_A, sizeof(double) * A_from_gpu.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(TAU_from_gpu.data(), d_TAU, sizeof(double) * TAU_from_gpu.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

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

    std::cout << "Cusolver QRF (double) Latency: " << time << " ms" << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_TAU));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
