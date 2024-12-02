#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mma.h>
#include "cusolver_utils.h"

#define NUM_WARPUP 5
#define NUM_REPEAT 10

template <typename T>
void random_initialize_matrix(T *A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U) {
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{0U}; i < m; ++i) {
            A[i + j * lda] = static_cast<T>(rand());
        }
    }
}

__global__ void tcgemm(__half *d_A, int lda, __half *d_B,
                       int ldb, __half *d_C, int ldc) {
    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, __half,
                           nvcuda::wmma::col_major>
        a_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, __half,
                           nvcuda::wmma::col_major>
        b_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, __half>
        acc_frags;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, __half> c_frag;

    nvcuda::wmma::fill_fragment(acc_frags, static_cast<float>(0));

    nvcuda::wmma::load_matrix_sync(a_frags, d_A, lda);
    nvcuda::wmma::load_matrix_sync(b_frags, d_B, ldb);
    nvcuda::wmma::mma_sync(acc_frags, a_frags, b_frags, acc_frags);

    nvcuda::wmma::load_matrix_sync(c_frag, d_C, ldc, nvcuda::wmma::mem_col_major);
    for (size_t i{0}; i < c_frag.num_elements; ++i)
    {
        c_frag.x[i] = acc_frags.x[i];
    }
    nvcuda::wmma::store_matrix_sync(d_C, c_frag, ldc, nvcuda::wmma::mem_col_major);
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 32, n = 8, k = 16;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    std::vector<__half> A(m * k, 0);
    std::vector<__half> B(k * n, 0);
    std::vector<__half> C(m * n, 0);

    std::default_random_engine eng(0U);
    // std::uniform_int_distribution<int> dis(0, 5);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return __float2half_rn(dis(eng)); };
    std::generate(A.begin(), A.end(), rand);
    std::generate(B.begin(), B.end(), rand);

    // print_matrix(n, k, A.data(), lda);

    __half *d_A = nullptr;
    __half *d_B = nullptr;
    __half *d_C = nullptr;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(__half) * A.size()));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(__half) * B.size()));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(__half) * C.size()));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(__half) * A.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(__half) * B.size(),
                          cudaMemcpyHostToDevice));

    tcgemm<<<1, 32>>>(d_A, lda, d_B, ldb, d_C, ldc);

    CUDA_CHECK_LAST_ERROR();

    // float alpha = 1, beta = 1;

    // cudaEvent_t start, stop;
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));
    // for(int i{0}; i < NUM_WARPUP; ++i)
    // {
    //     CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * C.size()));
    //     CUBLAS_CHECK(cublasGemmEx(
    //         cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_A,
    //         CUDA_R_16F, lda,
    //             d_A, CUDA_R_16F, lda, &beta, d_C, CUDA_R_32F, ldc,
    //             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    //     CUBLAS_CHECK(cublasGemmEx(
    //         cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_B,
    //         CUDA_R_16F, ldb,
    //             d_B, CUDA_R_16F, ldb, &beta, d_C, CUDA_R_32F, ldc,
    //             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    // }
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // float time = 0, temp_time = 0;
    // for(int i{0}; i < NUM_REPEAT; ++i)
    // {
    //     CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * C.size()));
    //     CUDA_CHECK(cudaEventRecord(start, stream));

    //     CUBLAS_CHECK(cublasGemmEx(
    //         cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_A,
    //         CUDA_R_16F, lda,
    //             d_A, CUDA_R_16F, lda, &beta, d_C, CUDA_R_32F, ldc,
    //             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    //     CUBLAS_CHECK(cublasGemmEx(
    //         cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_B,
    //         CUDA_R_16F, ldb,
    //             d_B, CUDA_R_16F, ldb, &beta, d_C, CUDA_R_32F, ldc,
    //             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    //     CUDA_CHECK(cudaEventRecord(stop, stream));
    //     CUDA_CHECK(cudaEventSynchronize(stop));
    //     CUDA_CHECK_LAST_ERROR();
    //     CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    //     time += temp_time;
    // }
    // time /= NUM_REPEAT;

    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(__half) * C.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // float effective_tflops = (4.0 * n * k * n) / (time * 1e-3) / 1e12;
    // std::cout << "TCGEMM (float) Latency: " << time << " ms" << std::endl;
    // std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS" <<
    // std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
