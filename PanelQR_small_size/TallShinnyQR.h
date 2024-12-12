#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int block_size, int m, int n,
                    T *A, int lda, T *R, int ldr, T *work, int lwork) {
    int reduction_time = ceil((log(m) - log(n)) / (log(block_size) - log(n)));
    // printf("size %d, reduction_time: %d\n", m, reduction_time);

    assert(block_size % n == 0);
    assert((m % block_size) % n == 0);
    assert(reduction_time < 16);
    assert(n <= 128);
    int share_memory_size = reduction_time * block_size * n * sizeof(T);

    CUDA_CHECK(cudaFuncSetAttribute(householder_kernel<T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));

    dim3 block_dim(32, 32);  // if change block_dim, also change acc_per_thread
                             // and q_per_thread mannually in kernelQR.h
    int block_num = (m + block_size - 1) / block_size;
    householder_kernel<T><<<block_num, block_dim, share_memory_size>>>(
        block_size, m, n, A, lda, R, ldr, work, lwork);
}

template void tsqr<double>(cublasHandle_t cublas_handle,
                                     int block_size, int m, int n, double *A,
                                     int lda, double *R, int ldr, double *work,
                                     int lwork);
