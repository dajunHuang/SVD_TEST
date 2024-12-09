#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T, int BLCOK_SIZE_X>
void hou_tsqr_panel(cublasHandle_t cublas_handle, int m, int n, T *A, int lda,
                    T *R, int ldr, T *work, int lwork) {
    int reduction_time = ceil((log(m) - log(n)) / (log(BLCOK_SIZE_X) - log(n)));
    // printf("size %d, reduction_time: %d\n", m, reduction_time);
    int share_memory_size = reduction_time * BLCOK_SIZE_X * n * sizeof(T);

    CUDA_CHECK(cudaFuncSetAttribute(my_hou_kernel<T, BLCOK_SIZE_X>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));

    // int numSMs = 0;
    // int maxBlocksPerSM = 0;
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
    // hou_tsqr_panel<T, 128, 32>, 32*16, share_memory_size); int maxBlocks =
    // maxBlocksPerSM * numSMs; printf("numSMs: %d maxBlocksPerSM: %d maxBlocks:
    // %d\n", numSMs, maxBlocksPerSM, maxBlocks);
    // void *kernelArgs[] = {&m, &n, &A, &lda, &R, &ldr, &work, &lwork};
    // cudaLaunchCooperativeKernel((void*)my_hou_kernel<T, M, N>, blockNum,
    // blockDim, kernelArgs, share_memory_size);

    assert(BLCOK_SIZE_X % n == 0);
    assert((m % BLCOK_SIZE_X) % n == 0);
    assert(reduction_time < 16);
    assert(n <= 128);

    dim3 block_dim(32, 16);
    int block_num = (m + BLCOK_SIZE_X - 1) / BLCOK_SIZE_X;
    my_hou_kernel<T, BLCOK_SIZE_X><<<block_num, block_dim, share_memory_size>>>(
        m, n, A, lda, R, ldr, work, lwork);
}

// template void hou_tsqr_panel<float, 128, 32>(cublasHandle_t cublas_handle,
//                                              int m, int n, float *A, int lda,
//                                              float *R, int ldr, float *work,
//                                              int lwork);
template void hou_tsqr_panel<double, 128>(cublasHandle_t cublas_handle, int m,
                                          int n, double *A, int lda, double *R,
                                          int ldr, double *work, int lwork);
