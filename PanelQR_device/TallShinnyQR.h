#pragma once

#include <cusolverDn.h>

#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T, size_t M, size_t N>
void hou_tsqr_panel(cublasHandle_t cublas_handle, size_t m, size_t n, T *A,
                    size_t lda, T *Y, size_t ldy, T *R, size_t ldr, T *work,
                    size_t lwork) {
    dim3 blockDim(32, 32, 1);
    unsigned int blockNum = (m + M - 1) / M;
    size_t share_memory_size = 3 * M * N * sizeof(T);
    CUDA_CHECK(cudaFuncSetAttribute(QR_kernel<T, M, N>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));
    QR_kernel<T, M, N><<<blockNum, blockDim, share_memory_size>>>(
        m, n, A, lda, Y, ldy, R, ldr, work, lwork);
}

template void hou_tsqr_panel<float, 128, 32>(cublasHandle_t cublas_handle,
                                             size_t m, size_t n, float *A,
                                             size_t lda, float *Y, size_t ldy,
                                             float *R, size_t ldr, float *work,
                                             size_t lwork);
// template void hou_tsqr_panel<double, 128, 32, 3>(cublasHandle_t
// cublas_handle,
//                                               size_t m, size_t n, double *A,
//                                               size_t lda, double *Y, size_t
//                                               ldy, double *R, size_t ldr,
//                                               double *work, size_t lwork);