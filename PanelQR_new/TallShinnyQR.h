#pragma once

#include <cusolverDn.h>

#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T, long M, long N>
void hou_tsqr_panel(cublasHandle_t cublas_handle, long m, long n, T *A,
                    long lda, T *R, long ldr, T *work) {
    if (n > N) {
        std::cout << "hou_tsqr_panel QR the n must <= N" << std::endl;
        exit(1);
    }

    // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
    //  所以使用一个block处理一个最大为<M,N>的矩阵块，并对它进行QR分解
    dim3 blockDim(32, 16);

    if (0 != (m % M) % n) {
        std::cout << "hou_tsqr_panel QR the m%M must be multis of n"
                  << std::endl;
        exit(1);
    }

    // 2.使用按列进行分段的方式进行QR分解
    // 2.1 把瘦高矩阵进行按列分段 
    long blockNum = (m + M - 1) / M;
    long ldwork = blockNum * n;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    my_hou_kernel_new<T, M, N><<<blockNum, blockDim>>>(m, n, A, lda, work, ldwork, R, ldr);
}

template void hou_tsqr_panel<float, 128, 32>(cublasHandle_t cublas_handle,
                                             long m, long n, float *A, long lda,
                                             float *R, long ldr, float *work);
template void hou_tsqr_panel<double, 128, 32>(cublasHandle_t cublas_handle,
                                              long m, long n, double *A,
                                              long lda, double *R, long ldr,
                                              double *work);