#pragma once

#include <cusolverDn.h>

#include <iostream>
#include<cmath>
#include "kernelQR.h"
#include "myBase.h"

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T, int M, int N>
void hou_tsqr_panel(cublasHandle_t cublas_handle, int m, int n, T *A, int lda,
                    T *R, int ldr, T *work, int lwork) {
    if (n > N) {
        std::cout << "hou_tsqr_panel QR the n must <= N" << std::endl;
        exit(1);
    }

    int reduction_time = ceil(log(m / 32) / log(4));
    // printf("size %d, reduction_time: %d\n", m, reduction_time);
    int share_memory_size = reduction_time * M * N * sizeof(T);

    CUDA_CHECK(cudaFuncSetAttribute(my_hou_kernel<T, M, N>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));

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
    int blockNum = (m + M - 1) / M;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    my_hou_kernel<T, M, N><<<blockNum, blockDim, share_memory_size>>>(
        m, n, A, lda, R, ldr, work, lwork);

    printDeviceMatrixV2(R, ldr, 16, 16);
}

template void hou_tsqr_panel<float, 128, 32>(cublasHandle_t cublas_handle,
                                             int m, int n, float *A, int lda,
                                             float *R, int ldr, float *work,
                                             int lwork);