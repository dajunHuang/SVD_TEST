#pragma once

#include <cusolverDn.h>

#include <cmath>
#include <iostream>

#include "kernelQR.h"
#include "myBase.h"

template <typename T>
__global__ void pre_QR_reduction(int block_size, int m, int n, T *A, int lda,
                                 T *R, int ldr) {
    __shared__ T shared_RR[128];
    const int block_idx_x = blockIdx.x;

    int block_data_height = min(m - block_idx_x * block_size, block_size);
    if (block_data_height > 0) {
        T *A_from = &A[block_idx_x * block_size];
        T *Q_to = &A[block_idx_x * block_size];
        T *R_to = &R[block_idx_x * n];
        int lda_from = lda, ldq_to = lda, ldr_to = ldr;
        qr_kernel<T>(block_data_height, n, A_from, lda_from, Q_to, ldq_to, R_to,
                     ldr_to, shared_RR);
    }
}

template __global__ void pre_QR_reduction<double>(int block_size, int m, int n,
                                                  double *A, int lda, double *R,
                                                  int ldr);

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int block_size, int m, int n, T *A,
          int lda, T *R, int ldr, T *d_work1, int ldwork1, T *d_work2,
          int ldwork2) {
    T *temp_A = A;
    int ldtemp_A = lda;
    int pre_reduction_data_height[16];
    dim3 block_dim(32, 32);  // if change block_dim, also change acc_per_thread
                             // and q_per_thread mannually in kernelQR.h
    int block_num = (m + block_size - 1) / block_size;

    assert(block_size % n == 0);
    assert((m % block_size) % n == 0);
    assert(n <= 128);

    int pre_reduction_time =
        ceil((log(m) - log(13824)) / (log(block_size) - log(n)));
    // printf("size: %d, pre_reduction_time: %d\n", m, pre_reduction_time);

    if (pre_reduction_time > 0) {
        // pre_reudction
        pre_reduction_data_height[0] = m;
        pre_QR_reduction<T><<<block_num, block_dim>>>(block_size, m, n, A, lda,
                                                      d_work1, ldwork1);

        m = ((m + block_size - 1) / block_size) * n;
        block_num = (m + block_size - 1) / block_size;

        for (int i = 1; i < pre_reduction_time; ++i) {
            pre_reduction_data_height[i] = m;
            pre_QR_reduction<T><<<block_num, block_dim>>>(
                block_size, m, n, &d_work1[(i - 1) * n * ldwork1], ldwork1,
                &d_work1[i * n * ldwork1], ldwork1);

            m = ((m + block_size - 1) / block_size) * n;
            block_num = (m + block_size - 1) / block_size;
        }
        temp_A = &d_work1[(pre_reduction_time - 1) * n * ldwork1];
        ldtemp_A = ldwork1;
    }

    int in_block_reduction_time =
        ceil((log(m) - log(n)) / (log(block_size) - log(n)));
    // printf("size %d, in_block_reduction_time: %d\n", m,
    // in_block_reduction_time);

    int share_memory_size =
        in_block_reduction_time * block_size * n * sizeof(T);

    CUDA_CHECK(cudaFuncSetAttribute(householder_kernel<T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size));

    householder_kernel<T><<<block_num, block_dim, share_memory_size>>>(
        block_size, m, n, temp_A, ldtemp_A, R, ldr, d_work2, ldwork2);

    // stride gemm
    if (pre_reduction_time > 0) {
        // for(int i = 0; i < pre_reduction_time; ++i) {
        //     printf("%d\n", pre_reduction_data_height[i]);
        // }
        double one = 1, zero = 0;
        for (int i = pre_reduction_time - 2; i >= 0; --i) {
            m = pre_reduction_data_height[i + 1];
            cublasDgemmStridedBatched(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, block_size, n, n, &one,
                &d_work1[i * n * ldwork1], ldwork1, block_size,
                &d_work1[(i + 1) * n * ldwork1], ldwork1, n, &zero,
                &d_work1[i * n * ldwork1], ldwork1, block_size, m / block_size);
            int last_block_idx = m / block_size;
            int last_block_offset = last_block_idx * block_size;
            if (last_block_offset < m) {
                cublasDgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m % block_size, n,
                    n, &one,
                    &d_work1[i * n * ldwork1 + last_block_offset],
                    ldwork1,
                    &d_work1[(i + 1) * n * ldwork1 + last_block_idx * n],
                    ldwork1, &zero,
                    &d_work1[i * n * ldwork1 + last_block_offset],
                    ldwork1);
            }
            // cudaDeviceSynchronize();
        }
        m = pre_reduction_data_height[0];
        cublasDgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  block_size, n, n, &one, A, lda, block_size,
                                  d_work1, ldwork1, n, &zero, A, lda,
                                  block_size, m / block_size);
        int last_block_idx = m / block_size;
        int last_block_offset = last_block_idx * block_size;
        if (last_block_offset < m) {
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m % block_size,
                        n, n, &one, &A[last_block_offset], lda,
                        &d_work1[last_block_idx * n], ldwork1, &zero,
                        &A[last_block_offset], lda);
        }
    }
}

template void tsqr<double>(cublasHandle_t cublas_handle, int block_size, int m,
                           int n, double *A, int lda, double *R, int ldr,
                           double *d_work1, int ldwork1, double *d_work2,
                           int ldwork2);
