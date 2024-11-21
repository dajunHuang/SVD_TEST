#pragma once

#include <iostream>

#include "kernelQR.h"
#include "myBase.h"
#include <cusolverDn.h>

// hou_tsqr_panel实现
// 这个是copilot实现的版本
// template <long M, long N>
// void hou_tsqr_panel(double *A, long lda, double *R, long ldr, double *tau, double *work, long
// lwork)
// {
//     // 1. 计算QR分解
//     // 1.1 计算QR分解的工作空间
//     long info;
//     long lwork_query = -1;
//     double *work_query = new double[1];
//     dgeqrf_(&M, &N, A, &lda, tau, work_query, &lwork_query, &info);
//     long lwork_geqrf = (long)work_query[0];
//     delete[] work_query;
//     // 1.2 计算QR分解
//     dgeqrf_(&M, &N, A, &lda, tau, work, &lwork, &info);
//     // 2. 计算R
//     // 2.1 计算R的工作空间
//     lwork_query = -1;
//     dorgqr_(&M, &N, &N, A, &lda, tau, work_query, &lwork_query, &info);
//     long lwork_orgqr = (long)work_query[0];
//     delete[] work_query;
//     // 2.2 计算R
//     dorgqr_(&M, &N, &N, A, &lda, tau, work, &lwork, &info);
//     // 3. 拷贝R
//     for (long i = 0; i < N; i++)
//     {
//         for (long j = 0; j < N; j++)
//         {
//             R[i + j * ldr] = A[i + j * lda];
//         }
//     }
// }

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T, long M, long N>
void hou_tsqr_panel(cublasHandle_t cublas_handle,
                    long m,
                    long n,
                    T *A,
                    long lda,
                    T *R,
                    long ldr,
                    T *work)
{
  if (n > N)
  {
    std::cout << "hou_tsqr_panel QR the n must <= N" << std::endl;
    exit(1);
  }

  // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
  //  所以使用一个block处理一个最大为<M,N>的矩阵块，并对它进行QR分解
  dim3 blockDim(32, 16);
  // dim3 blockDim(32, 8); // 仅仅为了4090

  // 1.如果m<=M,就直接调用核函数进行QR分解
  if (m <= M)
  {
    // 调用核函数进行QR分解
    // 分解后A矩阵中存放的是Q矩阵，R矩阵中存放的是R矩阵
    my_hou_kernel<M, N><<<1, blockDim>>>(m, n, A, lda, R, ldr);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    return;
  }

  if (0 != (m % M) % n) 
  {
    std::cout << "hou_tsqr_panel QR the m%M must be multis of n" << std::endl;
    exit(1);
  }

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  if (std::is_same<T, double>::value)
  {
    cuda_data_type      = CUDA_R_64F;
    cublas_compute_type = CUBLAS_COMPUTE_64F;
  }
  else if (std::is_same<T, float>::value)
  {
    cuda_data_type      = CUDA_R_32F;
    cublas_compute_type = CUBLAS_COMPUTE_32F;
  }
  else if (std::is_same<T, half>::value)
  {
    cuda_data_type      = CUDA_R_16F;
    cublas_compute_type = CUBLAS_COMPUTE_16F;
  }

  // 2.使用按列进行分段的方式进行QR分解
  // 2.1 把瘦高矩阵进行按列分段
  long blockNum = (m + M - 1) / M;
  long ldwork   = blockNum * n;

  // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
  my_hou_kernel<M, N><<<blockNum, blockDim>>>(m, n, A, lda, work, ldwork);
  // std::cout << __func__ << " " << __LINE__ << " m=" << m << ", n=" << n << ", M=" << M <<
  // std::endl; CHECK(cudaGetLastError()); cudaDeviceSynchronize();

  // std::cout << "print dA:" << std::endl;
  // printDeviceMatrixV2(A, lda, m, n);

  // std::cout << "print dwork:" << std::endl;
  // printDeviceMatrixV2(work, ldwork, ldwork, n);

  // 2.3再对R进行QR分解,也就是对work进行递归调用此函数
  hou_tsqr_panel<T, M, N>(cublas_handle, ldwork, n, work, ldwork, R, ldr, work + n * ldwork);

  // std::cout << "print dR:" << std::endl;
  // printDeviceMatrixV2(R, ldr, n, n);

  // std::cout << "print dA:" << std::endl;
  // printDeviceMatrixV2(A, lda, m, n);

  // 3.求出最终的Q，存放到A中
  // 注意这里使用了一个batch乘积的方法，是一个非常有趣的思想,需要结合瘦高矩阵的分块矩阵理解，非常有意思
  T tone = 1.0, tzero = 0.0;
  cublasGemmStridedBatchedEx(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             M,
                             n,
                             n,
                             &tone,
                             A,
                             cuda_data_type,
                             lda,
                             M,

                             work,
                             cuda_data_type,
                             ldwork,
                             n,

                             &tzero,

                             A,
                             cuda_data_type,
                             lda,
                             M,
                             m / M,

                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT);

  //   cublasDgemmStridedBatched(cublas_handle,
  //                             CUBLAS_OP_N,
  //                             CUBLAS_OP_N,
  //                             M,
  //                             n,
  //                             n,
  //                             &tone,
  //                             A,
  //                             lda,
  //                             M,
  //                             work,
  //                             ldwork,
  //                             n,
  //                             &tzero,
  //                             A,
  //                             lda,
  //                             M,
  //                             m / M);

  // std::cout << __func__ << " " << __LINE__ << " m=" << m << ", n=" << n << ", M=" << M <<
  // std::endl;

  // CHECK(cudaGetLastError());
  // cudaDeviceSynchronize();

  // std::cout << "print dA:" << std::endl;
  // printDeviceMatrixV2(A, lda, m, n);

  // 3.2如果m/M还有剩余的话，还需要计算最后一个块的Q进行乘法计算，才能得到最终的Q
  long mm = m % M;
  if (0 < mm)
  {
#if MY_DEBUG
    std::cout << __func__ << " " << __LINE__ << " come m % M !=0 case." << std::endl;
#endif

    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 mm,
                 n,
                 n,
                 &tone,
                 A + (m - mm),
                 cuda_data_type,
                 lda,

                 work + (m / M * n),
                 cuda_data_type,
                 ldwork,

                 &tzero,
                 A + (m - mm),
                 cuda_data_type,
                 lda,

                 cublas_compute_type,
                 CUBLAS_GEMM_DEFAULT);

    // cublasDgemm(cublas_handle,
    //             CUBLAS_OP_N,
    //             CUBLAS_OP_N,
    //             mm,
    //             n,
    //             n,
    //             &tone,
    //             A + (m - mm),
    //             lda,
    //             work + (m / M * n),
    //             ldwork,
    //             &tzero,
    //             A + (m - mm),
    //             lda);
  }

  // std::cout << "print dA:" << std::endl;
  // printDeviceMatrixV2(A, lda, m, n);
}