// #include "../include/TensorBLAS.h"
// #inclue "cuSolver.h"
#include "mat_size.h"
#include <cusolverDn.h>

// const float sone = 1.0;
// const float snegone = -1.0;
// const float szero = 0.0;

// const double done = 1.0;

template <typename T>
static __global__ void setInitialValue(long int m, long int n, T *a, long int lda, T val)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    a[i + j * lda] = val;
  }
}

template <typename T>
static __global__ void matrixCpy(long int m, long int n, T *a, long int lda, T *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = a[i + j * lda];
  }
}

__global__ static void
matrixCpyH2F(long int m, long int n, half *a, long int lda, float *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __half2float(a[i + j * lda]);
  }
}

__global__ static void
matrixCpyF2H(long int m, long int n, float *a, long int lda, half *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __float2half(a[i + j * lda]);
  }
}

// void tc_ozimmu_syr2k_p2(cublasHandle_t handle, long int n, long int k, double alpha, double *A,
// long int lda, double *B, long int ldb, double beta, double *C, long int ldc, long int nb)
// {
//     // 两个可以合并到一起,合并到一个循环中
//     // 这一部分应该是可以并行的,
//     // 1.考虑是否存在异步函数
//     // 2.考虑启动多个线程,在多个线程中启动同时启动多个核函数
//     cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
//                                nb, nb, k, &alpha,
//                                A, CUDA_R_64F, lda, nb,
//                                B, CUDA_R_64F, ldb, nb,
//                                &beta, C, CUDA_R_64F, ldc, nb + nb * lda,
//                                n / nb, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//     cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
//                                nb, nb, k, &alpha,
//                                B, CUDA_R_64F, ldb, nb,
//                                A, CUDA_R_64F, lda, nb,
//                                &done, C, CUDA_R_64F, ldc, nb + nb * lda,
//                                n / nb, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

//     for (int i = 1; n / nb / i / 2 >= 1; i *= 2)
//     {
//         cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
//                                    i * nb, i * nb, k, &alpha,
//                                    A + i * nb, CUDA_R_64F, lda, 2 * i * nb,
//                                    B, CUDA_R_64F, ldb, 2 * i * nb,
//                                    &beta, C + i * nb, CUDA_R_64F, ldc, 2 * (i * nb + i * nb *
//                                    lda), n / nb / i / 2, CUDA_R_64F,
//                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//         cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
//                                    i * nb, i * nb, k, &alpha,
//                                    B + i * nb, CUDA_R_64F, ldb, 2 * i * nb,
//                                    A, CUDA_R_64F, lda, 2 * i * nb,
//                                    &done, C + i * nb, CUDA_R_64F, ldc, 2 * (i * nb + i * nb *
//                                    lda), n / nb / i / 2, CUDA_R_64F,
//                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//     }
// }

// void tc_ozimmu_syr2k_p3(cublasHandle_t handle, long int n, long int k, double alpha, double *A,
// long int lda, double *B, long int ldb, double beta, double *C, long int ldc, long int nb)
// {

//     int length;
//     int64_t *matSize = find_mat_size_syrk(n, &length);
//     int offset;
//     int rest_n = n;

//     printf("n=%d, k=%d,nb=%d,length=%d.\n", n, k, nb, length);

//     for (int i = length; i >= 0; i--)
//     {

//         int nn = matSize[i];

//         if (i < length)
//             offset += matSize[i + 1];
//         else
//             offset = 0;

//         printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha,
//         beta);

//         if (nn % 8192 == 0)
//         {
//             tc_ozimmu_syr2k_p2(handle, nn, k, alpha, A + offset, lda, B + offset, ldb, beta, C +
//             offset + offset * ldc, ldc, nb);
//         }
//         else
//         {
//             // 下面这个代码是可以替换为SYR2K的方式
//             cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
//                          &alpha, A + offset, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
//                          &beta, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
//                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
//                          &alpha, B + offset, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
//                          &done, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
//                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//         }
//         if (i != 0)
//         {
//             rest_n -= nn;
//             cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
//                          &alpha, A + offset + nn, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
//                          &beta, C + offset + offset * ldc + nn, CUDA_R_64F, ldc, CUDA_R_64F,
//                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
//                          &alpha, B + offset + nn, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
//                          &done, C + offset + offset * ldc + nn, CUDA_R_64F, ldc, CUDA_R_64F,
//                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//         }
//         else
//             return;
//     }
//     return;
// }
// void tc_ozimmu_syr2k(cublasHandle_t handle, long int n, long int k, double alpha, double *A, long
// int lda, double *B, long int ldb, double beta, double *C, long int ldc, long int nb)
// {
//     if (n % 2 || k % 2)
//     {
//         double *A_, *C_, *B_;
//         long int N = n, K = k, lda_, ldb_, ldc_;
//         n += n % 2;
//         k += k % 2;
//         lda_ = lda + lda % 2;
//         ldb_ = ldb + ldb % 2;
//         ldc_ = ldc + ldc % 2;
//         cudaMalloc(&A_, sizeof(double) * n * k);
//         cudaMalloc(&B_, sizeof(double) * n * k);
//         cudaMalloc(&C_, sizeof(double) * n * n);
//         printf("%ld, %ld\n", n, k);
//         dim3 grid1((n + 31) / 32, (k + 31) / 32);
//         dim3 block(32, 32);
//         setInitialValueDouble<<<grid1, block>>>(n, k, A_, lda_, 0.0);
//         setInitialValueDouble<<<grid1, block>>>(n, k, B_, ldb_, 0.0);
//         dim3 grid2((n + 31) / 32, (n + 31) / 32);
//         setInitialValueDouble<<<grid2, block>>>(n, n, C_, ldc_, 1.0);
//         dim3 grid3((N + 31) / 32, (K + 31) / 32);
//         matrixCpyDouble<<<grid3, block>>>(N, K, A, lda, A_, lda_);
//         matrixCpyDouble<<<grid3, block>>>(N, K, B, ldb, B_, ldb_);

//         tc_ozimmu_syr2k_p3(handle, n, k, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_, nb);
//         dim3 grid4((N + 31) / 32, (N + 31) / 32);
//         matrixCpyDouble<<<grid4, block>>>(N, N, C_, ldc_, C, ldc);

//         printf("check ok\n");
//         cudaFree(A_);
//         cudaFree(B_);
//         cudaFree(C_);
//     }
//     else
//     {
//         tc_ozimmu_syr2k_p3(handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nb);
//     }
// }

template <typename T>
void tc_ozimmu_syr2k_p2(cublasHandle_t handle,
                        long int n,
                        long int k,
                        T alpha,
                        T *A,
                        long int lda,
                        T *B,
                        long int ldb,
                        T beta,
                        T *C,
                        long int ldc,
                        long int nb)
{
  T tOne = 1.0;

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

  //   cublasGemmStridedBatchedEx(handle,
  //                              CUBLAS_OP_N,
  //                              CUBLAS_OP_T,
  //                              nb,
  //                              nb,
  //                              k,
  //                              &alpha,
  //                              A,
  //                              CUDA_R_64F,
  //                              lda,
  //                              nb,
  //                              B,
  //                              CUDA_R_64F,
  //                              ldb,
  //                              nb,
  //                              &beta,
  //                              C,
  //                              CUDA_R_64F,
  //                              ldc,
  //                              nb + nb * ldc,
  //                              n / nb,
  //                              CUDA_R_64F,
  //                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasGemmStridedBatchedEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             nb,
                             nb,
                             k,
                             &alpha,
                             A,
                             cuda_data_type,
                             lda,
                             nb,
                             B,
                             cuda_data_type,
                             ldb,
                             nb,
                             &beta,
                             C,
                             cuda_data_type,
                             ldc,
                             nb + nb * ldc,
                             n / nb,
                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  //   cublasGemmStridedBatchedEx(handle,
  //                              CUBLAS_OP_N,
  //                              CUBLAS_OP_T,
  //                              nb,
  //                              nb,
  //                              k,
  //                              &alpha,
  //                              B,
  //                              CUDA_R_64F,
  //                              ldb,
  //                              nb,
  //                              A,
  //                              CUDA_R_64F,
  //                              lda,
  //                              nb,
  //                              &done,
  //                              C,
  //                              CUDA_R_64F,
  //                              ldc,
  //                              nb + nb * ldc,

  //                              n / nb,
  //                              CUDA_R_64F,
  //                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasGemmStridedBatchedEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             nb,
                             nb,
                             k,
                             &alpha,
                             B,
                             cuda_data_type,
                             ldb,
                             nb,
                             A,
                             cuda_data_type,
                             lda,
                             nb,
                             &tOne,
                             C,
                             cuda_data_type,
                             ldc,
                             nb + nb * ldc,

                             n / nb,
                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  for (int i = 1; n / nb / i / 2 >= 1; i *= 2)
  {
    // cublasGemmStridedBatchedEx(handle,
    //                            CUBLAS_OP_N,
    //                            CUBLAS_OP_T,
    //                            i * nb,
    //                            i * nb,
    //                            k,
    //                            &alpha,
    //                            A + i * nb,
    //                            CUDA_R_64F,
    //                            lda,
    //                            2 * i * nb,
    //                            B,
    //                            CUDA_R_64F,
    //                            ldb,
    //                            2 * i * nb,
    //                            &beta,
    //                            C + i * nb,
    //                            CUDA_R_64F,
    //                            ldc,
    //                            2 * (i * nb + i * nb * ldc),
    //                            n / nb / i / 2,
    //                            CUDA_R_64F,
    //                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasGemmStridedBatchedEx(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               i * nb,
                               i * nb,
                               k,
                               &alpha,
                               A + i * nb,
                               cuda_data_type,
                               lda,
                               2 * i * nb,
                               B,
                               cuda_data_type,
                               ldb,
                               2 * i * nb,
                               &beta,
                               C + i * nb,
                               cuda_data_type,
                               ldc,
                               2 * (i * nb + i * nb * ldc),
                               n / nb / i / 2,
                               cublas_compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // cublasGemmStridedBatchedEx(handle,
    //                            CUBLAS_OP_N,
    //                            CUBLAS_OP_T,
    //                            i * nb,
    //                            i * nb,
    //                            k,
    //                            &alpha,
    //                            B + i * nb,
    //                            CUDA_R_64F,
    //                            ldb,
    //                            2 * i * nb,
    //                            A,
    //                            CUDA_R_64F,
    //                            lda,
    //                            2 * i * nb,
    //                            &done,
    //                            C + i * nb,
    //                            CUDA_R_64F,
    //                            ldc,
    //                            2 * (i * nb + i * nb * ldc),
    //                            n / nb / i / 2,
    //                            CUDA_R_64F,
    //                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasGemmStridedBatchedEx(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               i * nb,
                               i * nb,
                               k,
                               &alpha,
                               B + i * nb,
                               cuda_data_type,
                               ldb,
                               2 * i * nb,
                               A,
                               cuda_data_type,
                               lda,
                               2 * i * nb,
                               &tOne,
                               C + i * nb,
                               cuda_data_type,
                               ldc,
                               2 * (i * nb + i * nb * ldc),
                               n / nb / i / 2,
                               cublas_compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

#if 0
template <typename T>
void tc_ozimmu_syr2k_p3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        T alpha,
                        T *A,
                        long int lda,
                        T *B,
                        long int ldb,
                        T beta,
                        T *C,
                        long int ldc,
                        long int nb)
{

  int length;
  int64_t *matSize = find_mat_size_syrk(n, &length);
  int offset;
  int rest_n = n;

  T tOne = 1.0;

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

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {
      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      tc_ozimmu_syr2k_p2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    else
    {

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      // printf("Matrix A:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, A + offset, lda);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, A + offset, lda);
      // printDeviceMatrixV2(A + offset, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + (nn - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (nn - 10) + (k - 10) * ldb, ldb, 10, 10);

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, A + offset, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
      //              &beta, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // // printf("%s:%d\n", __FILE__, __LINE__);
      // // printf("Matrix C1, done=%lf:\n", done);
      // // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // // cudaDeviceSynchronize();

      // // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // // const double done = 1.0;

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, B + offset, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
      //              &done, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // 这个函数也没有呢
      cublasDsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &alpha,
                   A + offset,
                   lda,
                   B + offset,
                   ldb,
                   &beta,
                   C + offset + offset * ldc,
                   ldc);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    if (i != 0)
    {
      rest_n -= nn;

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);

      // printf("Matrix A:\n");
      // printDeviceMatrixV2(A + offset + nn, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + nn + (rest_n - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (n - 10) + (k - 10) * ldb, ldb, 10, 10);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                A + offset + nn,
      //                CUDA_R_64F,
      //                lda,
      //                B + offset,
      //                CUDA_R_64F,
      //                ldb,
      //                &beta,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                B + offset + nn,
      //                CUDA_R_64F,
      //                ldb,
      //                A + offset,
      //                CUDA_R_64F,
      //                lda,
      //                &done,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &tOne,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);
    }
    else
      return;
  }
  return;
}
#endif

template <typename T>
void tc_ozimmu_syr2k_p3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        T alpha,
                        T *A,
                        long int lda,
                        T *B,
                        long int ldb,
                        T beta,
                        T *C,
                        long int ldc,
                        long int nb);

template <>
void tc_ozimmu_syr2k_p3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        double alpha,
                        double *A,
                        long int lda,
                        double *B,
                        long int ldb,
                        double beta,
                        double *C,
                        long int ldc,
                        long int nb)
{

  double done = 1.0;

  int length;
  int64_t *matSize = find_mat_size_syrk(n, &length);
  int offset;
  int rest_n = n;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_64F;
  cublas_compute_type = CUBLAS_COMPUTE_64F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {
      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      tc_ozimmu_syr2k_p2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    else
    {

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      // printf("Matrix A:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, A + offset, lda);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, A + offset, lda);
      // printDeviceMatrixV2(A + offset, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + (nn - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (nn - 10) + (k - 10) * ldb, ldb, 10, 10);

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, A + offset, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
      //              &beta, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // // printf("%s:%d\n", __FILE__, __LINE__);
      // // printf("Matrix C1, done=%lf:\n", done);
      // // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // // cudaDeviceSynchronize();

      // // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // // const double done = 1.0;

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, B + offset, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
      //              &done, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // 这个函数也没有呢
      cublasDsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &alpha,
                   A + offset,
                   lda,
                   B + offset,
                   ldb,
                   &beta,
                   C + offset + offset * ldc,
                   ldc);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    if (i != 0)
    {
      rest_n -= nn;

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);

      // printf("Matrix A:\n");
      // printDeviceMatrixV2(A + offset + nn, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + nn + (rest_n - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (n - 10) + (k - 10) * ldb, ldb, 10, 10);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                A + offset + nn,
      //                CUDA_R_64F,
      //                lda,
      //                B + offset,
      //                CUDA_R_64F,
      //                ldb,
      //                &beta,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                B + offset + nn,
      //                CUDA_R_64F,
      //                ldb,
      //                A + offset,
      //                CUDA_R_64F,
      //                lda,
      //                &done,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &done,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);
    }
    else
      return;
  }
  return;
}

template <>
void tc_ozimmu_syr2k_p3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        float alpha,
                        float *A,
                        long int lda,
                        float *B,
                        long int ldb,
                        float beta,
                        float *C,
                        long int ldc,
                        long int nb)
{

  int length;
  int64_t *matSize = find_mat_size_syrk(n, &length);
  int offset;
  int rest_n = n;

  float fone = 1.0;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_32F;
  cublas_compute_type = CUBLAS_COMPUTE_32F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {
      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      tc_ozimmu_syr2k_p2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    else
    {

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      // printf("Matrix A:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, A + offset, lda);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, A + offset, lda);
      // printDeviceMatrixV2(A + offset, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + (nn - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (nn - 10) + (k - 10) * ldb, ldb, 10, 10);

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, A + offset, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
      //              &beta, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // // printf("%s:%d\n", __FILE__, __LINE__);
      // // printf("Matrix C1, done=%lf:\n", done);
      // // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // // cudaDeviceSynchronize();

      // // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // // const double done = 1.0;

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, B + offset, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
      //              &done, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // 这个函数也没有呢
      cublasSsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &alpha,
                   A + offset,
                   lda,
                   B + offset,
                   ldb,
                   &beta,
                   C + offset + offset * ldc,
                   ldc);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    if (i != 0)
    {
      rest_n -= nn;

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);

      // printf("Matrix A:\n");
      // printDeviceMatrixV2(A + offset + nn, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + nn + (rest_n - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (n - 10) + (k - 10) * ldb, ldb, 10, 10);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                A + offset + nn,
      //                CUDA_R_64F,
      //                lda,
      //                B + offset,
      //                CUDA_R_64F,
      //                ldb,
      //                &beta,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                B + offset + nn,
      //                CUDA_R_64F,
      //                ldb,
      //                A + offset,
      //                CUDA_R_64F,
      //                lda,
      //                &done,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &fone,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);
    }
    else
      return;
  }
  return;
}

template <>
void tc_ozimmu_syr2k_p3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        half alpha,
                        half *A,
                        long int lda,
                        half *B,
                        long int ldb,
                        half beta,
                        half *C,
                        long int ldc,
                        long int nb)
{

  int length;
  int64_t *matSize = find_mat_size_syrk(n, &length);
  int offset;
  int rest_n = n;

  half hone = 1.0;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_16F;
  cublas_compute_type = CUBLAS_COMPUTE_16F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {
      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      tc_ozimmu_syr2k_p2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }
    else
    {

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);

      // printf("Matrix A:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, A + offset, lda);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, A + offset, lda);
      // printDeviceMatrixV2(A + offset, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + (nn - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (nn - 10) + (k - 10) * ldb, ldb, 10, 10);

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, A + offset, CUDA_R_64F, lda, B + offset, CUDA_R_64F, ldb,
      //              &beta, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // // printf("%s:%d\n", __FILE__, __LINE__);
      // // printf("Matrix C1, done=%lf:\n", done);
      // // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);
      // // // cudaDeviceSynchronize();

      // // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);

      // // const double done = 1.0;

      // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
      //              &alpha, B + offset, CUDA_R_64F, ldb, A + offset, CUDA_R_64F, lda,
      //              &done, C + offset + offset * ldc, CUDA_R_64F, ldc, CUDA_R_64F,
      //              //  CUBLAS_GEMM_DEFAULT);
      //              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      float *A_, *B_, *C_;

      float sAlpha = __half2float(alpha);
      float sBeta  = __half2float(beta);

      cudaMalloc(&A_, sizeof(float) * nn * k);
      cudaMalloc(&B_, sizeof(float) * nn * k);
      cudaMalloc(&C_, sizeof(float) * nn * nn);
      //   printf("%ld, %ld\n", n, k);
      dim3 grid1((nn + 31) / 32, (k + 31) / 32);
      dim3 block(32, 32);

      matrixCpyH2F<<<grid1, block>>>(nn, k, A + offset, lda, A_, nn);
      matrixCpyH2F<<<grid1, block>>>(nn, k, B + offset, ldb, B_, nn);
      matrixCpyH2F<<<grid1, block>>>(nn, nn, C + offset + offset * ldc, ldc, C_, nn);

      cublasSsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &sAlpha,
                   A_,
                   nn,
                   B_,
                   nn,
                   &sBeta,
                   C_,
                   nn);

      matrixCpyF2H<<<grid1, block>>>(nn, k, C_, nn, C + offset + offset * ldc, ldc);

      cudaFree(A_);
      cudaFree(B_);
      cudaFree(C_);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc, ldc, 10, 10);
      // printf("Matrix end C2:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + (nn - 10) + (nn - 10) * ldc, ldc, 10, 10);
    }

    if (i != 0)
    {
      rest_n -= nn;

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);

      // printf("Matrix A:\n");
      // printDeviceMatrixV2(A + offset + nn, lda, 10, 10);

      // printf("Matrix end A:\n");
      // printDeviceMatrixV2(A + offset + nn + (rest_n - 10) + (k - 10) * lda, lda, 10, 10);

      // printf("Matrix B:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, k, B + offset, ldb);
      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, B + offset, ldb);
      // printDeviceMatrixV2(B + offset, ldb, 10, 10);

      // printf("Matrix end B:\n");
      // printDeviceMatrixV2(B + offset + (n - 10) + (k - 10) * ldb, ldb, 10, 10);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                A + offset + nn,
      //                CUDA_R_64F,
      //                lda,
      //                B + offset,
      //                CUDA_R_64F,
      //                ldb,
      //                &beta,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      //   cublasGemmEx(handle,
      //                CUBLAS_OP_N,
      //                CUBLAS_OP_T,
      //                rest_n,
      //                nn,
      //                k,
      //                &alpha,
      //                B + offset + nn,
      //                CUDA_R_64F,
      //                ldb,
      //                A + offset,
      //                CUDA_R_64F,
      //                lda,
      //                &done,
      //                C + offset + offset * ldc + nn,
      //                CUDA_R_64F,
      //                ldc,
      //                CUDA_R_64F,
      //                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &hone,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      // printf("%s:%d\n", __FILE__, __LINE__);
      // printf("Matrix C2:\n");
      // // printMatrixDeviceBlockDouble("/dev/stdout", nn, nn, C + offset + offset * ldc, ldc);

      // // printMatrixDeviceBlockDouble("/dev/stdout", 10, 10, C + offset + offset * ldc, ldc);
      // printDeviceMatrixV2(C + offset + offset * ldc + nn, ldc, 10, 10);
      // printf("Matrix end C:\n");
      // printDeviceMatrixV2(C + offset + offset * ldc + nn + (rest_n - 10) + (nn - 10) * ldc, ldc,
      // 10, 10);
    }
    else
      return;
  }
  return;
}

template <typename T>
void tc_ozimmu_syr2k(cublasHandle_t handle,
                     long int n,
                     long int k,
                     T alpha,
                     T *A,
                     long int lda,
                     T *B,
                     long int ldb,
                     T beta,
                     T *C,
                     long int ldc,
                     long int nb)
{
  if (n % 2 || k % 2)
  {
    T *A_, *C_, *B_;
    long int N = n, K = k, lda_, ldb_, ldc_;
    n += n % 2;
    k += k % 2;
    lda_ = lda + lda % 2;
    ldb_ = ldb + ldb % 2;
    ldc_ = ldc + ldc % 2;
    cudaMalloc(&A_, sizeof(T) * n * k);
    cudaMalloc(&B_, sizeof(T) * n * k);
    cudaMalloc(&C_, sizeof(T) * n * n);
    printf("%ld, %ld\n", n, k);
    dim3 grid1((n + 31) / 32, (k + 31) / 32);
    dim3 block(32, 32);
    setInitialValue<<<grid1, block>>>(n, k, A_, lda_, T(0.0));
    setInitialValue<<<grid1, block>>>(n, k, B_, ldb_, T(0.0));
    dim3 grid2((n + 31) / 32, (n + 31) / 32);
    setInitialValue<<<grid2, block>>>(n, n, C_, ldc_, T(1.0));
    dim3 grid3((N + 31) / 32, (K + 31) / 32);
    matrixCpy<<<grid3, block>>>(N, K, A, lda, A_, lda_);
    matrixCpy<<<grid3, block>>>(N, K, B, ldb, B_, ldb_);

    tc_ozimmu_syr2k_p3(handle, n, k, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_, nb);
    dim3 grid4((N + 31) / 32, (N + 31) / 32);
    matrixCpy<<<grid4, block>>>(N, N, C_, ldc_, C, ldc);

    printf("check ok\n");
    cudaFree(A_);
    cudaFree(B_);
    cudaFree(C_);
  }
  else
  {
    tc_ozimmu_syr2k_p3(handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nb);
  }
}

template void tc_ozimmu_syr2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              double alpha,
                              double *A,
                              long int lda,
                              double *B,
                              long int ldb,
                              double beta,
                              double *C,
                              long int ldc,
                              long int nb);
template void tc_ozimmu_syr2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              float alpha,
                              float *A,
                              long int lda,
                              float *B,
                              long int ldb,
                              float beta,
                              float *C,
                              long int ldc,
                              long int nb);

template void tc_ozimmu_syr2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              half alpha,
                              half *A,
                              long int lda,
                              half *B,
                              long int ldb,
                              half beta,
                              half *C,
                              long int ldc,
                              long int nb);