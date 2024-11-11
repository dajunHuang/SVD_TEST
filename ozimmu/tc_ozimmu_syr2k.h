#pragma once

#include <cusolverDn.h>

// void tc_ozimmu_syr2k(cublasHandle_t handle, long int n, long int k,  double alpha, double* A,
// long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb);

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
                     long int nb);
