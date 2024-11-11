#pragma once

#include <cusolverDn.h>

// void tc_ozimmu_syr2k(cublasHandle_t handle, long int n, long int k,  double alpha, double* A,
// long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb);

void tc_syr2k(cublasHandle_t handle,
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
              __half *hwork,
              long int nb);
