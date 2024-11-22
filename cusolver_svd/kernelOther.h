#include <cuda_fp16.h>
#include <stdio.h>

template <typename T>
__global__ void clearMatrix(long m, long n, T *A, long ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d\n", __LINE__);
    // __syncthreads();

    if (i < m && j < n) {
        A[i + j * ldA] = 0.0;
    }
}

template <typename T>
void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *A,
                              long ldA) {
    clearMatrix<<<gridDim, blockDim>>>(m, n, A, ldA);
}

template void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m,
                                       long n, double *A, long ldA);
template void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m,
                                       long n, float *A, long ldA);
template void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m,
                                       long n, half *A, long ldA);

template <typename T>
static __global__ void kernel_setMetrixTrValue(long m, long n, T *A, long ldA,
                                               T v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("come %d\n", __LINE__);
    // __syncthreads();
    if ((i < m) && (i < n)) {
        A[i + i * ldA] = v;
    }
}

template <typename T>
void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m, long n,
                                   T *A, long ldA, T v) {
    kernel_setMetrixTrValue<<<gridDim, blockDim>>>(m, n, A, ldA, v);
}

template void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m,
                                            long n, double *A, long ldA,
                                            double v);
template void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m,
                                            long n, float *A, long ldA,
                                            float v);
template void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m,
                                            long n, half *A, long ldA, half v);

template <typename T>
__global__ void copyMatrixL2U(long n, T *A, long ldA) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d\n", __LINE__);
    // __syncthreads();

    if (i < n && j < n) {
        if (j > i) A[i + j * ldA] = A[j + i * ldA];
    }
}

template <typename T>
void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, T *A,
                               long ldA) {
    copyMatrixL2U<<<gridDim, blockDim>>>(n, A, ldA);
}

template void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n,
                                        double *A, long ldA);
template void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n,
                                        float *A, long ldA);
template void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n,
                                        half *A, long ldA);

template <typename T>
__global__ void copyAndClear(long m, long n, T *srcM, long lds, T *dstM,
                             long ldd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        dstM[i + j * ldd] = srcM[i + j * lds];
        srcM[i + j * lds] = 0.0;
    }
}

template <typename T>
void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m, long n,
                               T *srcM, long lds, T *dstM, long ldd) {
    copyAndClear<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m,
                                        long n, double *srcM, long lds,
                                        double *dstM, long ldd);

template void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m,
                                        long n, float *srcM, long lds,
                                        float *dstM, long ldd);

template void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m,
                                        long n, half *srcM, long lds,
                                        half *dstM, long ldd);

template <typename T>
__global__ void IminusQ(long m, long n, T *Q, long ldq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        if (i == j) {
            Q[i + j * ldq] = (T)1.0 - Q[i + j * ldq];
        } else {
            Q[i + j * ldq] = -Q[i + j * ldq];
        }

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

template <typename T>
void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, T *Q,
                          long ldq) {
    IminusQ<<<gridDim, blockDim>>>(m, n, Q, ldq);
}

template void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n,
                                   double *Q, long ldq);

template void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n,
                                   float *Q, long ldq);

template void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n,
                                   half *Q, long ldq);

template <typename T>
__global__ void Identity(long m, long n, T *Q, long ldq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        if (i == j) {
            Q[i + j * ldq] = 1;
        } else {
            Q[i + j * ldq] = 0;
        }
    }
}

template <typename T>
void launchKernel_Identity(dim3 gridDim, dim3 blockDim, long m, long n, T *Q,
                          long ldq) {
    Identity<<<gridDim, blockDim>>>(m, n, Q, ldq);
}

template void launchKernel_Identity(dim3 gridDim, dim3 blockDim, long m, long n,
                                   double *Q, long ldq);

template void launchKernel_Identity(dim3 gridDim, dim3 blockDim, long m, long n,
                                   float *Q, long ldq);

template void launchKernel_Identity(dim3 gridDim, dim3 blockDim, long m, long n,
                                   half *Q, long ldq);

__global__ void AminusB(long m, long n, double *A, long ldA, double *B,
                        long ldB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        A[i + j * ldA] -= B[i + j * ldB];

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_AminusB(dim3 gridDim, dim3 blockDim, long m, long n,
                          double *A, long ldA, double *B, long ldB) {
    AminusB<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

__global__ void AbsAminusAbsB(long m, long n, double *A, long ldA, double *B,
                              long ldB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        // A[i + j * ldA] = abs(A[i + j * ldA]);
        double t = abs(A[i + j * ldA]);

        A[i + j * ldA] = t - abs(B[i + j * ldB]);

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_AbsAminusAbsB(dim3 gridDim, dim3 blockDim, long m, long n,
                                double *A, long ldA, double *B, long ldB) {
    AbsAminusAbsB<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

template <typename T>
__global__ void copyMatrix(long m, long n, T *srcM, long lds, T *dstM,
                           long ldd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        dstM[i + j * ldd] = srcM[i + j * lds];
    }
}

template <typename T>
void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m, long n,
                             T *srcM, long lds, T *dstM, long ldd) {
    copyMatrix<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m,
                                      long n, double *srcM, long lds,
                                      double *dstM, long ldd);
template void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m,
                                      long n, float *srcM, long lds,
                                      float *dstM, long ldd);

template void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m,
                                      long n, half *srcM, long lds, half *dstM,
                                      long ldd);

// 将src矩阵拷贝到dst的转置中
// 这个和函数是以src建立的，所以以src为主
template <typename T>
__global__ void copyMatrixAToTranpB(long m, long n, T *srcM, long lds, T *dstM,
                                    long ldd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // dst[j][i] = src[i][j]
    if (i < m && j < n) {
        dstM[j + i * ldd] = srcM[i + j * lds];
    }
}

template <typename T>
void launchKernel_copyMatrixAToTranpB(dim3 gridDim, dim3 blockDim, long m,
                                      long n, T *srcM, long lds, T *dstM,
                                      long ldd) {
    copyMatrixAToTranpB<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchKernel_copyMatrixAToTranpB(dim3 gridDim, dim3 blockDim,
                                               long m, long n, double *srcM,
                                               long lds, double *dstM,
                                               long ldd);

template void launchKernel_copyMatrixAToTranpB(dim3 gridDim, dim3 blockDim,
                                               long m, long n, float *srcM,
                                               long lds, float *dstM, long ldd);

// get U from LU factorization
// 从A=LU分解中得到U,也是使用核函数的思想
// 参数：a是A矩阵，u是待输出的矩阵U
template <typename T>
__global__ void getU(int m, int n, T *A, int ldA, T *U, int ldU) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i > j)
            U[i + j * ldU] = 0;
        else
            U[i + j * ldU] = A[i + j * ldA];
    }
}

template <typename T>
void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, T *A, int ldA,
                       T *U, int ldU) {
    getU<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

template void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n,
                                double *A, int ldA, double *U, int ldU);

template void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n,
                                float *A, int ldA, float *U, int ldU);

template void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n,
                                half *A, int ldA, half *U, int ldU);

template <typename T>
__global__ void moveU(int m, int n, T *A, int ldA, T *U, int ldU) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i > j) {
            U[i + j * ldU] = 0;
        } else if(i == j) {
            U[i + j * ldU] = A[i + j * ldA];
            A[i + j * ldA] = 1;
        } else {
            U[i + j * ldU] = A[i + j * ldA];
            A[i + j * ldA] = 0;
        }
    }
}

template <typename T>
void launchKernel_moveU(dim3 gridDim, dim3 blockDim, int m, int n, T *A,
                        int ldA, T *U, int ldU) {
    moveU<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

template void launchKernel_moveU(dim3 gridDim, dim3 blockDim, int m, int n,
                                 double *A, int ldA, double *U, int ldU);

template void launchKernel_moveU(dim3 gridDim, dim3 blockDim, int m, int n,
                                 float *A, int ldA, float *U, int ldU);

template void launchKernel_moveU(dim3 gridDim, dim3 blockDim, int m, int n,
                                 half *A, int ldA, half *U, int ldU);

template <typename T>
__global__ void getLower(long m, long n, T *dA, long lda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        if (i < j) {
            dA[i + j * lda] = 0.0;
        } else if (i == j) {
            dA[i + j * lda] = 1.0;
        }

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

template <typename T>
void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, T *A,
                           long ldA) {
    getLower<<<gridDim, blockDim>>>(m, n, A, ldA);
}

template void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n,
                                    double *A, long ldA);
template void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n,
                                    float *A, long ldA);

template <typename T>
__global__ void copyLower(long m, long n, T *dA, long lda, T *dY, long ldy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        if (i < j) {
            dY[i + j * ldy] = 0.0;
        } else {
            dY[i + j * ldy] = dA[i + j * lda];
        }
    }
}

template <typename T>
void launchKernel_copyLower(dim3 gridDim, dim3 blockDim, long m, long n, T *A,
                            long ldA, T *Y, long ldy) {
    copyLower<<<gridDim, blockDim>>>(m, n, A, ldA, Y, ldy);
}

template void launchKernel_copyLower(dim3 gridDim, dim3 blockDim, long m,
                                     long n, double *A, long ldA, double *Y, long ldy);
template void launchKernel_copyLower(dim3 gridDim, dim3 blockDim, long m,
                                     long n, float *A, long ldA, float *Y, long ldy);

template <typename T>
__global__ void kernel_cpyATr2Vector(long m, long n, T *A, long ldA, T *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("come %d\n", __LINE__);
    // __syncthreads();
    if ((i < m) && (i < n)) {
        B[i] = A[i + i * ldA];
    }
}

template <typename T>
void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim, long m, long n,
                                 T *A, long ldA, T *B) {
    kernel_cpyATr2Vector<<<gridDim, blockDim>>>(m, n, A, ldA, B);
}
template void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim, long m,
                                          long n, double *A, long ldA,
                                          double *B);

template void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim, long m,
                                          long n, float *A, long ldA, float *B);

__global__ void scaleMatrixA(long m, long n, double *A, long ldA,
                             double scaler) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        A[i + j * ldA] *= scaler;

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_scaleMatrixA(dim3 gridDim, dim3 blockDim, long m, long n,
                               double *A, long ldA, double scaler) {
    scaleMatrixA<<<gridDim, blockDim>>>(m, n, A, ldA, scaler);
}

__global__ void findAbsMaxKernel(double *d_array, double *d_max, int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = abs(d_array[idx]);
    } else {
        // sdata[tid] = -INFINITY; // Ensure out of bounds values do not affect
        // max
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the maximum value for this block to the output array
    if (tid == 0) {
        d_max[blockIdx.x] = sdata[0];
    }
}

double findVectorAbsMax(double *d_array, int n) {
    double *d_max;
    double *h_max = new double[(n + 255) / 256];
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&d_max, blocksPerGrid * sizeof(double));

    findAbsMaxKernel<<<blocksPerGrid, threadsPerBlock,
                       threadsPerBlock * sizeof(double)>>>(d_array, d_max, n);

    cudaMemcpy(h_max, d_max, blocksPerGrid * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Perform final reduction on the CPU
    // double max_val = -INFINITY;
    double max_val = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        max_val = std::max(max_val, h_max[i]);
    }

    cudaFree(d_max);
    delete[] h_max;

    return max_val;
}