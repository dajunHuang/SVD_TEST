#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include "cusolver_utils.h"
namespace cg = cooperative_groups;

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__device__ void householder_QR(const size_t m, const size_t n, T *A,
                               const size_t lda, T *R, const size_t ldr, T *RR,
                               T *acc, T *q, const unsigned int threadIdx_x,
                               const unsigned int threadIdx_y,
                               const unsigned int blockDim_x) {
    size_t rowDataNum{(m + blockDim_x - 1) / blockDim_x};
    const size_t op_col{threadIdx_y};
    // T acc[rowDataNum] = {0};
    T nu = 0;

    for (size_t col{0}; col < n; col++) {
        nu = 0;
        if (col == threadIdx_y) {
            for (size_t i{0}; i < rowDataNum; ++i) {
                acc[i] = 0;
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m && idx_x >= col) {
                    acc[i] = A[idx_x + col * lda] * A[idx_x + col * lda];
                }
                nu += acc[i];
            }

            T norm_x{sqrt(warpAllReduceSum(nu))};
            T scale{static_cast<T>(1.0) / norm_x};

            for (size_t i{0}; i < rowDataNum; ++i) {
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m && idx_x >= col) {
                    A[idx_x + col * lda] *= scale;
                }
            }

            __syncwarp();

            if (threadIdx_x == 0) {
                T u1 = A[col + col * lda];
                A[col + col * lda] += (u1 >= 0) ? 1 : -1;
                RR[col] = (u1 >= 0) ? -norm_x : norm_x;
            }

            __syncwarp();

            scale = 1 / (sqrt(abs(A[col + col * lda])));

            for (size_t i{0}; i < rowDataNum; ++i) {
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m && idx_x >= col) {
                    A[idx_x + col * lda] *= scale;
                }
            }
        }

        __syncthreads();

        nu = 0;

        if (op_col > col) {
            for (size_t i{0}; i < rowDataNum; ++i) {
                acc[i] = 0;
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m && idx_x >= col) {
                    acc[i] = A[idx_x + col * lda] * A[idx_x + col * lda];
                }
                nu += acc[i];
            }

            T utx = warpAllReduceSum(nu);

            for (size_t i{0}; i < rowDataNum; ++i) {
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m && idx_x >= col) {
                    A[idx_x + op_col * lda] -= utx * A[idx_x + col * lda];
                }
            }

            __syncwarp();
        }
    }

    __syncthreads();

    if (threadIdx_x < threadIdx_y) {
        R[threadIdx_x + threadIdx_y * ldr] = A[threadIdx_x + threadIdx_y * lda];
    } else if (threadIdx_x > threadIdx_y) {
        R[threadIdx_x + threadIdx_y * ldr] = 0;
    } else {
        R[threadIdx_x + threadIdx_y * ldr] = RR[threadIdx_y];
    }

    for (size_t i{0}; i < rowDataNum; ++i) {
        size_t idx_x{threadIdx_x + i * blockDim_x};
        if (idx_x == op_col) {
            q[i] = 1;
        } else {
            q[i] = 0;
        }
    }

    __syncwarp();

    for (int col{static_cast<int>(n) - 1}; col >= 0; --col) {
        if (op_col >= col) {
            nu = 0;
            for (size_t i{0}; i < rowDataNum; ++i) {
                acc[i] = 0;
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m) {
                    acc[i] = A[idx_x + col * lda] * q[i];
                }
                nu += acc[i];
            }

            T utq = warpAllReduceSum(nu);

            for (size_t i{0}; i < rowDataNum; ++i) {
                size_t idx_x{threadIdx_x + i * blockDim_x};
                if (idx_x < m) {
                    q[i] -= utq * A[idx_x + col * lda];
                }
            }

            __syncwarp();
        }
    }

    for (size_t i{0}; i < rowDataNum; ++i) {
        size_t idx_x{threadIdx_x + i * blockDim_x};
        if (idx_x < m) {
            A[idx_x + op_col * lda] = q[i];
        }
    }
}

template __device__ void householder_QR<float>(const size_t m, const size_t n,
                                               float *A, const size_t lda,
                                               float *R, const size_t ldr,
                                               float *RR, float *acc, float *q,
                                               const unsigned int threadIdx_x,
                                               const unsigned int threadIdx_y,
                                               const unsigned int blockDim_x);
// template __device__ void householder_QR<double>(
//     const size_t m, const size_t n, double *A, const size_t lda, double *R,
//     const size_t ldr, double *RR, const unsigned int threadIdx_x,
//     const unsigned int threadIdx_y, const unsigned int blockDim_x);

// change type
template <typename T, size_t M, size_t N>
__global__ void QR_kernel(const size_t m, const size_t n, const T *A,
                          const size_t lda, T *Y, const size_t ldy, T *R,
                          const size_t ldr, T *work, const size_t ldwork) {
    const size_t ldsa{M}, ldsr{N};
    extern __shared__ T shared_A[];
    __shared__ T shared_RR[N];
    __shared__ int shared_work_height[3];
    __shared__ int idx;
    // T acc[rowDataNum] = {0};
    T acc[4] = {0};
    // T q[rowDataNum] = {0};
    T q[4] = {0};

    const unsigned int blockIdx_x{blockIdx.x};
    const unsigned int threadIdx_x{threadIdx.x};
    const unsigned int threadIdx_y{threadIdx.y};
    const unsigned int blockDim_x{blockDim.x};

    int mm{min(static_cast<int>(m - blockIdx_x * M), static_cast<int>(M))};
    size_t rowDataNum{(mm + blockDim_x - 1) / blockDim_x};
    // if (threadIdx_x == 0 && threadIdx_y == 0) {
    //     printf("blockIdx_x: %u mm: %d rowDataNum: %lu\n", blockIdx_x, mm,
    //            rowDataNum);
    // }
    T *A_in_share = &shared_A[0];

    for (size_t i{0}; i < rowDataNum; ++i) {
        if (threadIdx_x + i * blockDim_x < mm) {
            A_in_share[threadIdx_x + i * blockDim_x + threadIdx_y * ldsa] =
                A[blockIdx_x * M + threadIdx_x + i * blockDim_x +
                  threadIdx_y * lda];
        }
    }

    __syncthreads();

    // if (threadIdx_x == 0 && threadIdx_y == 0) {
    //     printf(
    //         "mm: %d n: %lu ldsa: %lu ldwork: %lu threadIdx_x: %u threadIdx_y:
    //         "
    //         "%u blockDim_x: %u\n",
    //         mm, n, ldsa, ldwork, threadIdx_x, threadIdx_y, blockDim_x);
    // }
    householder_QR<T>(mm, n, A_in_share, ldsa, &work[blockIdx_x * N], ldwork,
                      &shared_RR[0], &acc[0], &q[0], threadIdx_x, threadIdx_y,
                      blockDim_x);

    int work_height = ((m + M - 1) / M) * N;
    if (threadIdx_x == 0 && threadIdx_y == 0) {
        idx = 0;
        // if (blockIdx_x == 0) {
        //     printf("shared_work_height[%d] = %d\n", idx, work_height);
        // }
        shared_work_height[idx++] = work_height;
    }

    while (work_height > M) {
        mm = min(static_cast<int>(work_height - blockIdx_x * M),
                 static_cast<int>(M));

        if (mm > 0) {
            rowDataNum = (mm + blockDim_x - 1) / blockDim_x;
            A_in_share = &shared_A[idx * N * ldsa];

            for (size_t i{0}; i < rowDataNum; ++i) {
                if (threadIdx_x + i * blockDim_x < mm) {
                    A_in_share[threadIdx_x + i * blockDim_x +
                               threadIdx_y * ldsa] =
                        work[blockIdx_x * M + threadIdx_x + i * blockDim_x +
                             threadIdx_y * ldwork];
                }
            }

            __syncthreads();

            // if (blockIdx_x == 1 && threadIdx_x == 0 && threadIdx_y == 0) {
            //     printf(
            //         "idx: %d mm: %d n: %lu ldsa: %lu ldwork: %lu threadIdx_x:
            //         %u " "threadIdx_y: %u blockDim_x: %u\n", idx, mm, n,
            //         ldsa, ldwork, threadIdx_x, threadIdx_y, blockDim_x);
            // }
            // householder_QR<T>(mm, n, A_in_share, ldsa, work + blockIdx_x * N,
            //                   ldwork, &shared_RR[0], &acc[0], &q[0],
            //                   threadIdx_x, threadIdx_y, blockDim_x);
        }

        work_height = ((work_height + M - 1) / M) * N;
        if (threadIdx_x == 0 && threadIdx_y == 0) {
            // if (blockIdx_x == 0) {
            //     printf("shared_work_height[%d] = %d\n", idx, work_height);
            // }
            shared_work_height[idx++] = work_height;
        }
    }

    // 最后一个 QR 计算
}

template __global__ void QR_kernel<float, 128, 32>(
    const size_t m, const size_t n, const float *A, const size_t lda, float *Y,
    const size_t ldy, float *R, const size_t ldr, float *work,
    const size_t ldwork);
// template __global__ void QR_kernel<double, 128, 32, 3>(
//     const size_t m, const size_t n, const double *A, const size_t lda,
//     double *Y, const size_t ldy, double *R, const size_t ldr, double *work,
//     const size_t ldwork);
