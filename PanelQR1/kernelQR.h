#include <cooperative_groups.h>
#include <mma.h>

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

__device__ void block_tcgemm(int mm, float *C, const int ldc, __half *A,
                             const int lda, __half *B, const int ldb,
                             int warp_liner_idx) {
    const int warp_row_idx = warp_liner_idx % 8;
    const int warp_col_idx = warp_liner_idx / 8;
    __half *warp_A, *warp_B;

    if (warp_row_idx < mm / 16) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                               nvcuda::wmma::col_major>
            a_frags;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                               nvcuda::wmma::col_major>
            b_frags;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            c_frag;

        nvcuda::wmma::fill_fragment(c_frag, static_cast<float>(0));

        for (int i = 0; i < (32 / 16); ++i) {
            warp_A = &A[(warp_row_idx * 16) + (i * 16) * lda];
            warp_B = &B[(i * 16) + (warp_col_idx * 16) * ldb];

            nvcuda::wmma::load_matrix_sync(a_frags, warp_A, lda);
            nvcuda::wmma::load_matrix_sync(b_frags, warp_B, ldb);
            nvcuda::wmma::mma_sync(c_frag, a_frags, b_frags, c_frag);
            nvcuda::wmma::store_matrix_sync(C, c_frag, ldc,
                                            nvcuda::wmma::mem_col_major);
        }
    }
}

template <typename T, int M, int N>
__global__ void my_hou_kernel(const int m, const int n, T *A, const int lda,
                              T *R, const int ldr, T *work, const int ldwork) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    __shared__ __half shared_A_half[M * N];
    __shared__ __half shared_B_half[N * N];
    extern __shared__ T shared_A[];
    const int ldaa = M;
    __shared__ T RR[N];
    __shared__ int
        shared_work_height[7];  // maximux size of m is 32 * ((128 / 32) ^ (7 -
                                // 1)) = 131,072 , max reduction_time = 6
    __shared__ int idx;
    T acc[4], q[4];  // 128 / 32 = 4
    const int lda_half = M, ldb_half = N;

    // 2. 找到本线程的ID
    const int i = static_cast<int>(threadIdx.x);
    const int j = static_cast<int>(threadIdx.y);
    const int blockIdx_x = static_cast<int>(blockIdx.x);
    const int blockDim_x = static_cast<int>(blockDim.x);
    const int blockDim_y = static_cast<int>(blockDim.y);
    const int warp_liner_idx = (i + j * blockDim_x / 32);

    cg::grid_group grid = cg::this_grid();

    if (i == 0 && j == 0) {
        idx = 0;
        shared_work_height[0] = static_cast<int>(m);
        // if (blockIdx_x == 0) {
        //     printf("shared_work_height[%d] = %d\n", idx,
        //     static_cast<int>(m));
        // }
    }

    int nn = static_cast<int>(n);

    __syncthreads();

    while (shared_work_height[idx] > N) {
        int work_height = shared_work_height[idx];
        int mm = min(work_height - static_cast<int>(blockIdx_x * M),
                     static_cast<int>(M));

        if (mm > 0) {
            // if (i == 0 && j == 0) {
            //     printf("idx = %d, blockIdx_x = %d, mm = %d\n", idx,
            //     blockIdx_x,
            //            mm);
            // }
            int rowDataNum = (mm + blockDim_x - 1) / blockDim_x;
            int colDataNum = (nn + blockDim_y - 1) / blockDim_y;

            T *AA = &shared_A[idx * N * ldaa];

            if (idx == 0) {
                // 假定n=N=32，每一个线程拷贝2列
                for (int k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + j * ldaa] =
                            A[blockIdx_x * M + i + k * blockDim_x + j * lda];
                        AA[i + k * blockDim_x + (j + 16) * ldaa] =
                            A[blockIdx_x * M + i + k * blockDim_x +
                              (j + 16) * lda];
                    }
                }
                // __syncthreads();
                // if (blockIdx_x == 0 && i == 0 && j == 0) {
                //     printf("load from A[%d][%d] to shared_A[0]\n",
                //            blockIdx_x * M, 0);
                //     for (int v = 0; v < 36; v++) {
                //         for (int l = 0; l < 4; l++) {
                //             printf("AA[%d][%d] = %f ", v, l, AA[v + l * ldaa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
            } else {
                // 假定n=N=32，每一个线程拷贝2列
                for (int k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + j * ldaa] =
                            work[blockIdx_x * M + i + k * blockDim_x +
                                 j * ldwork];
                        AA[i + k * blockDim_x + (j + 16) * ldaa] =
                            work[blockIdx_x * M + i + k * blockDim_x +
                                 (j + 16) * ldwork];
                    }
                }
                // __syncthreads();
                // if (blockIdx.x == 0 && i == 0 && j == 0) {
                //     printf("load from work[%d][%d] to shared_A[%d]\n",
                //            blockIdx_x * M, 0, idx);
                //     for (int v = 0; v < 36; v++) {
                //         for (int l = 0; l < 4; l++) {
                //             printf("AA[%d][%d] = %f ", v, l, AA[v + l * ldaa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
            }

            __syncthreads();

            // 进行HouseHolder分解，先计算HouseHolder向量
            // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)=
            // u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
            for (int cols = 0; cols < nn; cols++) {
                // 先计算HouseHolder向量
                // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)=
                // u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
                T nu = 0.0;
                if (j == cols % blockDim_y) {
                    // 0.求normx
                    // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        acc[k] = 0.0;
                        // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            acc[k] = AA[i + k * blockDim_x + cols * ldaa] *
                                     AA[i + k * blockDim_x + cols * ldaa];
                        }
                        nu += acc[k];
                    }

                    // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
                    T norm_x_squre = warpAllReduceSum(nu);
                    T norm_x = sqrt(norm_x_squre);

                    // if(blockIdx_x == 0 && i == 0 && j == 0) {
                    //     printf("norm_x_squre: %f ", norm_x_squre);
                    //     printf("\n");
                    // }

                    // 1、求u=x/norm(x);
                    T scale = 1.0 / norm_x;
#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            AA[i + k * blockDim_x + cols * ldaa] *= scale;
                        }
                    }
                    __syncwarp();

                    // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
                    if (0 == i) {
                        T u1 = AA[cols + cols * mm];
                        AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;
                        // // 把normx存放到RR中，也就是对角线的元素
                        RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
                    }

                    __syncwarp();

                    // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
                    scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            AA[i + k * blockDim_x + cols * ldaa] *= scale;
                        }
                    }
                }

                __syncthreads();
                // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
                // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
                // 每个线程按列需要处理多个列
                for (int h = 0; h < colDataNum; h++) {
                    T nu = 0.0;
                    int opCols = j + h * blockDim_y;

                    // 只更新当前列后面的列
                    if (cols < opCols && opCols < nn) {
                        // 先计算u'x
#pragma unroll
                        for (int k = 0; k < rowDataNum; k++) {
                            acc[k] = 0.0;
                            // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                            if (i + k * blockDim_x < mm &&
                                i + k * blockDim_x >= cols) {
                                acc[k] = AA[i + k * blockDim_x + cols * ldaa] *
                                         AA[i + k * blockDim_x + opCols * ldaa];
                            }
                            nu += acc[k];
                        }
                        T utx = warpAllReduceSum(nu);

                        // 计算x-uu'x
#pragma unroll
                        for (int k = 0; k < rowDataNum; k++) {
                            // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                            if (i + k * blockDim_x < mm &&
                                i + k * blockDim_x >= cols) {
                                AA[i + k * blockDim_x + opCols * ldaa] -=
                                    utx * AA[i + k * blockDim_x + cols * ldaa];
                            }
                        }
                        __syncwarp();
                    }
                }
            }

            T *R_to;
            int ldr_to;
            if (work_height <= M) {
                R_to = R;
                ldr_to = ldr;
            } else {
                R_to = &work[blockIdx_x * N];
                ldr_to = ldwork;
            }

            __syncthreads();

            // 获得R矩阵，将AA的上三角部分拷贝到R中
            // 以R矩阵来进行循环
            int rRowDataNum = (nn + (blockDim_x - 1)) / blockDim_x;
            for (int h = 0; h < colDataNum; h++) {
                int opCols = j + h * blockDim_y;
                if (opCols >= nn) continue;
#pragma unroll
                for (int k = 0; k < rRowDataNum; k++) {
                    if (i + k * blockDim_x < opCols) {
                        R_to[i + k * blockDim_x + opCols * ldr_to] =
                            AA[i + k * blockDim_x + opCols * ldaa];
                        AA[i + k * blockDim_x + opCols * ldaa] = 0.0;
                    } else if (i + k * blockDim_x > opCols) {
                        R_to[i + k * blockDim_x + opCols * ldr_to] = 0.0;
                    } else {
                        // 这个赋值完全可以放到上面RR的赋值哪儿，从而不需要RR的共享内存
                        R_to[opCols + opCols * ldr_to] = RR[opCols];
                    }
                }
            }

            __syncthreads();

            // if (blockIdx.x == 0 && i == 0 && j == 0) {
            //     if (work_height <= M) {
            //         printf("save to R\n");
            //     } else {
            //         printf("save to work[%d][%d]\n", blockIdx_x * N, 0);
            //     }
            //     for (int v = 0; v < 32; v++) {
            //         for (int l = 0; l < 4; l++) {
            //             printf("R_to[%d][%d] = %f ", v, l,
            //                    R_to[v + l * ldr_to]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            // 来求Q，使用的方法是Q=(I-uu')Q,
            // 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q q表示是Q矩阵的1列
            // double q[rowDataNum * 2];
            for (int h = 0; h < colDataNum; h++) {
                // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
                int opCols = j + h * blockDim_y;

                if (opCols >= nn) continue;

                for (int k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x == opCols) {
                        q[k] = 1.0;
                    } else {
                        q[k] = 0.0;
                    }
                }

                __syncwarp();

                for (int cols = nn - 1; cols >= 0; cols--) {
                    // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
                    // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
                    // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
                    if (opCols >= cols) {
                        // 2、计算u'q
                        T nu = 0.0;
                        for (int k = 0; k < rowDataNum; k++) {
                            acc[k] = 0.0;
                            if (i + k * blockDim_x < mm) {
                                acc[k] =
                                    AA[i + k * blockDim_x + cols * ldaa] * q[k];
                            }
                            nu += acc[k];
                        }

                        T utq = warpAllReduceSum(nu);

                        // 3.计算q-uu'q
                        for (int k = 0; k < rowDataNum; k++) {
                            if (i + k * blockDim_x < mm) {
                                q[k] -=
                                    utq * AA[i + k * blockDim_x + cols * ldaa];
                            }
                        }

                        __syncwarp();
                    }
                }

                // 4.把计算出来的q拷贝到A中
                for (int k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + opCols * ldaa] = q[k];
                    }
                }
            }
        }

        if (i == 0 && j == 0) {
            work_height = ((work_height + M - 1) / M) * N;
            shared_work_height[++idx] = work_height;
            // if (blockIdx_x == 0) {
            //     printf("shared_work_height[%d] = %d\n", idx, work_height);
            // }
        }

        // __syncthreads();
        grid.sync();
    }

    if (i == 0 && j == 0) {
        idx -= 1;
    }

    while (true) {
        __syncthreads();

        if (idx < 0) {
            break;
        }

        int work_height = shared_work_height[idx];
        int mm = min(work_height - static_cast<int>(blockIdx_x * M),
                     static_cast<int>(M));

        if (mm > 0) {
            const int rowDataNumA = (mm + blockDim_x - 1) / blockDim_x;
            const int colDataNumA = (N + blockDim_y - 1) / blockDim_y;
            const int rowDataNumB = (N + blockDim_x - 1) / blockDim_x;
            const int colDataNumB = colDataNumA;

            T *work_q_next = &work[blockIdx_x * N];
            T *q_this = &shared_A[idx * N * ldaa];
            T *work_q_to = &work[blockIdx_x * M];

            if (work_height > M) {
                for (int row_load_idx = 0; row_load_idx < rowDataNumA;
                     row_load_idx++) {
                    for (int col_load_idx = 0; col_load_idx < colDataNumA;
                         col_load_idx++) {
                        int row_idx = i + row_load_idx * blockDim_x;
                        int col_idx = j + col_load_idx * blockDim_y;
                        if (row_idx < mm) {
                            shared_A_half[row_idx + col_idx * lda_half] =
                                __float2half(q_this[row_idx + col_idx * ldaa]);
                        }
                    }
                }

                for (int row_load_idx = 0; row_load_idx < rowDataNumB;
                     row_load_idx++) {
                    for (int col_load_idx = 0; col_load_idx < colDataNumB;
                         col_load_idx++) {
                        int row_idx = i + row_load_idx * blockDim_x;
                        int col_idx = j + col_load_idx * blockDim_y;
                        shared_B_half[row_idx + col_idx * ldb_half] =
                            __float2half(
                                work_q_next[row_idx + col_idx * ldwork]);
                    }
                }

                __syncthreads();
                block_tcgemm(mm, q_this, ldaa, shared_A_half, lda_half,
                             shared_B_half, ldb_half, warp_liner_idx);
            }
            if (idx > 0) {
                for (int row_load_idx = 0; row_load_idx < rowDataNumA;
                     row_load_idx++) {
                    for (int col_load_idx = 0; col_load_idx < colDataNumA;
                         col_load_idx++) {
                        int row_idx = i + row_load_idx * blockDim_x;
                        int col_idx = j + col_load_idx * blockDim_y;
                        if (row_idx < mm) {
                            work_q_to[row_idx + col_idx * ldwork] =
                                q_this[row_idx + col_idx * ldaa];
                        }
                    }
                }
            }
        }

        if (i == 0 && j == 0) {
            idx--;
        }
    }

    // copy shared_A[0] to A
    int mm = min(m - static_cast<int>(blockIdx_x * M), static_cast<int>(M));
    T *AA = &shared_A[0];
    int rowDataNum = (mm + blockDim_x - 1) / blockDim_x;
    for (int k = 0; k < rowDataNum; k++) {
        if (i + k * blockDim_x < mm) {
            A[blockIdx_x * M + i + k * blockDim_x + j * lda] =
                AA[i + k * blockDim_x + j * ldaa];
            A[blockIdx_x * M + i + k * blockDim_x + (j + 16) * lda] =
                AA[i + k * blockDim_x + (j + 16) * ldaa];
        }
    }
}

template __global__ void my_hou_kernel<float, 128, 32>(const int m, const int n,
                                                       float *A, const int lda,
                                                       float *R, const int ldr,
                                                       float *work,
                                                       const int ldwork);
