#include <cooperative_groups.h>
#include <mma.h>

#include "cusolver_utils.h"

// namespace cg = cooperative_groups;

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ void block_tcgemm(int mm, double *C, const int ldc, double *A,
                             const int lda, double *B, const int ldb,
                             int warp_liner_idx) {
    // if(threadIdx.x == 0 && threadIdx.y == 0) {
    //     for(int i = 0; i < 128; ++i) {
    //         for(int j = 0; j < 32; ++j) {
    //             double sum = 0;
    //             for(int k = 0; k < 32; ++k) {
    //                 sum += A[i + k * lda] * B[k + j * ldb];
    //             }
    //             C[i + j * ldc] = sum;
    //         }
    //     }
    // }
    //const int warp_row_idx = warp_liner_idx % 8;
    //const int warp_col_idx = warp_liner_idx / 8;
    //const int rowGemmNum = 2;
    //const int colGemmNum = 2;
    //double *warp_A, *warp_B ,*warp_C;
    //
    //nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double,
    //    nvcuda::wmma::col_major>
    //        a_frags;
    //nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double,
    //    nvcuda::wmma::col_major>
    //        b_frags;
    //nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>
    //    c_frag;

    //for(int i = 0; i < rowGemmNum; ++i) {
    //    if(warp_row_idx + i * 8 >= mm / 8)
    //        break;
    //    for(int j = 0; j < colGemmNum; ++j) {
    //        warp_C = &C[i * 64 + warp_row_idx * 8 + (j * 16 + warp_col_idx * 8) * ldc];
    //        nvcuda::wmma::fill_fragment(c_frag, static_cast<double>(0));
    //        for(int k = 0; k < (32 / 4); ++k) {
    //            warp_A = &A[i * 64 + warp_row_idx * 8 + (k * 4) * lda];
    //            warp_B = &B[k * 4 + (j * 16 + warp_col_idx * 8) * ldb];
    //            nvcuda::wmma::load_matrix_sync(a_frags, warp_A, lda);
    //            nvcuda::wmma::load_matrix_sync(b_frags, warp_B, ldb);
    //            nvcuda::wmma::mma_sync(c_frag, a_frags, b_frags, c_frag);
    //        }
    //        nvcuda::wmma::store_matrix_sync(warp_C, c_frag, ldc,
    //                nvcuda::wmma::mem_col_major);
    //    }
    //}
    const int rowNum = 4, colNum = 2;
    for(int i = 0; i < rowNum; ++i) {
        if(i * 32 + threadIdx.x >= mm)
            break;
        for(int j = 0; j < colNum; ++j) {
            double sum = 0;
            for(int k = 0; k < 32; ++k) {
                sum += A[i * 32 + threadIdx.x + k * lda] * B[k + (j * 16 + threadIdx.y) * ldb];
            }            
            C[i * 32 + threadIdx.x + (j * 16 + threadIdx.y) * ldc] = sum;
        }
    }
}

__device__ volatile int syncCounter = 0;

template <typename T, int M, int N>
__global__ void my_hou_kernel(const int m, const int n, T *A, const int lda,
                              T *R, const int ldr, T *work, const int ldwork) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    extern __shared__ T shared_A[];
    __shared__ T temp_A[M * N];
    const int ldaa = M;
    __shared__ T RR[N];
    __shared__ int shared_work_height[7];  // max reduction_time = 6
    __shared__ int idx;

    const int i = static_cast<int>(threadIdx.x);
    const int j = static_cast<int>(threadIdx.y);
    const int blockIdx_x = static_cast<int>(blockIdx.x);
    const int blockDim_x = static_cast<int>(blockDim.x);
    const int blockDim_y = static_cast<int>(blockDim.y);
    const int warp_liner_idx = (i + j * blockDim_x / 32);

    // cg::grid_group grid = cg::this_grid();

    if (i == 0 && j == 0) {
        if(blockIdx_x == 0) {
            syncCounter = 0;
        }
        idx = 0;
        shared_work_height[0] = m;
        // if (blockIdx_x == 0) {
        //     printf("shared_work_height[%d] = %d\n", idx,
        //     static_cast<int>(m));
        // }
    }

    __syncthreads();

    int nn = n;

    int numBlocks = 0;
    int endBlockNum = 0;

    while (shared_work_height[idx] > N) {
        int work_height = shared_work_height[idx];
        // int mm = min(work_height - blockIdx_x * M, M);
        int mm = ((work_height - blockIdx_x * M) < M) ? (work_height - blockIdx_x * M) : M;

        numBlocks = (work_height + M - 1) / M;
        endBlockNum = endBlockNum + numBlocks;

        // if(blockIdx_x == 1 && i == 0 && j == 0) {
        //     printf("idx: %d syncCounter: %d numBlocks: %d endBlockNum: %d\n",
        //     idx, syncCounter, numBlocks, endBlockNum);
        // }

        if (mm > 0) {
            // if (i == 0 && j == 0) {
            //     printf("idx = %d, blockIdx_x = %d, mm = %d\n", idx,
            //     blockIdx_x,
            //            mm);
            // }
            int rowDataNum = (mm + blockDim_x - 1) / blockDim_x;
            int colDataNum = (nn + blockDim_y - 1) / blockDim_y;
            T acc[4], q[4];  // 128 / 32 = 4

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
                __syncthreads();
                // if (blockIdx_x == 7 && i == 0 && j == 0) {
                //     printf("load from A[%d][%d] to shared_A[0]\n",
                //             blockIdx_x * M, 0);
                //     for (int v = 0; v < mm; v++) {
                //         for (int l = 0; l < 32; l++) {
                //             printf("%9.6f ", AA[v + l *
                //                     ldaa]);
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
                __syncthreads();
                // if (blockIdx_x == 7 && i == 0 && j == 0) {
                //     printf("load from work[%d][%d] to shared_A[%d]\n",
                //             blockIdx_x * M, 0, idx);
                //     for (int v = 0; v < mm; v++) {
                //         for (int l = 0; l < 32; l++) {
                //             printf("%9.6f ", AA[v + l *
                //                     ldaa]);
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

                    // if(blockIdx_x == 7 && idx == 2 && cols == 0) {
                    //     printf("x: %d y: %d nu: %9.3f, AA[%d][%d]: %9.3f\n", i, j, nu, i, cols, AA[i + cols * ldaa]);
                    // }

                    // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
                    T norm_x_squre = warpAllReduceSum(nu);
                    T norm_x = sqrt(norm_x_squre);

                    // 1、求u=x/norm(x);
                    T scale = 1.0 / norm_x;

                    // if(blockIdx_x == 7 && i == 0 && idx == 2 && cols == 0) {
                    //    printf("norm_x_squre: %9.3f, norm_x: %9.3f, scale: %9.3f\n", norm_x_squre, norm_x, scale);
                    // }

#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            AA[i + k * blockDim_x + cols * ldaa] *= scale;
                        }
                    }
                    __syncwarp();

                    // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0 && j == 0) {
                    //     printf("after %d times scale\n", cols);
                    //     for(int v = 0; v < mm; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", AA[v + l * ldaa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }

                    // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程;来计算即可
                    if (0 == i) {
                        T u1 = AA[cols + cols * mm];
                        AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;
                        // // 把normx存放到RR中，也就是对角线的元素
                        RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
                    }

                    // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0 && j == 0) {
                    //     printf("after %d set identity\n", cols);
                    //     for(int v = 0; v < mm; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", AA[v + l * ldaa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }
                    __syncwarp();

                    // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
                    scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));

                    // if(blockIdx_x == 7 && i == 0 && idx == 2 && cols == 0) {
                    //    printf("scale: %9.3f\n", scale);
                    // }

#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            AA[i + k * blockDim_x + cols * ldaa] *= scale;
                        }
                    }

                    // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0 && j == 0) {
                    //     printf("after %d times scale 2\n", cols);
                    //     for(int v = 0; v < mm; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", AA[v + l * ldaa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }
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

                        
                        // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0) {
                        //     printf("opcol: %d, utx: %9.6f\n", opCols, utx);
                        // }
                        // 计算x-uu'x
#pragma unroll
                        for (int k = 0; k < rowDataNum; k++) {
                            // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                            if (i + k * blockDim_x < mm &&
                                i + k * blockDim_x >= cols) {
                                AA[i + k * blockDim_x + opCols * ldaa] -=
                                    utx * AA[i + k * blockDim_x + cols * ldaa];
                                // if(idx == 2 && blockIdx_x == 7 && opCols == 1){
                                //     printf("update A[%d][%d] to %9.6f\n", i + k * blockDim_x, opCols, AA[i + k * blockDim_x + opCols * ldaa]);
                                // } 
                            }
                        }

                        __syncwarp();
                        // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0 && (opCols == 2)) {
                        //     printf("after update opCol: %d by %d using utx: %9.3f\n", opCols, cols, utx);
                        //     for(int v = 0; v < mm; ++v) {
                        //         for(int l = 0; l < n; ++l) {
                        //             printf("%9.3lf ", AA[v + l * ldaa]);
                        //         }
                        //         printf("\n");
                        //     }
                        // }

                    }
                }
                // if(cols == 0 && idx == 2 && blockIdx_x == 7 && i == 0 && j == 0) {
                //     printf("after %d all\n", cols);
                //     for(int v = 0; v < mm; ++v) {
                //         for(int l = 0; l < n; ++l) {
                //             printf("%9.3f ", AA[v + l * ldaa]);
                //         }
                //         printf("\n");
                //     }
                // }
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

            // if (blockIdx_x == 7 && i == 0 && j == 0) {
            //     if (work_height <= M) {
            //         printf("save to R\n");
            //     } else {
            //      printf("save to work[%d][%d]\n", blockIdx_x * N, 0);
            //     }
            //     for (int v = 0; v < 32; v++) {
            //         for (int l = 0; l < 32; l++) {
            //             printf("%9.6f ",
            //                     R_to[v + l * ldr_to]);
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

                // 4.把计算出来的q拷贝到 temp_A 中
                for (int k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        temp_A[i + k * blockDim_x + opCols * ldaa] = q[k];
                    }
                }
            }

            __syncthreads();

            for (int k = 0; k < rowDataNum; k++) {
                if (i + k * blockDim_x < mm) {
                    AA[i + k * blockDim_x + j * ldaa] =
                        temp_A[i + k * blockDim_x + j * ldaa];
                    AA[i + k * blockDim_x + (j + 16) * ldaa] =
                        temp_A[i + k * blockDim_x + (j + 16) * ldaa];
                }
            }

            __syncthreads();

            // if (blockIdx.x == 0 && i == 0 && j == 0) {
            //     printf("save to shared_A[%d]\n", idx);
            //     for (int v = 0; v < mm; v++) {
            //         for (int l = 0; l < 32; l++) {
            //             printf("%9.6f ", AA[v + l * ldaa]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            __threadfence();

            if (i == 0 && j == 0) {
                atomicAdd((int *)&syncCounter, 1);
                // printf("idx: %d blockIdx: %d Add syncCounter to %d (endBlockNum: %d)\n", idx, blockIdx_x, syncCounter,
                // endBlockNum);
            }
        }

        while (syncCounter < endBlockNum) {
            // printf("1 %d %d\n", syncCounter, endBlockNum);
        }

        if (i == 0 && j == 0) {
            work_height = ((work_height + M - 1) / M) * N;
            shared_work_height[++idx] = work_height;
            // if (blockIdx_x == 0) {
            //     printf("shared_work_height[%d] = %d\n", idx, work_height);
            // }
        }

        __syncthreads();
        // grid.sync();
    }

    if (i == 0 && j == 0) {
        idx -= 1;
    }

    __syncthreads();

    // perform tensorcore gemm to obtain final Q
    while (idx >= 0) {
        int work_height = shared_work_height[idx];
        // int mm = min(work_height - blockIdx_x * M, M);
        int mm = ((work_height - blockIdx_x * M) < M) ? (work_height - blockIdx_x * M) : M;

        numBlocks = (work_height + M - 1) / M;
        endBlockNum = endBlockNum - numBlocks * 2;

        if (mm > 0) {
            // if (i == 0 && j == 0) {
            //     printf("idx = %d, work_height = %d, blockidx = %d\n", idx,
            //            work_height, blockIdx_x);
            // }
            const int rowDataNumA = (mm + blockDim_x - 1) / blockDim_x;
            const int colDataNumA = (N + blockDim_y - 1) / blockDim_y;

            T *work_q_next = &work[blockIdx_x * N];
            T *q_this = &shared_A[idx * N * ldaa];
            T *work_q_to = &work[blockIdx_x * M];

            if (work_height > M) {
                // if (i == 0 && j == 0) {
                //     printf(
                //             "blockidx = %d, gemm: shared_A[%d] * work[%d] -> shared_A[%d]\n", blockIdx_x, idx, blockIdx_x * N, idx);
                // }

                __syncthreads();

                // if(blockIdx_x == 0 && i == 0 && j == 0) {
                //     printf("A\n");
                //     for(int v = 0; v < mm; ++v) {
                //         for(int l = 0; l < 32; ++l) {
                //             printf("%9.6f ", q_this[v + l * ldaa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }

                // if(blockIdx_x == 0 && i == 0 && j == 0) {
                //     printf("B\n");
                //     for(int v = 0; v < 32; ++v) {
                //         for(int l = 0; l < 32; ++l) {
                //             printf("%9.6f ", work_q_next[v + l * ldwork]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
                __threadfence();
                __syncthreads();
                if (i == 0 && j == 0) {
                    atomicAdd((int *)&syncCounter, -1);
                    // printf("idx: %d blockIdx: %d Add syncCounter to %d (endBlockNum: %d)\n", idx, blockIdx_x, syncCounter,
                    // endBlockNum);
                }
                while (syncCounter > endBlockNum + numBlocks) {
                    // printf("2 %d %d\n", syncCounter, endBlockNum);
                }

                block_tcgemm(mm, temp_A, ldaa, q_this, ldaa, work_q_next,
                             ldwork, warp_liner_idx);

                __threadfence();
                __syncthreads();
                if (i == 0 && j == 0) {
                    atomicAdd((int *)&syncCounter, -1);
                    // printf("idx: %d blockIdx: %d Add syncCounter to %d (endBlockNum: %d)\n", idx, blockIdx_x, syncCounter,
                    // endBlockNum);
                }
                while (syncCounter > endBlockNum) {
                    // printf("3 %d %d\n", syncCounter, endBlockNum);
                }

                // if(blockIdx_x == 0 && i == 0 && j == 0) {
                //     printf("C\n");
                //     for(int v = 0; v < mm; ++v) {
                //         for(int l = 0; l < N; ++l) {
                //             printf("%9.6f ", temp_A[v + l * ldaa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }

                for (int row_load_idx = 0; row_load_idx < rowDataNumA;
                        row_load_idx++) {
                    for (int col_load_idx = 0; col_load_idx < colDataNumA;
                            col_load_idx++) {
                        int row_idx = i + row_load_idx * blockDim_x;
                        int col_idx = j + col_load_idx * blockDim_y;
                        if (row_idx < mm) {
                            work_q_to[row_idx + col_idx * ldwork] =
                                temp_A[row_idx + col_idx * ldaa];
                        }
                    }
                }
            } else { // last 128 size block in A
                __threadfence();
                if (i == 0 && j == 0) {
                    atomicAdd((int *)&syncCounter, -2);
                    // printf("idx: %d blockIdx: %d Add syncCounter to %d (endBlockNum: %d)\n", idx, blockIdx_x, syncCounter,
                    // endBlockNum);
                }
                while (syncCounter > endBlockNum) {
                    // printf("4 %d %d\n", syncCounter, endBlockNum);
                }

                // if (i == 0 && j == 0) {
                //     printf("blockidx = %d, move shared_A[%d] -> work[%d]\n",
                //            blockIdx_x, idx, blockIdx_x * M);
                // }
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
        } else { // if mm > 0
            while (syncCounter > endBlockNum) {
                // printf("5 %d %d\n", syncCounter, endBlockNum);
            }
        }
        
        
        if (i == 0 && j == 0) {
            idx--;
        }

        // grid.sync();
        __syncthreads();
    }

    // int mm = min(m - blockIdx_x * M, M);
    // int rowDataNum = (mm + blockDim_x - 1) / blockDim_x;
    // for (int k = 0; k < rowDataNum; k++) {
    //     if (i + k * blockDim_x < mm) {
    //         A[blockIdx_x * M + i + k * blockDim_x + j * lda] =
    //             work[i + k * blockDim_x + j * ldwork];
    //         A[blockIdx_x * M + i + k * blockDim_x + (j + 16) * lda] =
    //             work[i + k * blockDim_x + (j + 16) * ldwork];
    //     }
    // }
}

// template __global__ void my_hou_kernel<float, 128, 32>(const int m, const int
// n,
//         float *A, const int lda,
//         float *R, const int ldr,
//         float *work,
//         const int ldwork);
template __global__ void my_hou_kernel<double, 128, 32>(
    const int m, const int n, double *A, const int lda, double *R,
    const int ldr, double *work, const int ldwork);
