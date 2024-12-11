#include <cooperative_groups.h>
#include <mma.h>

#include "cusolver_utils.h"

// namespace cg = cooperative_groups;

#pragma once
template <typename T>
static __inline__ __device__ T warp_all_reduce_sum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ void block_gemm(int m, int n, double *C, const int ldc, double *A,
                           const int lda, double *B, const int ldb) {
    // const int warp_row_idx = warp_liner_idx % 8;
    // const int warp_col_idx = warp_liner_idx / 8;
    // const int rowGemmNum = 2;
    // const int colGemmNum = 2;
    // double *warp_A, *warp_B ,*warp_C;
    //
    // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double,
    //     nvcuda::wmma::col_major>
    //         a_frags;
    // nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double,
    //     nvcuda::wmma::col_major>
    //         b_frags;
    // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>
    //     c_frag;

    // for(int thread_idx_x = 0; thread_idx_x < rowGemmNum; ++thread_idx_x) {
    //     if(warp_row_idx + thread_idx_x * 8 >= block_data_height / 8)
    //         break;
    //     for(int thread_idx_y = 0; thread_idx_y < colGemmNum; ++thread_idx_y)
    //     {
    //         warp_C = &C[thread_idx_x * 64 + warp_row_idx * 8 + (thread_idx_y
    //         * 16 + warp_col_idx * 8) * ldc];
    //         nvcuda::wmma::fill_fragment(c_frag, static_cast<double>(0));
    //         for(int k = 0; k < (32 / 4); ++k) {
    //             warp_A = &A[thread_idx_x * 64 + warp_row_idx * 8 + (k * 4) *
    //             lda]; warp_B = &B[k * 4 + (thread_idx_y * 16 + warp_col_idx *
    //             8) * ldb]; nvcuda::wmma::load_matrix_sync(a_frags, warp_A,
    //             lda); nvcuda::wmma::load_matrix_sync(b_frags, warp_B, ldb);
    //             nvcuda::wmma::mma_sync(c_frag, a_frags, b_frags, c_frag);
    //         }
    //         nvcuda::wmma::store_matrix_sync(warp_C, c_frag, ldc,
    //                 nvcuda::wmma::mem_col_major);
    //     }
    // }
    const int block_dim_x = blockDim.x, block_dim_y = blockDim.y;
    const int thread_idx_x = threadIdx.x, thread_idx_y = threadIdx.y;
    const int num_row = (m + block_dim_x - 1) / block_dim_x;
    const int num_col = (n + block_dim_y - 1) / block_dim_y;
    for (int row_repeat_idx = 0; row_repeat_idx < num_row; ++row_repeat_idx) {
        int row_idx = row_repeat_idx * block_dim_x + thread_idx_x;
        if (row_idx >= m) break;
        for (int col_repeat_idx = 0; col_repeat_idx < num_col;
             ++col_repeat_idx) {
            int col_idx = col_repeat_idx * block_dim_y + thread_idx_y;
            if (col_idx >= n) break;
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[row_idx + k * lda] * B[k + col_idx * ldb];
            }
            C[row_idx + col_idx * ldc] = sum;
        }
    }
}

__device__ volatile int sync_counter = 0;

template <typename T>
__global__ void my_hou_kernel(const int block_size, const int m, const int n,
                              T *A, const int lda, T *R, const int ldr, T *work,
                              const int ldwork) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    extern __shared__ T all_shared_A[];
    __shared__ T RR[128];                       // n <= 128
    __shared__ int shared_all_data_height[16];  // reduction_time < 16
    __shared__ int reduction_time;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;
    const int block_dim_x = blockDim.x;
    const int block_dim_y = blockDim.y;

    if (thread_idx_x == 0 && thread_idx_y == 0) {
        if (block_idx_x == 0) {
            sync_counter = 0;
        }
        reduction_time = 0;
        shared_all_data_height[0] = m;
        // if (block_idx_x == 0) {
        //     printf("shared_all_data_height[%d] = %d\n", reduction_time,
        //     static_cast<int>(m));
        // }
    }

    __syncthreads();

    const int ldsa = block_size;
    int num_reduction_block = 0;
    int count_end_block = 0;

    while (shared_all_data_height[reduction_time] > n) {
        int all_data_height = shared_all_data_height[reduction_time];
        int block_data_height =
            min(all_data_height - block_idx_x * block_size, block_size);

        num_reduction_block = (all_data_height + block_size - 1) / block_size;
        count_end_block += num_reduction_block;

        // if(block_idx_x == 1 && thread_idx_x == 0 && thread_idx_y == 0) {
        //     printf("reduction_time: %d sync_counter: %d num_reduction_block:
        //     %d count_end_block: %d\n", reduction_time, sync_counter,
        //     num_reduction_block, count_end_block);
        // }

        if (block_data_height > 0) {
            // if (thread_idx_x == 0 && thread_idx_y == 0) {
            //     printf("reduction_time = %d, block_idx_x = %d,
            //     block_data_height = %d\n", reduction_time, block_idx_x,
            //            block_data_height);
            // }
            int num_data_row =
                (block_data_height + block_dim_x - 1) / block_dim_x;
            int num_data_col = (n + block_dim_y - 1) / block_dim_y;
            T accumulate[4], q_per_thread[8];  // 128 / 32 * 2 = 8

            T *shared_A = &all_shared_A[reduction_time * n * ldsa];

            for (int k = 0; k < num_data_row; k++) {
                int row_idx = thread_idx_x + k * block_dim_x;
                if (row_idx < block_data_height) {
                    for (int h = 0; h < num_data_col; ++h) {
                        int col_idx = thread_idx_y + h * block_dim_y;
                        if (col_idx < n) {
                            shared_A[row_idx + col_idx * ldsa] =
                                A[block_idx_x * block_size + row_idx +
                                  col_idx * lda];
                        }
                    }
                }
            }
            __syncthreads();
            // if (block_idx_x == 7 && thread_idx_x == 0 && thread_idx_y == 0) {
            //     printf("load from A[%d][%d] to shared_A[0]\n",
            //             block_idx_x * block_size, 0);
            //     for (int v = 0; v < block_data_height; v++) {
            //         for (int l = 0; l < 32; l++) {
            //             printf("%9.6f ", shared_A[v + l *
            //                     ldsa]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            __syncthreads();

            // 进行HouseHolder分解，先计算HouseHolder向量
            // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)=
            // u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
            for (int cols = 0; cols < n; cols++) {
                // 先计算HouseHolder向量
                // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)=
                // u(1)+sign(u(1)); 3、u=u/sqrt(abs(u(1)))
                T nu = 0.0;
                if (thread_idx_y == cols % block_dim_y) {
                    // 0.求normx
                    // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
                    for (int k = 0; k < num_data_row; k++) {
                        accumulate[k] = 0.0;
                        int row_idx = thread_idx_x + k * block_dim_x;
                        // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                        if (row_idx < block_data_height && row_idx >= cols) {
                            accumulate[k] = shared_A[row_idx + cols * ldsa] *
                                            shared_A[row_idx + cols * ldsa];
                        }
                        nu += accumulate[k];
                    }

                    // if(block_idx_x == 7 && reduction_time == 2 && cols == 0)
                    // {
                    //     printf("x: %d y: %d nu: %9.3f, shared_A[%d][%d]:
                    //     %9.3f\n", thread_idx_x, thread_idx_y, nu,
                    //     thread_idx_x, cols, shared_A[thread_idx_x + cols *
                    //     ldsa]);
                    // }

                    // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
                    T norm_x_squre = warp_all_reduce_sum(nu);
                    T norm_x = sqrt(norm_x_squre);

                    // 1、求u=x/norm(x);
                    T scale = 1.0 / norm_x;

                    // if(block_idx_x == 7 && thread_idx_x == 0 &&
                    // reduction_time == 2 && cols == 0) {
                    //    printf("norm_x_squre: %9.3f, norm_x: %9.3f, scale:
                    //    %9.3f\n", norm_x_squre, norm_x, scale);
                    // }

#pragma unroll
                    for (int k = 0; k < num_data_row; k++) {
                        int row_idx = thread_idx_x + k * block_dim_x;
                        if (row_idx < block_data_height && row_idx >= cols) {
                            shared_A[row_idx + cols * ldsa] *= scale;
                        }
                    }
                    __syncwarp();

                    // if(cols == 0 && reduction_time == 2 && block_idx_x == 7
                    // && thread_idx_x == 0 && thread_idx_y == 0) {
                    //     printf("after %d times scale\n", cols);
                    //     for(int v = 0; v < block_data_height; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", shared_A[v + l * ldsa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }

                    // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程;来计算即可
                    if (0 == thread_idx_x) {
                        T u1 = shared_A[cols + cols * ldsa];
                        shared_A[cols + cols * ldsa] += (u1 >= 0) ? 1 : -1;
                        // // 把normx存放到RR中，也就是对角线的元素
                        RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
                    }

                    // if(cols == 0 && reduction_time == 2 && block_idx_x == 7
                    // && thread_idx_x == 0 && thread_idx_y == 0) {
                    //     printf("after %d set identity\n", cols);
                    //     for(int v = 0; v < block_data_height; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", shared_A[v + l * ldsa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }
                    __syncwarp();

                    // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
                    scale = 1 / (sqrt(abs(shared_A[cols + cols * ldsa])));

                    // if(block_idx_x == 7 && thread_idx_x == 0 &&
                    // reduction_time == 2 && cols == 0) {
                    //    printf("scale: %9.3f\n", scale);
                    // }

#pragma unroll
                    for (int k = 0; k < num_data_row; k++) {
                        int row_idx = thread_idx_x + k * block_dim_x;
                        if (row_idx < block_data_height && row_idx >= cols) {
                            shared_A[row_idx + cols * ldsa] *= scale;
                        }
                    }

                    // if(cols == 0 && reduction_time == 2 && block_idx_x == 7
                    // && thread_idx_x == 0 && thread_idx_y == 0) {
                    //     printf("after %d times scale 2\n", cols);
                    //     for(int v = 0; v < block_data_height; ++v) {
                    //         for(int l = 0; l < n; ++l) {
                    //             printf("%9.3f ", shared_A[v + l * ldsa]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }
                }

                __syncthreads();
                // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
                // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
                // 每个线程按列需要处理多个列
                for (int h = 0; h < num_data_col; h++) {
                    T nu = 0.0;
                    int opCols = thread_idx_y + h * block_dim_y;

                    // 只更新当前列后面的列
                    if (cols < opCols && opCols < n) {
                        // 先计算u'x
#pragma unroll
                        for (int k = 0; k < num_data_row; k++) {
                            accumulate[k] = 0.0;
                            int row_idx = thread_idx_x + k * block_dim_x;
                            // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                            if (row_idx < block_data_height &&
                                row_idx >= cols) {
                                accumulate[k] =
                                    shared_A[row_idx + cols * ldsa] *
                                    shared_A[row_idx + opCols * ldsa];
                            }
                            nu += accumulate[k];
                        }
                        T utx = warp_all_reduce_sum(nu);

                        // if(cols == 0 && reduction_time == 2 && block_idx_x ==
                        // 7 && thread_idx_x == 0) {
                        //     printf("opcol: %d, utx: %9.6f\n", opCols, utx);
                        // }
                        // 计算x-uu'x
#pragma unroll
                        for (int k = 0; k < num_data_row; k++) {
                            int row_idx = thread_idx_x + k * block_dim_x;
                            // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                            if (row_idx < block_data_height &&
                                row_idx >= cols) {
                                shared_A[row_idx + opCols * ldsa] -=
                                    utx * shared_A[row_idx + cols * ldsa];
                                // if(reduction_time == 2 && block_idx_x == 7 &&
                                // opCols == 1){
                                //     printf("update A[%d][%d] to %9.6f\n",
                                //     thread_idx_x + k * block_dim_x, opCols,
                                //     shared_A[thread_idx_x + k * block_dim_x +
                                //     opCols * ldsa]);
                                // }
                            }
                        }

                        __syncwarp();
                        // if(cols == 0 && reduction_time == 2 && block_idx_x ==
                        // 7 && thread_idx_x == 0 && (opCols == 2)) {
                        //     printf("after update opCol: %d by %d using utx:
                        //     %9.3f\n", opCols, cols, utx); for(int v = 0; v <
                        //     block_data_height; ++v) {
                        //         for(int l = 0; l < n; ++l) {
                        //             printf("%9.3lf ", shared_A[v + l *
                        //             ldsa]);
                        //         }
                        //         printf("\n");
                        //     }
                        // }
                    }
                }
                // if(cols == 0 && reduction_time == 2 && block_idx_x == 7 &&
                // thread_idx_x == 0 && thread_idx_y == 0) {
                //     printf("after %d all\n", cols);
                //     for(int v = 0; v < block_data_height; ++v) {
                //         for(int l = 0; l < n; ++l) {
                //             printf("%9.3f ", shared_A[v + l * ldsa]);
                //         }
                //         printf("\n");
                //     }
                // }
            }

            T *R_to;
            int ldr_to;
            if (all_data_height <= block_size) {
                R_to = R;
                ldr_to = ldr;
            } else {
                R_to = &A[block_idx_x * n];
                ldr_to = lda;
            }

            __syncthreads();

            // 获得R矩阵，将AA的上三角部分拷贝到R中
            // 以R矩阵来进行循环
            int num_r_data_row = (n + (block_dim_x - 1)) / block_dim_x;
            for (int h = 0; h < num_data_col; h++) {
                int opCols = thread_idx_y + h * block_dim_y;
                if (opCols >= n) continue;
#pragma unroll
                for (int k = 0; k < num_r_data_row; k++) {
                    int row_idx = thread_idx_x + k * block_dim_x;
                    if (row_idx < opCols) {
                        R_to[row_idx + opCols * ldr_to] =
                            shared_A[row_idx + opCols * ldsa];
                        shared_A[row_idx + opCols * ldsa] = 0.0;
                    } else if (row_idx > opCols) {
                        R_to[row_idx + opCols * ldr_to] = 0.0;
                    } else {
                        // 这个赋值完全可以放到上面RR的赋值哪儿，从而不需要RR的共享内存
                        R_to[opCols + opCols * ldr_to] = RR[opCols];
                    }
                }
            }

            __syncthreads();

            // if (block_idx_x == 7 && thread_idx_x == 0 && thread_idx_y == 0) {
            //     if (all_data_height <= block_size) {
            //         printf("save to R\n");
            //     } else {
            //      printf("save to work[%d][%d]\n", block_idx_x * n, 0);
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
            // 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q_per_thread
            // q表示是Q矩阵的1列 double q_per_thread[num_data_row * 2];
            for (int h = 0; h < num_data_col; h++) {
                // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
                int opCols = thread_idx_y + h * block_dim_y;

                if (opCols >= n) continue;

                for (int k = 0; k < num_data_row; k++) {
                    int row_idx = thread_idx_x + k * block_dim_x;
                    if (row_idx == opCols) {
                        q_per_thread[k + h * 4] = 1.0;
                    } else {
                        q_per_thread[k + h * 4] = 0.0;
                    }
                }

                __syncwarp();

                for (int cols = n - 1; cols >= 0; cols--) {
                    // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
                    // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
                    // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
                    if (opCols >= cols) {
                        // 2、计算u'q_per_thread
                        T nu = 0.0;
                        for (int k = 0; k < num_data_row; k++) {
                            accumulate[k] = 0.0;
                            int row_idx = thread_idx_x + k * block_dim_x;
                            if (row_idx < block_data_height) {
                                accumulate[k] =
                                    shared_A[row_idx + cols * ldsa] *
                                    q_per_thread[k + h * 4];
                            }
                            nu += accumulate[k];
                        }

                        T utq = warp_all_reduce_sum(nu);

                        // 3.计算q-uu'q_per_thread
                        for (int k = 0; k < num_data_row; k++) {
                            int row_idx = thread_idx_x + k * block_dim_x;
                            if (row_idx < block_data_height) {
                                q_per_thread[k + h * 4] -=
                                    utq * shared_A[row_idx + cols * ldsa];
                            }
                        }

                        __syncwarp();
                    }
                }
            }

            __syncthreads();

            for (int k = 0; k < num_data_row; k++) {
                int row_idx = thread_idx_x + k * block_dim_x;
                if (row_idx < block_data_height) {
                    for (int h = 0; h < num_data_col; h++) {
                        int col_idx = thread_idx_y + h * block_dim_y;
                        if (col_idx < n) {
                            shared_A[row_idx + col_idx * ldsa] =
                                q_per_thread[k + h * 4];
                        }
                    }
                }
            }

            __syncthreads();

            // if (blockIdx.x == 0 && thread_idx_x == 0 && thread_idx_y == 0) {
            //     printf("save to shared_A[%d]\n", reduction_time);
            //     for (int v = 0; v < block_data_height; v++) {
            //         for (int l = 0; l < 32; l++) {
            //             printf("%9.6f ", shared_A[v + l * ldsa]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            __threadfence();

            if (thread_idx_x == 0 && thread_idx_y == 0) {
                atomicAdd((int *)&sync_counter, 1);
                // printf("reduction_time: %d blockIdx: %d Add sync_counter to
                // %d (count_end_block: %d)\n", reduction_time, block_idx_x,
                // sync_counter, count_end_block);
            }
        }

        while (sync_counter < count_end_block) {
            // printf("1 %d %d\n", sync_counter, count_end_block);
        }

        if (thread_idx_x == 0 && thread_idx_y == 0) {
            all_data_height =
                ((all_data_height + block_size - 1) / block_size) * n;
            shared_all_data_height[++reduction_time] = all_data_height;
            // if (block_idx_x == 0) {
            //     printf("shared_all_data_height[%d] = %d\n", reduction_time,
            //     all_data_height);
            // }
        }

        __syncthreads();
    }

    if (thread_idx_x == 0 && thread_idx_y == 0) {
        reduction_time -= 1;
    }

    __syncthreads();

    // perform gemm to obtain final Q
    while (reduction_time >= 0) {
        int all_data_height = shared_all_data_height[reduction_time];
        int block_data_height =
            min(all_data_height - block_idx_x * block_size, block_size);
        // int block_data_height = ((all_data_height - block_idx_x *
        // block_size) < block_size) ? (all_data_height - block_idx_x *
        // block_size) : block_size;

        num_reduction_block = (all_data_height + block_size - 1) / block_size;
        count_end_block = count_end_block - num_reduction_block * 2;

        if (block_data_height > 0) {
            // if (thread_idx_x == 0 && thread_idx_y == 0) {
            //     printf("reduction_time = %d, all_data_height = %d, blockidx =
            //     %d\n", reduction_time,
            //            all_data_height, block_idx_x);
            // }
            const int num_data_row =
                (block_data_height + block_dim_x - 1) / block_dim_x;
            const int num_data_col = (n + block_dim_y - 1) / block_dim_y;

            T *q_next = &A[block_idx_x * n];
            T *q_this = &all_shared_A[reduction_time * n * ldsa];
            T *q_to = &A[block_idx_x * block_size];
            T *q_work =
                &work[block_idx_x *
                      block_size];  // may able to remove q_work and work

            if (all_data_height > block_size) {
                // if (thread_idx_x == 0 && thread_idx_y == 0) {
                //     printf(
                //             "blockidx = %d, gemm: shared_A[%d] * work[%d] ->
                //             shared_A[%d]\n", block_idx_x, reduction_time,
                //             block_idx_x * n, reduction_time);
                // }

                // __syncthreads();

                // if(block_idx_x == 0 && thread_idx_x == 0 && thread_idx_y ==
                // 0) {
                //     printf("A\n");
                //     for(int v = 0; v < block_data_height; ++v) {
                //         for(int l = 0; l < 32; ++l) {
                //             printf("%9.6f ", q_this[v + l * ldsa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }

                // if(block_idx_x == 0 && thread_idx_x == 0 && thread_idx_y ==
                // 0) {
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
                if (thread_idx_x == 0 && thread_idx_y == 0) {
                    atomicAdd((int *)&sync_counter, -1);
                    // printf("reduction_time: %d blockIdx: %d Add sync_counter
                    // to %d (count_end_block: %d)\n", reduction_time,
                    // block_idx_x, sync_counter, count_end_block);
                }
                while (sync_counter > count_end_block + num_reduction_block) {
                    // printf("2 %d %d\n", sync_counter, count_end_block);
                }

                block_gemm(block_data_height, n, q_work, ldwork, q_this, ldsa,
                           q_next, lda);

                __threadfence();
                __syncthreads();
                if (thread_idx_x == 0 && thread_idx_y == 0) {
                    atomicAdd((int *)&sync_counter, -1);
                    // printf("reduction_time: %d blockIdx: %d Add sync_counter
                    // to %d (count_end_block: %d)\n", reduction_time,
                    // block_idx_x, sync_counter, count_end_block);
                }

                while (sync_counter > count_end_block) {
                    // printf("3 %d %d\n", sync_counter, count_end_block);
                }

                // if(block_idx_x == 0 && thread_idx_x == 0 && thread_idx_y ==
                // 0) {
                //     printf("C\n");
                //     for(int v = 0; v < block_data_height; ++v) {
                //         for(int l = 0; l < n; ++l) {
                //             printf("%9.6f ", temp_A[v + l * ldsa]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }

                for (int row_load_idx = 0; row_load_idx < num_data_row;
                     row_load_idx++) {
                    int row_idx = thread_idx_x + row_load_idx * block_dim_x;
                    if (row_idx < block_data_height) {
                        for (int col_load_idx = 0; col_load_idx < num_data_col;
                             col_load_idx++) {
                            int col_idx =
                                thread_idx_y + col_load_idx * block_dim_y;
                            if (col_idx < n) {
                                q_to[row_idx + col_idx * lda] =
                                    q_work[row_idx + col_idx * ldwork];
                            }
                        }
                    }
                }
            } else {  // last 128 size block in A
                __threadfence();
                if (thread_idx_x == 0 && thread_idx_y == 0) {
                    atomicAdd((int *)&sync_counter, -2);
                    // printf("reduction_time: %d blockIdx: %d Add sync_counter
                    // to %d (count_end_block: %d)\n", reduction_time,
                    // block_idx_x, sync_counter, count_end_block);
                }
                while (sync_counter > count_end_block) {
                    // printf("4 %d %d\n", sync_counter, count_end_block);
                }

                // if (thread_idx_x == 0 && thread_idx_y == 0) {
                //     printf("blockidx = %d, move shared_A[%d] -> work[%d]\n",
                //            block_idx_x, reduction_time, block_idx_x *
                //            block_size);
                // }
                for (int row_load_idx = 0; row_load_idx < num_data_row;
                     row_load_idx++) {
                    int row_idx = thread_idx_x + row_load_idx * block_dim_x;
                    if (row_idx < block_data_height) {
                        for (int col_load_idx = 0; col_load_idx < num_data_col;
                             col_load_idx++) {
                            int col_idx =
                                thread_idx_y + col_load_idx * block_dim_y;
                            if (col_idx < n) {
                                q_to[row_idx + col_idx * lda] =
                                    q_this[row_idx + col_idx * ldsa];
                            }
                        }
                    }
                }
            }
        } else {  // if block_data_height > 0
            while (sync_counter > count_end_block) {
                // printf("5 %d %d\n", sync_counter, count_end_block);
            }
        }

        if (thread_idx_x == 0 && thread_idx_y == 0) {
            reduction_time--;
        }

        // grid.sync();
        __syncthreads();
    }
}

// template __global__ void my_hou_kernel<float, 128, 32>(const int m, const int
// n,
//         float *A, const int lda,
//         float *R, const int ldr,
//         float *work,
//         const int ldwork);
template __global__ void my_hou_kernel<double>(const int block_size,
                                               const int m, const int n,
                                               double *A, const int lda,
                                               double *R, const int ldr,
                                               double *work, const int ldwork);
