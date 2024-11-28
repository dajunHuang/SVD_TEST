#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include "cusolver_utils.h"
namespace cg = cooperative_groups;

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        // val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int M, int N>
__global__ void my_hou_kernel(const int m, const int n, T *A,
                              const int lda, T *Y, const int ldy, T *R,
                              const int ldr, T *work, const int ldwork) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    __shared__ T shared_A[M * N * 1], RR[N];
    __shared__ int shared_work_height[2];
    __shared__ int idx;
    const int ldaa{M};
    T acc[4], q[4];

    // 2. 找到本线程的ID
    const int i{static_cast<int>(threadIdx.x)};
    const int j{static_cast<int>(threadIdx.y)};
    const int blockIdx_x{static_cast<int>(blockIdx.x)};
    const int blockDim_x{static_cast<int>(blockDim.x)};
    const int blockDim_y{static_cast<int>(blockDim.y)};

    if (i == 0 && j == 0) {
        idx = 0;
        shared_work_height[0] = static_cast<int>(m);
    }

    __syncthreads();
    int nn{static_cast<int>(n)};

    while (shared_work_height[idx] > M) {
        int work_height{shared_work_height[idx]};
        int mm{min(work_height - static_cast<int>(blockIdx_x * M),
                   static_cast<int>(M))};

        if (mm > 0) {
            int rowDataNum{(mm + blockDim_x - 1) / blockDim_x};
            int colDataNum{(nn + blockDim_y - 1) / blockDim_y};

            T *AA = &shared_A[idx * N * ldaa];

            if (idx == 0) {
                // 假定n=N=32，每一个线程拷贝2列
                for (long k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + j * ldaa] =
                            A[blockIdx_x * M + i + k * blockDim_x + j * lda];
                        AA[i + k * blockDim_x + (j + 16) * ldaa] =
                            A[blockIdx_x * M + i + k * blockDim_x +
                              (j + 16) * lda];
                    }
                }
            } else {
                // 假定n=N=32，每一个线程拷贝2列
                for (long k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + j * ldaa] =
                            work[blockIdx_x * M + i + k * blockDim_x +
                                 j * ldwork];
                        AA[i + k * blockDim_x + (j + 16) * ldaa] =
                            work[blockIdx_x * M + i + k * blockDim_x +
                                 (j + 16) * ldwork];
                    }
                }
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

                    // 1、求u=x/norm(x);
                    T scale = 1.0 / norm_x;
#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            // AA[i + k * blockDim_x + cols * ldaa] *= scale;
                            if(blockIdx_x == 0 && i == 0 && j == 0)
                            {
                                printf("AA[%d][%d] ldaa: %d %d %d %d %d\n", i + k * blockDim_x, cols, ldaa, i + k * blockDim_x + cols * ldaa, idx * N * ldaa, mm, nn);
                            }
                        }
                    }

                    if(blockIdx_x == 0 && i == 0 && j == 0) {
                        printf("\n");
                    }

                    __syncwarp();

                    // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
                    if (0 == i) {
                        T u1 = AA[cols + cols * mm];
                        // AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;
                        // // 把normx存放到RR中，也就是对角线的元素
                        // RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
                    }

                    __syncwarp();

                    // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
                    scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
                    for (int k = 0; k < rowDataNum; k++) {
                        if (i + k * blockDim_x < mm &&
                            i + k * blockDim_x >= cols) {
                            // AA[i + k * blockDim_x + cols * ldaa] *= scale;
                        }
                    }
                }

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
                                // AA[i + k * blockDim_x + opCols * ldaa] -=
                                //     utx * AA[i + k * blockDim_x + cols * ldaa];
                            }
                        }
                        __syncwarp();
                    }
                }
            }

            // 获得R矩阵，将AA的上三角部分拷贝到R中
            // 以R矩阵来进行循环
            int rRowDataNum = (nn + (blockDim.x - 1)) / blockDim_x;
            for (int h = 0; h < colDataNum; h++) {
                int opCols = j + h * blockDim_y;

                if (opCols >= nn) continue;

#pragma unroll
                for (int k = 0; k < rRowDataNum; k++) {
                    if (i + k * blockDim_x < opCols) {
                        // work[blockIdx_x * N + i + k * blockDim_x +
                        //      opCols * ldwork] =
                        //     AA[i + k * blockDim_x + opCols * ldaa];
                        // AA[i + k * blockDim_x + opCols * ldaa] = 0.0;
                    } else if (i + k * blockDim_x > opCols) {
                        work[blockIdx_x * N + i + k * blockDim_x +
                             opCols * ldwork] = 0.0;
                    } else {
                        // 这个赋值完全可以放到上面RR的赋值哪儿，从而不需要RR的共享内存
                        work[opCols + opCols * ldwork] = RR[opCols];
                    }
                }
            }

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
                                // q[k] -=
                                //     utq * AA[i + k * blockDim_x + cols * ldaa];
                            }
                        }

                        __syncwarp();
                    }
                }

                // 4.把计算出来的q拷贝到A中
                for (long k = 0; k < rowDataNum; k++) {
                    if (i + k * blockDim_x < mm) {
                        AA[i + k * blockDim_x + opCols * lda] = q[k];
                    }
                }
            }
        }

        __syncthreads();

        if (i == 0 && j == 0) {
            work_height = ((work_height + M - 1) / M) * N;
            // if (blockIdx_x == 0) {
            //     printf("shared_work_height[%d] = %d\n", idx, work_height);
            // }
            shared_work_height[++idx] = work_height;
        }
    }
}

template __global__ void my_hou_kernel<float, 128, 32>(
    const int m, const int n, float *A, const int lda, float *Y,
    const int ldy, float *R, const int ldr, float *work,
    const int ldwork);
