#include <cuda_fp16.h>

#define BLOCK_SIZE 128
#define BLOCK_DIM_Y 16
#define NUM_Q_COL 2  // (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
#define BLOCK_DIM_X 32
#define NUM_Q_ROW (BLOCK_SIZE + BLOCK_DIM_X - 1) / BLOCK_DIM_X
#define MAX_N 32

template <typename T>
struct shared_memory;
template <>
struct shared_memory<float> {
    __device__ static float *get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};
template <>
struct shared_memory<double> {
    __device__ static double *get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum_ori(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void tsqr_kernel(int m, int n, T *A, int lda, T *R, int ldr) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    shared_memory<T> shared;
    T *AA = shared.get_pointer();
    __shared__ T RR[MAX_N];

    int ldsa = BLOCK_SIZE;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;
    const int block_dim_x = BLOCK_DIM_X;
    const int block_dim_y = BLOCK_DIM_Y;

    // 1.求出本block处理的矩阵的尺寸
    int block_data_height = min(m - block_idx_x * BLOCK_SIZE, BLOCK_SIZE);

    // 理论上不会出现mm<=0的情况，这里只是错误处理
    if (0 >= block_data_height) {
        return;
    }

    A = A + block_idx_x * BLOCK_SIZE;
    R = R + block_idx_x * n;

    // 每个线程处理的数据个数
    int rowDataNum = (block_data_height + (block_dim_x - 1)) / block_dim_x;
    int colDataNum = (n + (block_dim_y - 1)) / block_dim_y;

    T acc[NUM_Q_ROW];

    // 假定n=N=32，每一个线程拷贝2列
    for (int k = 0; k < rowDataNum; k++) {
        if (thread_idx_x + k * block_dim_x < block_data_height) {
            AA[thread_idx_x + k * block_dim_x + thread_idx_y * ldsa] =
                A[thread_idx_x + k * block_dim_x + thread_idx_y * lda];
            AA[thread_idx_x + k * block_dim_x + (thread_idx_y + 16) * ldsa] =
                A[thread_idx_x + k * block_dim_x + (thread_idx_y + 16) * lda];
        }
    }

    // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
    __syncthreads();

    // 进行HouseHolder分解，先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
    // 3、u=u/sqrt(abs(u(1)))
    for (int cols = 0; cols < n; cols++) {
        // 先计算HouseHolder向量
        // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
        // 3、u=u/sqrt(abs(u(1)))
        T nu = 0.0;
        if (thread_idx_y == cols % block_dim_y) {
            // 0.求normx
            // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
            for (int k = 0; k < rowDataNum; k++) {
                acc[k] = 0.0;
                // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                if (thread_idx_x + k * block_dim_x < block_data_height &&
                    thread_idx_x + k * block_dim_x >= cols) {
                    acc[k] = AA[thread_idx_x + k * block_dim_x + cols * ldsa] *
                             AA[thread_idx_x + k * block_dim_x + cols * ldsa];
                }
                nu += acc[k];
            }

            // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
            T norm_x_squre = warpAllReduceSum_ori(nu);
            T norm_x = sqrt(norm_x_squre);

            // 1、求u=x/norm(x);
            T scale = 1.0 / norm_x;
#pragma unroll
            for (int k = 0; k < rowDataNum; k++) {
                if (thread_idx_x + k * block_dim_x < block_data_height &&
                    thread_idx_x + k * block_dim_x >= cols) {
                    AA[thread_idx_x + k * block_dim_x + cols * ldsa] *= scale;
                }
            }

            __syncwarp();

            // 2、求u(1)= u(1)+sign(u(1)); 每列找一个线程来计算即可
            if (0 == thread_idx_x) {
                T u1 = AA[cols + cols * block_data_height];

                AA[cols + cols * ldsa] += (u1 >= 0) ? 1 : -1;

                // 把normx存放到RR中，也就是对角线的元素
                RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
            }

            __syncwarp();

            // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
            scale = 1 / (sqrt(abs(AA[cols + cols * ldsa])));
#pragma unroll
            for (int k = 0; k < rowDataNum; k++) {
                if (thread_idx_x + k * block_dim_x < block_data_height &&
                    thread_idx_x + k * block_dim_x >= cols) {
                    AA[thread_idx_x + k * block_dim_x + cols * ldsa] *= scale;
                }
            }
        }

        __syncthreads();
        // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
        // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
        // 每个线程按列需要处理多个列
        for (int h = 0; h < colDataNum; h++) {
            T nu = 0.0;
            int opCols = thread_idx_y + h * block_dim_y;

            // 只更新当前列后面的列
            if (cols < opCols && opCols <= n) {
                // 先计算u'x
#pragma unroll
                for (int k = 0; k < rowDataNum; k++) {
                    acc[k] = 0.0;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (thread_idx_x + k * block_dim_x < block_data_height &&
                        thread_idx_x + k * block_dim_x >= cols) {
                        acc[k] =
                            AA[thread_idx_x + k * block_dim_x + cols * ldsa] *
                            AA[thread_idx_x + k * block_dim_x + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warpAllReduceSum_ori(nu);

                // 计算x-uu'x
#pragma unroll
                for (int k = 0; k < rowDataNum; k++) {
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (thread_idx_x + k * block_dim_x < block_data_height &&
                        thread_idx_x + k * block_dim_x >= cols) {
                        AA[thread_idx_x + k * block_dim_x + opCols * ldsa] -=
                            utx *
                            AA[thread_idx_x + k * block_dim_x + cols * ldsa];
                    }
                }
                __syncwarp();
            }
        }
    }

    __syncthreads();
    // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

    // 获得R矩阵，将AA的上三角部分拷贝到R中
    // 以R矩阵来进行循环
    int rRowDataNum = (n + (block_dim_x - 1)) / block_dim_x;
    for (int h = 0; h < colDataNum; h++) {
        int opCols = thread_idx_y + h * block_dim_y;

        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            if (thread_idx_x + k * block_dim_x < opCols) {
                R[thread_idx_x + k * block_dim_x + opCols * ldr] =
                    AA[thread_idx_x + k * block_dim_x + opCols * ldsa];
                AA[thread_idx_x + k * block_dim_x + opCols * ldsa] = 0.0;
            } else if (thread_idx_x + k * block_dim_x > opCols) {
                R[thread_idx_x + k * block_dim_x + opCols * ldr] = 0.0;
            } else {
                // 这个赋值完全可以放到上面RR的赋值哪儿，从而不需要RR的共享内存
                R[opCols + opCols * ldr] = RR[opCols];
            }
        }
    }

    // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
    // q表示是Q矩阵的1列
    T q[NUM_Q_ROW * NUM_Q_COL];
    for (int h = 0; h < colDataNum; h++) {
        // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
        int opCols = thread_idx_y + h * block_dim_y;

        if (opCols >= n) continue;

        for (int k = 0; k < rowDataNum; k++) {
            if (thread_idx_x + k * block_dim_x == opCols) {
                q[k] = 1.0;
            } else {
                q[k] = 0.0;
            }
        }

        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
            // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
            // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
            if (opCols >= cols) {
                // 2、计算u'q
                T nu = 0.0;
                for (int k = 0; k < rowDataNum; k++) {
                    acc[k] = 0.0;
                    if (thread_idx_x + k * block_dim_x < block_data_height) {
                        acc[k] =
                            AA[thread_idx_x + k * block_dim_x + cols * ldsa] *
                            q[k];
                    }
                    nu += acc[k];
                }

                T utq = warpAllReduceSum_ori(nu);

                // 3.计算q-uu'q
                for (int k = 0; k < rowDataNum; k++) {
                    if (thread_idx_x + k * block_dim_x < block_data_height) {
                        q[k] -=
                            utq *
                            AA[thread_idx_x + k * block_dim_x + cols * ldsa];
                    }
                }

                __syncwarp();
            }
        }

        // 4.把计算出来的q拷贝到A中
        for (int k = 0; k < rowDataNum; k++) {
            if (thread_idx_x + k * block_dim_x < block_data_height) {
                A[thread_idx_x + k * block_dim_x + opCols * lda] = q[k];
            }
        }
    }
}

template __global__ void tsqr_kernel<float>(int m, int n, float *A, int lda,
                                            float *R, int ldr);
template __global__ void tsqr_kernel<double>(int m, int n, double *A, int lda,
                                             double *R, int ldr);