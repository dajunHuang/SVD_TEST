#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#define M 16384  // 矩阵的大小
#define N 16384  // 矩阵B的列数

// 错误检查宏
#define CHECK_CUDA_CALL(call)                                      \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << std::endl;                                \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

#define CHECK_CUBLAS_CALL(call)                                   \
    {                                                             \
        cublasStatus_t status = call;                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            std::cerr << "CUBLAS error: " << status << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

void random_init(double *data, int size) {
    for (int i = 0; i < size; i++) {
        // data[i] = static_cast<double>(rand()) / RAND_MAX;
        data[i] = 1;
    }
}

__global__ void checkValueDouble(int m, int n, double *A, int lda, double tol) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        double const A_val{static_cast<double>(A[i + j * lda])};
        if (isnan(A_val) || isinf(A_val)) {
            printf("A[%d, %d] = %f\n", i, j, A_val);
        }
    }
}

int main(int argc, char *argv[]) {
    // 1. 初始化CUDA和cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_CALL(cublasCreate(&handle));

    double *A, *B;
    double *d_A, *d_B;
    double alpha = 1.0;

    double const fp64_abs_tol = 1.0e-4f;

    // 2. 在主机上分配内存
    A = (double *)malloc(M * M * sizeof(double));  // 三角矩阵A
    B = (double *)malloc(M * N * sizeof(double));  // 矩阵B

    // 3. 初始化矩阵A和B
    random_init(A, M * M);  // 矩阵A
    random_init(B, M * N);  // 矩阵B

    // // 确保A是三角矩阵
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < M; j++) {
    //         if(i < j)
    //             A[i + j * M] = 0.0;
    //     }
    // }

    // 4. 在GPU上分配内存
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_A, M * M * sizeof(double)));
    CHECK_CUDA_CALL(cudaMalloc((void **)&d_B, M * N * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_CUDA_CALL(
        cudaMemcpy(d_A, A, M * M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(
        cudaMemcpy(d_B, B, M * N * sizeof(double), cudaMemcpyHostToDevice));

    // 6. 运行cublasDtrsm函数
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    CHECK_CUBLAS_CALL(cublasDtrsm(
        handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT, M, N, &alpha, d_A, M, d_B, M));
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    dim3 gridb((M + 31) / 32, (N + 31) / 32);
    dim3 blockb(32, 32);
    checkValueDouble<<<gridb, blockb>>>(M, N, d_B, M, fp64_abs_tol);

    // 7. 输出性能结果
    std::cout << "[cublas dtrsm] " << "m: " << M << ", n: " << N << ", "
              << "latency: " << time_ms << " ms, "
              << "Effective TFLOPS: " << (long)M * M * N / time_ms / 1e9
              << " TFLOPS, " << std::endl;

    // 可选：验证计算结果是否正确
    // ...

    // 9. 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);

    // 10. 销毁cuBLAS句柄
    cublasDestroy(handle);

    return 0;
}
