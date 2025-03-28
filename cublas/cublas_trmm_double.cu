#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 32768 // 矩阵的大小
#define N 32768 // 矩阵B的列数

// 错误检查宏
#define CHECK_CUDA_CALL(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_CUBLAS_CALL(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void random_init(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    // 1. 初始化CUDA和cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_CALL(cublasCreate(&handle));

    double *A, *B, *C;
    double *d_A, *d_B, *d_C;
    double alpha = 1.0;

    // 2. 在主机上分配内存
    A = (double*)malloc(M * M * sizeof(double));  // 方阵A
    B = (double*)malloc(M * N * sizeof(double));  // 矩阵B
    C = (double*)malloc(M * N * sizeof(double));  // 矩阵C

    // 3. 初始化矩阵A和B
    random_init(A, M * M);  // 矩阵A
    random_init(B, M * N);  // 矩阵B

    // 4. 在GPU上分配内存
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_A, M * M * sizeof(double)));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_B, M * N * sizeof(double)));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_C, M * N * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_CUDA_CALL(cudaMemcpy(d_A, A, M * M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_B, B, M * N * sizeof(double), cudaMemcpyHostToDevice));

    // 6. 运行cublasDtrmm函数
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                  M, N, &alpha, d_A, M, d_B, M, d_C, M);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    // 7. 输出性能结果
    std::cout << "[cublas dtrmm] " << "m: " << M << ", n: " << N << ", "
              << "latency: " << time_ms << " ms, "
              << "Effective TFLOPS: " << (long)M * M * N / time_ms / 1e9 << " TFLOPS, "
              << std::endl;

    // 8. 清理
    CHECK_CUDA_CALL(cudaMemcpy(C, d_B, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    // 可选：验证计算结果是否正确
    // ...

    // 9. 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    // 10. 销毁cuBLAS句柄
    cublasDestroy(handle);

    return 0;
}

