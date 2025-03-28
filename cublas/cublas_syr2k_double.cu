#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 32768// 矩阵的大小
#define K 32768// 矩阵A和B的列数

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
    double alpha = 1.0, beta = 1.0;

    // 2. 在主机上分配内存
    A = (double*)malloc(N * K * sizeof(double));
    B = (double*)malloc(N * K * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));  // 对称矩阵C

    // 3. 初始化矩阵A, B, C
    random_init(A, N * K);
    random_init(B, N * K);
    random_init(C, N * N);  // 生成随机C矩阵
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            C[i * N + j] = C[j * N + i];  // 保证C是对称的
        }
    }

    // 4. 在GPU上分配内存
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_A, N * K * sizeof(double)));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_B, N * K * sizeof(double)));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_C, N * N * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_CUDA_CALL(cudaMemcpy(d_A, A, N * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_B, B, N * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // 6. 运行cublasDsyr2k函数
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    CHECK_CUBLAS_CALL(cublasDsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                   N, K, &alpha, d_A, N, d_B, N, &beta, d_C, N));
                                   cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    // 7. 输出性能结果
    std::cout << "[cublas dsyr2k] " << "n: " << N << ", k: " << K << ", "
              << "latency: " << time_ms << " ms, "
              << "Effective TFLOPS: " << (long)N * N * K * 2 / time_ms / 1e9 << " TFLOPS, "
              << std::endl;

    // 8. 清理
    CHECK_CUDA_CALL(cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));

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

