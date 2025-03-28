#include <iostream>
#include <cstdlib>
#include <ctime>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define N 16384  // 矩阵的大小
#define K 16384  // 矩阵A的列数

// Error检查宏
#define CHECK_HIP_CALL(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_HIPBLAS_CALL(call) \
    { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "HIPBLAS error: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void random_init(double* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main() {
    // 1. 初始化HIP和hipBLAS
    hipblasHandle_t handle;
    CHECK_HIPBLAS_CALL(hipblasCreate(&handle));

    double *A, *C;
    double *d_A, *d_C;
    double alpha = 1.0, beta = 1.0;

    // 2. 在主机上分配内存
    A = (double*)malloc(N * K * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));  // 对称矩阵C

    // 3. 初始化矩阵A和C
    random_init(A, N * K);
    random_init(C, N * N);  // 生成随机C矩阵
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            C[i * N + j] = C[j * N + i];  // 保证C是对称的
        }
    }

    // 4. 在GPU上分配内存
    CHECK_HIP_CALL(hipMalloc((void**)&d_A, N * K * sizeof(double)));
    CHECK_HIP_CALL(hipMalloc((void**)&d_C, N * N * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_HIP_CALL(hipMemcpy(d_A, A, N * K * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_CALL(hipMemcpy(d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice));

    // 6. 运行hipblasDsyrk函数
    float time_ms;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    CHECK_HIPBLAS_CALL(hipblasDsyrk(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N,
                                     N, K, &alpha, d_A, N, &beta, d_C, N));
    hipEventRecord(stop);

    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_ms, start, stop);

    // 7. 输出性能结果
    std::cout << "Execution time: " << time_ms << " ms" << std::endl;

    // 8. 清理
    CHECK_HIP_CALL(hipMemcpy(C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost));

    // 可选：验证计算结果是否正确
    // ...

    // 9. 释放资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_C);
    free(A);
    free(C);

    // 10. 销毁hipBLAS句柄
    hipblasDestroy(handle);

    return 0;
}

