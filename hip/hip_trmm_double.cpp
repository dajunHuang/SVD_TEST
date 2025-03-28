#include <iostream>
#include <cstdlib>
#include <ctime>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define N 16384 // 矩阵的大小
#define K 16384 // 矩阵B的列数

// 错误检查宏
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

    double *A, *B;
    double *d_A, *d_B;
    double alpha = 1.0;

    // 2. 在主机上分配内存
    A = (double*)malloc(N * N * sizeof(double));  // 三角矩阵A
    B = (double*)malloc(N * K * sizeof(double));  // 矩阵B

    // 3. 初始化矩阵A和B
    random_init(A, N * N);  // 矩阵A
    random_init(B, N * K);  // 矩阵B

    // 确保A是三角矩阵
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[i * N + j] = 0.0;  // 上三角矩阵，设置下三角为0
        }
    }

    // 4. 在GPU上分配内存
    CHECK_HIP_CALL(hipMalloc((void**)&d_A, N * N * sizeof(double)));
    CHECK_HIP_CALL(hipMalloc((void**)&d_B, N * K * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_HIP_CALL(hipMemcpy(d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_CALL(hipMemcpy(d_B, B, N * K * sizeof(double), hipMemcpyHostToDevice));

    // 6. 运行hipblasDtrmm函数
    float time_ms;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    CHECK_HIPBLAS_CALL(hipblasDtrmm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT,
                                  N, K, &alpha, d_A, N, d_B, N));
    hipEventRecord(stop);

    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_ms, start, stop);

    // 7. 输出性能结果
    std::cout << "Execution time: " << time_ms << " ms" << std::endl;

    // 8. 清理
    CHECK_HIP_CALL(hipMemcpy(B, d_B, N * K * sizeof(double), hipMemcpyDeviceToHost));

    // 可选：验证计算结果是否正确
    // ...

    // 9. 释放资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    free(A);
    free(B);

    // 10. 销毁hipBLAS句柄
    hipblasDestroy(handle);

    return 0;
}

