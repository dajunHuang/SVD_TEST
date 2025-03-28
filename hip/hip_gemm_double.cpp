#include <iostream>
#include <cstdlib>
#include <ctime>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define M 16384 // 矩阵A的行数
#define N 16384 // 矩阵B的列数
#define K 16384 // 矩阵A的列数，矩阵B的行数

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

    double *A, *B, *C;
    double *d_A, *d_B, *d_C;
    double alpha = 1.0;
    double beta = 0.0;

    // 2. 在主机上分配内存
    A = (double*)malloc(M * K * sizeof(double));  // 矩阵A
    B = (double*)malloc(K * N * sizeof(double));  // 矩阵B
    C = (double*)malloc(M * N * sizeof(double));  // 矩阵C

    // 3. 初始化矩阵A和B
    random_init(A, M * K);  // 初始化A
    random_init(B, K * N);  // 初始化B

    // 4. 在GPU上分配内存
    CHECK_HIP_CALL(hipMalloc((void**)&d_A, M * K * sizeof(double)));
    CHECK_HIP_CALL(hipMalloc((void**)&d_B, K * N * sizeof(double)));
    CHECK_HIP_CALL(hipMalloc((void**)&d_C, M * N * sizeof(double)));

    // 5. 将数据从主机复制到设备
    CHECK_HIP_CALL(hipMemcpy(d_A, A, M * K * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_CALL(hipMemcpy(d_B, B, K * N * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_CALL(hipMemcpy(d_C, C, M * N * sizeof(double), hipMemcpyHostToDevice));

    // 6. 运行hipblasDgemm函数
    float time_ms;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    CHECK_HIPBLAS_CALL(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    hipEventRecord(stop);

    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_ms, start, stop);

    // 7. 输出性能结果
    std::cout << "Execution time: " << time_ms << " ms" << std::endl;

    // 8. 清理
    CHECK_HIP_CALL(hipMemcpy(C, d_C, M * N * sizeof(double), hipMemcpyDeviceToHost));

    // 可选：验证计算结果是否正确
    // ...

    // 9. 释放资源
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(A);
    free(B);
    free(C);

    // 10. 销毁hipBLAS句柄
    hipblasDestroy(handle);

    return 0;
}

