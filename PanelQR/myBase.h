#pragma once

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// #define MY_DEBUG 1

// 下面的部分只在单个文件的内部进行调用
static cudaEvent_t start, stop;
static void startTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

static float stopTimer()
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}