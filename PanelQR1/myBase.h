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

template <typename T>
void printDeviceMatrixV2(T *dA, long ldA, long rows, long cols)
{
  T matrix;

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
      printf("%5.1f ", matrix);
    }
    printf("\n");
  }
}

template void printDeviceMatrixV2(double *dA, long ldA, long rows, long cols);
template void printDeviceMatrixV2(float *dA, long ldA, long rows, long cols);