#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "utils.h"

#define NUM_WARMUP 2
#define NUM_REPEAT 5

#define CURAND_CHECK(call)                                                   \
    do {                                                                     \
        curandStatus_t status = (call);                                       \
        if (status != CURAND_STATUS_SUCCESS) {                               \
            std::cerr << "cuRAND error " << status << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                        \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

bool parse_dimension(const char *text, int &value) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char *argv[]) {
    int m = 32768, n = 32768;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    if ((argc != 1 && (argc < 3 || argc > 5)) ||
        (argc >= 3 && (!parse_dimension(argv[1], m) || !parse_dimension(argv[2], n))) ||
        (argc >= 4 && std::string(argv[3]) != "L" && std::string(argv[3]) != "R") ||
        (argc >= 5 && std::string(argv[4]) != "L" && std::string(argv[4]) != "U")) {
        std::cerr << "Usage: " << argv[0] << " [m n [side=L|R [uplo=L|U]]]" << std::endl;
        return EXIT_FAILURE;
    }
    if (argc >= 4 && std::string(argv[3]) == "R") {
        side = CUBLAS_SIDE_RIGHT;
    }
    if (argc >= 5 && std::string(argv[4]) == "U") {
        uplo = CUBLAS_FILL_MODE_UPPER;
    }

    // Column-major: C = alpha * A * B + beta * C (left), or
    //              C = alpha * B * A + beta * C (right).
    const int k = side == CUBLAS_SIDE_LEFT ? m : n;
    const int lda = k, ldb = m, ldc = m;
    const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (static_cast<size_t>(k) > max_elements / k ||
        static_cast<size_t>(m) > max_elements / n) {
        std::cerr << "Matrix allocation size is too large" << std::endl;
        return EXIT_FAILURE;
    }
    const size_t a_elements = static_cast<size_t>(lda) * k;
    const size_t bc_elements = static_cast<size_t>(m) * n;
    const double alpha = 1.0, beta = 0.0;

    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * a_elements));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * bc_elements));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * bc_elements));

    // SYMM reads only the selected triangle of A and implicitly mirrors it;
    // the unused triangle need not be explicitly made symmetric.
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 3000ULL));
    CURAND_CHECK(curandSetStream(generator, stream));
    CURAND_CHECK(curandGenerateUniformDouble(generator, d_A, a_elements));
    CURAND_CHECK(curandGenerateUniformDouble(generator, d_B, bc_elements));
    // beta is zero, so cuBLAS does not read the initial contents of C.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CURAND_CHECK(curandDestroyGenerator(generator));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < NUM_WARMUP; ++i) {
        CUBLAS_CHECK(cublasDsymm(handle, side, uplo, m, n, &alpha, d_A, lda,
                                d_B, ldb, &beta, d_C, ldc));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float time_ms = 0.0f;
    for (int i = 0; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUBLAS_CHECK(cublasDsymm(handle, side, uplo, m, n, &alpha, d_A, lda,
                                d_B, ldb, &beta, d_C, ldc));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        time_ms += elapsed_ms;
    }
    time_ms /= NUM_REPEAT;

    // Symmetry reduces A's stored input, not the dense product's FLOP count.
    std::cout << "[cublas dsymm] m: " << m << ", n: " << n
              << ", side: " << (side == CUBLAS_SIDE_LEFT ? "L" : "R")
              << ", uplo: " << (uplo == CUBLAS_FILL_MODE_LOWER ? "L" : "U")
              << ", latency: " << time_ms << " ms, "
              << "Effective TFLOPS: " << 2.0 * m * n * k / time_ms / 1e9
              << " TFLOPS, " << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
