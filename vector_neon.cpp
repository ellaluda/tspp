#include <iostream>
#include <chrono>
#include <cmath>

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Sequential matrix multiplication
void matrixMultiplySequential(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Vectorized matrix multiplication using AVX
#ifdef __AVX__
void matrixMultiplyVectorized(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j += 8) {
            __m256 c = _mm256_setzero_ps();  // Initialize result vector to zero
            for (int k = 0; k < N; ++k) {
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);   // Load 8 elements of B
                __m256 a = _mm256_set1_ps(A[i * N + k]);     // Duplicate A[i][k]
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));   // Multiply and accumulate
            }
            _mm256_storeu_ps(&C[i * N + j], c);  // Store result in C
        }
    }
}
#endif

// Vectorized matrix multiplication using NEON
#ifdef __ARM_NEON
void matrixMultiplyVectorized(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t c = vdupq_n_f32(0);  // Initialize result vector to zero
            for (int k = 0; k < N; ++k) {
                float32x4_t b = vld1q_f32(&B[k * N + j]);   // Load 4 elements of B
                float32x4_t a = vdupq_n_f32(A[i * N + k]);  // Duplicate A[i][k]
                c = vmlaq_f32(c, a, b);                     // Multiply and accumulate
            }
            vst1q_f32(&C[i * N + j], c);  // Store result in C
        }
    }
}
#endif

int main() {
    const int N = 2048;  // Matrix size
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C1 = new float[N * N];
    float *C2 = new float[N * N];

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Measure time for sequential version
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplySequential(A, B, C1, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential version took: " << duration.count() << " seconds\n";

    // Measure time for vectorized version
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyVectorized(A, B, C2, N);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Vectorized version took: " << duration.count() << " seconds\n";

    float maxError = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        maxError = std::max(maxError, std::abs(C2[i] - C1[i]));
    }
    std::cout << "Max error: " << maxError << "\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;

    return 0;
}
