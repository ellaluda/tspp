#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>

// Assuming all matrices are of size N x N, stored in row-major order
const int N = 512; // Adjustable to 512, 1024, 2048, etc.

void matrixMultiplyAVX(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i += 8) {
        for (int j = 0; j < n; ++j) {
            __m256 c[8] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (int k = 0; k < n; ++k) {
                __m256 b = _mm256_broadcast_ss(&B[k * n + j]);

                for (int x = 0; x < 8; ++x) {
                    __m256 a = _mm256_loadu_ps(&A[(i + x) * n + k]);
                    c[x] = _mm256_fmadd_ps(a, b, c[x]);
                }
            }

            for (int x = 0; x < 8; ++x) {
                _mm256_storeu_ps(&C[(i + x) * n + j], c[x]);
            }
        }
    }
}

void matrixMultiplySequential(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    std::vector<float> A(N * N), B(N * N), C(N * N), C_seq(N * N);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.0001;
        B[i] = 0.0002;
    }

    // Matrix multiplication using AVX
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyAVX(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "AVX version time: " << elapsed.count() << " seconds\n";

    // Matrix multiplication using sequential method
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplySequential(A.data(), B.data(), C_seq.data(), N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Sequential version time: " << elapsed.count() << " seconds\n";

    // Compare results
    float maxError = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        maxError = std::max(maxError, std::abs(C[i] - C_seq[i]));
    }
    std::cout << "Max error: " << maxError << "\n";

    return 0;
}
