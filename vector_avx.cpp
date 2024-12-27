#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>

// 对齐分配器
template <typename T, std::size_t Alignment = 32>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() noexcept {}
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <typename U> struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return false;
}

// Assuming all matrices are of size N x N, stored in row-major order
const int N = 2048; // Adjustable to 512, 1024, 2048, etc.

void matrixMultiplyAVX(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i += 8) {
        for (int j = 0; j < n; ++j) {
            __m256 c[8] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (int k = 0; k < n; ++k) {
                __m256 b = _mm256_set1_ps(B[k * n + j]); // 更安全的广播方式
                for (int x = 0; x < 8; ++x) {
                    if (i + x < n) { // 防止越界
                        __m256 a = _mm256_loadu_ps(&A[(i + x) * n + k]);
                        c[x] = _mm256_fmadd_ps(a, b, c[x]);
                    }
                }
            }

            for (int x = 0; x < 8; ++x) {
                if (i + x < n) { // 防止越界
                    _mm256_storeu_ps(&C[(i + x) * n + j], c[x]);
                }
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
    // 使用对齐分配器分配内存
    std::vector<float, aligned_allocator<float>> A(N * N), B(N * N), C(N * N), C_seq(N * N);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX; // 随机浮点数 [0, 1)
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 初始化结果矩阵为零
    std::fill(C.begin(), C.end(), 0.0f);
    std::fill(C_seq.begin(), C_seq.end(), 0.0f);

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
        maxError = std::max(maxError, std::abs((C[i] - C_seq[i]) / C_seq[i]));
    }
    std::cout << "Max error: " << maxError << "\n";

    return 0;
}