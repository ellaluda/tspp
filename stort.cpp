#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

// Function to check if two arrays are equal
bool arrays_are_equal(const vector<int>& arr1, const vector<int>& arr2) {
    return arr1 == arr2;
}

// Function to perform parallel merge sort
void parallel_merge_sort(vector<int>& A, vector<int>& B, int left, int right, int depth, int max_depth) {
    if (right - left <= 1) return;

    if (depth < max_depth) {
        int middle = (left + right) / 2;

        #pragma omp task shared(A, B)
        parallel_merge_sort(A, B, left, middle, depth + 1, max_depth);

        #pragma omp task shared(A, B)
        parallel_merge_sort(A, B, middle, right, depth + 1, max_depth);

        #pragma omp taskwait

        inplace_merge(A.begin() + left, A.begin() + middle, A.begin() + right);
    } else {
        sort(A.begin() + left, A.begin() + right);
    }
}

int main(int argc, char* argv[]) {
    int N, p;
    cout << "Enter the number of N and the number of p: ";
    cin >> N >> p;

    vector<int> A(N);
    vector<int> B(N);
    vector<int> C(N);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 100000);

    srand((unsigned)time(nullptr));
    generate(A.begin(), A.end(), [&]() { return dis(gen); });
    C = A;

    auto start_time = chrono::high_resolution_clock::now();
    sort(C.begin(), C.end());
    auto end_time = chrono::high_resolution_clock::now();
    double qsort_time = chrono::duration<double>(end_time - start_time).count();

    cout << "qsort time:\t" << qsort_time << " seconds" << endl;

    B = A;
    start_time = chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(p)
    {
        #pragma omp single
        parallel_merge_sort(B, C, 0, N, 0, p);
    }

    end_time = chrono::high_resolution_clock::now();
    double parallel_time = chrono::duration<double>(end_time - start_time).count();

    bool is_equal = arrays_are_equal(B, C);
    double percentage = (parallel_time / qsort_time) * 100;

    cout << "Threads:\t" << p << endl;
    cout << "Parallel:\t" << parallel_time << " seconds" << endl;
    cout << "Par/sort:\t" << percentage << "%" << endl;
    cout << "Arr " << (is_equal ? "equal" : "different") << endl;

    return 0;
}
