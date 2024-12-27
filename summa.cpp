#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstdio>

// 初始化矩阵A和B
void initialize_matrices(int *A, int *B, int n, bool print) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = rand() % 10; // 随机初始化A
            B[i * n + j] = rand() % 10; // 随机初始化B
        }
    }

    // 打印矩阵A
    if (print) {
        std::cout << "Matrix A:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << A[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // 打印矩阵B
    if (print) {
        std::cout << "Matrix B:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << B[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// 验证矩阵乘法结果是否正确
bool verify_result(const std::vector<int> &A, const std::vector<int> &B, const std::vector<int> &C, int N) {
    std::vector<int> C_verify(N * N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C_verify[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    return C == C_verify;
}

// 局部矩阵块的乘法
void matrix_multiply(int *A, int *B, int *C, int local_L, int b) {
    for (int i = 0; i < local_L; i++) {
        for (int j = 0; j < local_L; j++) {
            for (int k = 0; k < b; k++) {
                C[i * local_L + j] += A[i * b + k] * B[k * local_L + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N, b;
    //srand(1); // 为了调试，使用固定的随机种子
    std::srand(std::time(0));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 从命令行读取N和b
    if (argc > 2) {
        N = atoi(argv[1]);
        b = atoi(argv[2]);
    }
    else {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <P> %s <matrix_size N> <block_size b>\n", argv[0]);
            fprintf(stderr, "Please provide the matrix size N and block size b.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // 确保进程数是平方数
    int sqrt_p = (int)sqrt(size);
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: The number of processes P must be a perfect square.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 确保矩阵大小能整除sqrt(P)
    if (N % sqrt_p != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: N must be divisible by sqrt(P).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_L = N / sqrt_p; // 每个进程负责的局部矩阵维度

    // 确保块大小满足要求
    if (b > local_L || local_L % b != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Block size b must be less than or equal to local matrix size L and L must be divisible by b.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 创建二维通信子
    int ndim = 2;
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm comm2D;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &comm2D);

    // 获取当前进程在2D网格中的坐标
    int id2D;
    int coords2D[2];
    MPI_Comm_rank(comm2D, &id2D);
    MPI_Cart_coords(comm2D, id2D, ndim, coords2D);
    // printf("Process ID: %d, Coordinates: (%d, %d)\n", id2D, coords2D[0], coords2D[1]);

    // 按行和列创建通信子
    int belongs[2];
    MPI_Comm commrow, commcol;

    belongs[0] = 0;
    belongs[1] = 1;
    MPI_Cart_sub(comm2D, belongs, &commrow);

    belongs[0] = 1;
    belongs[1] = 0;
    MPI_Cart_sub(comm2D, belongs, &commcol);

    // 根进程初始化全局矩阵
    std::vector<int> global_A;
    std::vector<int> global_B;
    std::vector<int> global_C;

    if (rank == 0) {
        global_A.resize(N * N);
        global_B.resize(N * N);
        global_C.resize(N * N, 0);
        initialize_matrices(global_A.data(), global_B.data(), N, true);
    } else {
        global_A.resize(N * N);
        global_B.resize(N * N);
    }
    
    double start_time = MPI_Wtime();  // 开始计时

    // 广播全局矩阵
    MPI_Bcast(global_A.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // 初始化子矩阵
    int local_size = local_L * local_L;
    std::vector<int> local_A(local_size);
    std::vector<int> local_B(local_size);
    std::vector<int> local_C(local_size, 0);

    // 计算当前进程负责的局部矩阵块的起始位置
    int row_start = coords2D[0] * local_L;
    int col_start = coords2D[1] * local_L;

    //printf("row_start=%d for procc %d\n", row_start, id2D);
    //printf("col_start=%d for procc %d\n", col_start, id2D);

    //从全局矩阵中分配子矩阵给各个进程
    for (int i = 0; i < local_L; i++) {
        for (int j = 0; j < local_L; j++) {
            local_A[i * local_L + j] = global_A[(row_start + i) * N + (col_start + j)];
            local_B[i * local_L + j] = global_B[(row_start + i) * N + (col_start + j)];
        }
    }

    // 输出各进程分配到的子矩阵
    // std::cout << "Rank " << id2D << " Submatrix A:" << std::endl;
    // for (int i = 0; i < local_L; i++) {
    //     for (int j = 0; j < local_L; j++) {
    //         std::cout << local_A[i * local_L + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Rank " << id2D << " Submatrix B:" << std::endl;
    // for (int i = 0; i < local_L; i++) {
    //     for (int j = 0; j < local_L; j++) {
    //         std::cout << local_B[i * local_L + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // 分块乘法
    // 由于局部矩阵可能很大，无法直接处理，所以进一步将局部矩阵划分为更小的块（block）进行计算。
    std::vector<int> Acol(local_L * b);  // 当前进程接收到的A的子块
    std::vector<int> Brow(b * local_L);  // 当前进程接收到的B的子块

    int num_block = N / b;
    for (int k = 0; k < num_block; k++) {  // 遍历全局矩阵block列
        //是否负责广播这个块的进程
        int index_col_in_matrix = k * b; // 一定是一个b的倍数，他走过了整k个block
        int index_process = k * b / local_L;
        if (coords2D[1] == index_process) {
            for (int i = 0; i < local_L; i++) {  // 遍历进程中的所有行
                for (int j = 0; j < b; j++) {  // 遍历当前block
                    int source_col = index_col_in_matrix - index_process * local_L + j; // 列块在局部矩阵中的列索引
                    // std::cout << "source_col:" << source_col << std::endl;
                    // if (source_col < local_L) {
                        Acol[i * b + j] = local_A[i * local_L + source_col];  // A_col存了[i*b: 当前block中的第几行, j: 当前block中的第几列] - 存了·一个block
                    // } 
                    // else {
                    //     Acol[i * b + j] = 0; // 边界情况填充 0
                    //     std::cout << "Acol out of bounder" << std::endl;
                    // }
                }
            }
            // printf("Process %d (coords %d,%d) broadcasting A block at k=%d in row_comm\n", id2D, coords2D[0], coords2D[1], k);
        }  // Acol - processer中的一列bolck

        MPI_Bcast(Acol.data(), local_L * b, MPI_INT, index_process % sqrt_p, commrow);

        // 如果当前进程负责广播 B 的行块
        if (coords2D[0] == index_process) {
            for (int i = 0; i < b; i++) {
                int source_row = index_col_in_matrix - index_process * local_L + i; // 行块在局部矩阵中的行索引
                // if (source_row < local_L) {
                    for (int j = 0; j < local_L; j++) {
                        Brow[i * local_L + j] = local_B[source_row * local_L + j];
                    }
                // }
                // else {
                //     for (int j = 0; j < local_L; j++) {
                //         Brow[i * local_L + j] = 0; // 边界情况填充 0
                //         std::cout << "Brow out of bounder" << std::endl;
                //     }
                // }
            }
            // printf("Process %d (coords %d,%d) broadcasting B block at k=%d in col_comm\n", id2D, coords2D[0], coords2D[1], k);
        }

        MPI_Bcast(Brow.data(), b * local_L, MPI_INT, (k * b / local_L) % sqrt_p, commcol);

        // printf("Process %d (coords %d,%d): Received A at k=%d:\n", id2D, coords2D[0], coords2D[1], k);
        // for (int i = 0; i < local_L; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < b; j++) {
        //         printf("%d ", Acol[i * b + j]);
        //     }
        //     printf("\n");
        // }

        // printf("Process %d (coords %d,%d): Received B at k=%d:\n", id2D, coords2D[0], coords2D[1], k);
        // for (int i = 0; i < b; i++) {
        //     printf("Row %d: ", i);
        //     for (int j = 0; j < local_L; j++) {
        //         printf("%d ", Brow[i * local_L + j]);
        //     }
        //     printf("\n");
        // }

        matrix_multiply(Acol.data(), Brow.data(), local_C.data(), local_L, b);
    }

    // 收集结果并验证
    std::vector<int> all_C;
    if (rank == 0) {
        all_C.resize(local_size * size);
    }
    MPI_Gather(local_C.data(), local_size, MPI_INT, all_C.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double local_execution_time = end_time - start_time;

    if (rank == 0) {
        // 遍历所有进程的结果
        for (int k = 0; k < size; k++) {
            int src_coords[2];
            // 获取进程的网格坐标
            MPI_Cart_coords(comm2D, k, 2, src_coords);
            // 当前进程k在网格中的行号和列号
            int src_row = src_coords[0];
            int src_col = src_coords[1];
            // 计算全局矩阵的行列索引
            for (int i = 0; i < local_L; i++) {
                int global_i = src_row * local_L + i;
                for (int j = 0; j < local_L; j++) {
                    int global_j = src_col * local_L + j;
                    global_C[global_i * N + global_j] = all_C[k * local_size + i * local_L + j];
                }
            }
        }

        printf("Total execution time (T): %f seconds\n", local_execution_time);
    }

    if (rank == 0) {
        printf("Global Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", global_C[i * N + j]);
            }
            printf("\n");
        }
    }

    if (rank == 0) {
        if (verify_result(global_A, global_B, global_C, N)) {
            printf("Result is correct!\n");
        } else {
            printf("Result is incorrect!\n");
        }
    }


    MPI_Comm_free(&commrow);
    MPI_Comm_free(&commcol);
    MPI_Comm_free(&comm2D);
    MPI_Finalize();
    return 0;
}