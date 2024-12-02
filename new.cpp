#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>

const int N = 10; // 矩阵宽度
const int M = 10; // 矩阵高度
const int ITERATIONS = 50; // 最大迭代次数
const int K = 10; // 当稳定状态达到K次时停止迭代


// 打印网格状态
void print_grid(const std::vector<int>& grid, int iter) {
    std::cout << "Iteration " << iter << ":\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << (grid[i * N + j] ? 'O' : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::string(20, '-') << '\n';
}

// 初始化滑翔机图案
void initialize_glider(std::vector<int>& grid, int rank, int size) {
    if (rank == 0) {
        if (M >= 3 && N >= 3) { // 确保滑翔机能放下
            grid[1 * N + 2] = 1;
            grid[2 * N + 3] = 1;
            grid[3 * N + 1] = 1;
            grid[3 * N + 2] = 1;
            grid[3 * N + 3] = 1;
        }
    }
    MPI_Bcast(grid.data(), N * M, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        print_grid(grid, 0); // 初始网格输出
    }
}

// 计算活细胞邻居的数量
int count_neighbors(const std::vector<int>& grid, int x, int y, bool use_torus) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue;
            int nx, ny;
            if (use_torus) {
                nx = (x + i + M) % M;
                ny = (y + j + N) % N;
            } else {
                nx = std::min(std::max(x + i, 0), M - 1);
                ny = std::min(std::max(y + j, 0), N - 1);
            }
            count += grid[nx * N + ny];
        }
    }
    return count;
}

// 计算下一代网格
void compute_next_generation(const std::vector<int>& current, std::vector<int>& next, int start_row, int end_row, bool use_torus) {
    for (int x = start_row; x < end_row; ++x) {
        for (int y = 0; y < N; ++y) {
            int live_neighbors = count_neighbors(current, x, y, use_torus);
            if (current[x * N + y] == 1) {
                next[x * N + y] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
            } else {
                next[x * N + y] = (live_neighbors == 3) ? 1 : 0;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); //初始化MPI环境

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //获取当前进程rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //获取总进程数

    bool use_torus = false;
    if (argc > 1) {
        use_torus = (std::string(argv[1]) == "1");
    }

    std::vector<int> grid(N * M, 0); // 当前网格
    std::vector<int> next_grid(N * M, 0); // 下一代网格
    std::vector<int> previous_grid(N * M, 0); // 上一代网格

    initialize_glider(grid, rank, size); //初始化滑翔机

    int rows_per_proc = M / size; //每个进程的行数
    int remainder = M % size; //余数，分配给前ramainder个进程
    //为每个进程分配本地网络，如果rank小于remainder，需要多处理一行
    std::vector<int> local_grid((rows_per_proc + (rank < remainder ? 1 : 0)) * N);

    // 用于Scatterv函数的计数和位移向量
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int offset = 0;

    //计算每个进程的发送计数和位移
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    double start_time = MPI_Wtime(); // 使用 MPI_Wtime 开始计时

    int live_count = 0; //当前活细胞总数
    //int prev_live_count = -1; // 上一次迭代活细胞数量
    //int stable_iterations = 0; //？用于判断稳定状态（活细胞数量连续两次迭代没发生变化）
    int is_stable = 0;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        MPI_Scatterv(grid.data(), sendcounts.data(), displs.data(), MPI_INT, local_grid.data(), local_grid.size(), MPI_INT, 0, MPI_COMM_WORLD);

        //compute_next_generation(grid, next_grid, displs[rank] / N, (displs[rank] + sendcounts[rank]) / N);

        compute_next_generation(grid, next_grid, displs[rank] / N, (displs[rank] + sendcounts[rank]) / N, use_torus);
        MPI_Gatherv(next_grid.data() + displs[rank], sendcounts[rank], MPI_INT, grid.data(), sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

        // 计算全局活细胞数
        int local_live_count = 0;
        for (int i = displs[rank] / N; i < (displs[rank] + sendcounts[rank]) / N; ++i) {
            for (int j = 0; j < N; ++j) {
                local_live_count += next_grid[i * N + j];
            }
        }
        MPI_Allreduce(&local_live_count, &live_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 检查稳定性
        if (rank == 0) {
            // 检查是否相同
                is_stable = (grid == previous_grid) ? 1 : 0;
                previous_grid = grid;
        }

        std::cout << "is_stable: " << is_stable << "\n";

        MPI_Bcast(&is_stable, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (is_stable == 1 && iter >= K) {
            std::cout << "Grid is stable!" << "\n";
            break;
        }

        if (rank == 0) {
            print_grid(grid, iter);
        }

        
        std::swap(grid, next_grid);
    }

    double end_time = MPI_Wtime(); // 使用 MPI_Wtime 结束计时

    if (rank == 0) {
        std::cout << "Total live cells at the end: " << live_count << std::endl;
        std::cout << "Total time taken: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}