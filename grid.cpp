#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

// 常量定义
const int N = 10; // 网格宽度
const int M = 10; // 网格高度
const int ITERATIONS = 50; // 最大迭代次数
const int K = 10; // 如果稳定 K 次则停止
#define INDEX(row, col) ((row) * N + (col)) // 用于计算一维数组的索引

// 打印网格的函数
void print_grid(const std::vector<int>& grid, int iter) {
    std::cout << "Iteration " << iter << ":\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // 根据细胞状态打印 'O' 或 '.'
            std::cout << (grid[i * N + j] ? 'O' : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::string(20, '-') << '\n';
}

// 初始化滑翔者图案的函数
void initialize_glider(std::vector<int>& grid, int rank) {
    if (rank == 0) { // 只有 rank 0 初始化图案
        if (M >= 3 && N >= 3) { // 确保网格足够大
            // 设置滑翔者的初始状态
            grid[1 * N + 2] = 1;
            grid[2 * N + 3] = 1;
            grid[3 * N + 1] = 1;
            grid[3 * N + 2] = 1;
            grid[3 * N + 3] = 1;
        }
    }
    // 将网格广播到所有进程
    MPI_Bcast(grid.data(), N * M, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) { // 只有 rank 0 打印初始网格
        print_grid(grid, 0);
    }
}

// 计算活邻居的函数
int count_neighbors(const std::vector<int>& grid, int x, int y, bool use_torus) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue; // 跳过中心细胞
            int nx, ny; // 邻居坐标
            if (use_torus) { // 如果使用环形拓扑
                nx = (x + i + M) % M; // 行坐标循环
                ny = (y + j + N) % N; // 列坐标循环
            } else {
                nx = std::min(std::max(x + i, 0), M - 1); // 行坐标限制在边界内
                ny = std::min(std::max(y + j, 0), N - 1); // 列坐标限制在边界内
            }
            count += grid[nx * N + ny]; // 累加邻居的活细胞数
        }
    }
    return count; // 返回活邻居数
}

// 计算下一代细胞状态的函数
void compute_next_generation(const std::vector<int>& current, std::vector<int>& next, int start_row, int end_row, bool use_torus) {
    // 遍历每一行
    for (int x = start_row; x < end_row; ++x) {
        // 遍历每一列
        for (int y = 0; y < N; ++y) {
            int live_neighbors = count_neighbors(current, x, y, use_torus); // 计算活邻居数
            // 根据规则更新下一代状态
            if (current[x * N + y] == 1) { // 如果当前细胞是活的
                next[x * N + y] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0; // 生存
            } else { // 如果当前细胞是死的
                next[x * N + y] = (live_neighbors == 3) ? 1 : 0; // 复活
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // 初始化 MPI

    int rank, size; // 进程的 rank 和总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的 rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取总进程数

    bool use_torus = true; // 使用环形拓扑

    // 初始化网格和上一个网格
    std::vector<int> grid(N * M, 0);
    std::vector<int> previous_grid(N * M, 0);

    initialize_glider(grid, rank); // 初始化滑翔者图案

    // 计算每个进程处理的行数
    int rows_per_proc = M / size; 
    int remainder = M % size; // 计算剩余行数
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0); // 每个进程的行数
    std::vector<int> local_grid((local_rows + 2) * N, 0); // +2 用于幽灵行
    std::vector<int> next_local_grid((local_rows + 2) * N, 0); // 下一代局部网格

    // 为每个进程准备发送和接收的行数和偏移量
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N; // 计算每个进程需要发送的行数
        displs[i] = offset; // 偏移量
        offset += sendcounts[i]; // 更新偏移量
    }

    double start_time = MPI_Wtime(); // 记录开始时间

    int live_count = 0; // 活细胞总数
    int is_stable = 0; // 用于判断是否稳定

    // 迭代计算细胞状态
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // 将全局网格分散到各个进程的局部网格中
        MPI_Scatterv(grid.data(), sendcounts.data(), displs.data(), MPI_INT, local_grid.data() + N, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Request requests[4]; // 存储非阻塞请求的数组
        int rank_up = rank - 1; // 上一个进程的 rank
        int rank_down = rank + 1; // 下一个进程的 rank
        int req_count = 0; // 请求计数器

        // 处理环形拓扑
        if (rank == 0) {
            rank_up = size - 1; // rank 0 的上一个进程为最后一个进程
        } else if (rank == size - 1) {
            rank_down = 0; // 最后一个进程的下一个进程为 rank 0
        }

        // 发送和接收幽灵行数据
        if (rank_up >= 0) {
            MPI_Isend(local_grid.data() + N, N, MPI_INT, rank_up, 0, MPI_COMM_WORLD, &requests[req_count++]); // 发送上幽灵行
            MPI_Irecv(local_grid.data(), N, MPI_INT, rank_up, 1, MPI_COMM_WORLD, &requests[req_count++]); // 接收下幽灵行
        }
        if (rank_down < size) {
            MPI_Isend(local_grid.data() + local_rows * N, N, MPI_INT, rank_down, 1, MPI_COMM_WORLD, &requests[req_count++]); // 发送下幽灵行
            MPI_Irecv(local_grid.data() + (local_rows + 1) * N, N, MPI_INT, rank_down, 0, MPI_COMM_WORLD, &requests[req_count++]); // 接收上幽灵行
        }

        // 等待所有非阻塞请求完成
        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        // 计算下一代细胞状态
        compute_next_generation(local_grid, next_local_grid, 1, local_rows + 1, use_torus);

        // 收集各个进程的下一代局部网格
        MPI_Gatherv(next_local_grid.data() + N, sendcounts[rank], MPI_INT, grid.data(), sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

        int local_live_count = 0; // 当前进程的活细胞计数
        // 计算当前进程的活细胞总数
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < N; ++j) {
                local_live_count += local_grid[(i + 1) * N + j]; // 注意局部网格的索引偏移
            }
        }

        // 汇总所有进程的活细胞数
        MPI_Allreduce(&local_live_count, &live_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Now live cells : " << live_count << std::endl; // 打印活细胞总数
        }

        // 判断是否稳定
        if (rank == 0) {
            is_stable = (grid == previous_grid) ? 1 : 0; // 检查当前网格和上一个网格是否相同
            previous_grid = grid; // 更新上一个网格
        }

        // 广播稳定状态给所有进程
        MPI_Bcast(&is_stable, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (is_stable == 1 && iter >= K) { // 如果稳定且迭代次数达到 K
            if (rank == 0) {
                std::cout << "Grid is stable!\n"; // 打印稳定信息
            }
            break; // 退出迭代
        }

        // 只有 rank 0 打印当前网格状态
        if (rank == 0) {
            print_grid(grid, iter);
        }
    }

    double end_time = MPI_Wtime(); // 记录结束时间

    // 只有 rank 0 打印最终结果
    if (rank == 0) {
        std::cout << "Total live cells at the end: " << live_count << std::endl; // 打印活细胞总数
        std::cout << "Total time taken: " << end_time - start_time << " seconds" << std::endl; // 打印总时间
    }

    MPI_Finalize(); // 结束 MPI
    return 0; // 返回 0 表示程序成功执行
}
