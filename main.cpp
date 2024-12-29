#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <ctime>

// 定义辅助宏：索引映射
#define INDEX(x, y, z, Nx, Ny) ((z) * (Ny) * (Nx) + (y) * (Nx) + (x))

// 定义派生数据类型的函数
MPI_Datatype create_face_type(int fixed_dim, int fixed_value, int N_x, int N_y, int N_z) {
    int sizes[3]    = {N_z + 2, N_y + 2, N_x + 2}; // 全局网格尺寸，包括 ghost cells
    int subsizes[3] = {0, 0, 0};                     // 子数组尺寸
    int starts[3]   = {0, 0, 0};                     // 起始位置

    MPI_Datatype face_type;

    if (fixed_dim == 0) { // x-face
        subsizes[0] = N_z;
        subsizes[1] = N_y;
        subsizes[2] = 1;
        starts[0]   = 1;
        starts[1]   = 1;
        starts[2]   = fixed_value;
    }
    else if (fixed_dim == 1) { // y-face
        subsizes[0] = N_z;
        subsizes[1] = 1;
        subsizes[2] = N_x;
        starts[0]   = 1;
        starts[1]   = fixed_value;
        starts[2]   = 1;
    }
    else if (fixed_dim == 2) { // z-face
        subsizes[0] = 1;
        subsizes[1] = N_y;
        subsizes[2] = N_x;
        starts[0]   = fixed_value;
        starts[1]   = 1;
        starts[2]   = 1;
    }

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &face_type);
    MPI_Type_commit(&face_type);
    return face_type;
}

int main(int argc, char** argv) {
    const int N = 240; // 全局网格大小
    const int n_iter = 100; // 最大迭代次数

    bool print = false;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建笛卡尔拓扑
    int dims[3] = {0, 0, 0}; 
    MPI_Dims_create(size, 3, dims);

    if (N % dims[0] != 0 || N % dims[1] != 0 || N % dims[2] != 0) {
        if (rank == 0) {
            std::cerr << "Error: Global grid size N must be divisible by dims[0], dims[1], and dims[2].\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (print)
    {
        // 如果满足条件，输出拓扑结构信息
        if (rank == 0) {
            std::cout << "Valid topology created with dimensions: (" 
                      << dims[0] << ", " << dims[1] << ", " << dims[2] << ").\n";
        }
    }

    if (size >= 8)
    {
        // 检查每个维度是否满足 > 1
        bool valid = true;
        for (int i = 0; i < 3; i++) {
            if (dims[i] <= 1) {
                valid = false;
                break;
            }
        }

        // 如果不满足条件，终止程序
        if (!valid) {
            if (rank == 0) {
                std::cerr << "Error: The number of processes cannot be divided into a 3D topology with each dimension > 1. Exiting program.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // 终止所有进程
        }

        // 如果满足条件，输出拓扑结构信息
        if (rank == 0) {
            std::cout << "Valid topology created with dimensions: (" 
                        << dims[0] << ", " << dims[1] << ", " << dims[2] << ").\n";
        }
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/1, &cart_comm);

    // 三维进程坐标
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // 定义局部网格在不同方向的大小(N / dims[i])
    int N_x = N / dims[0];
    int N_y = N / dims[1];
    int N_z = N / dims[2];

    // 包含 ghost cells 的局部网格尺寸
    int ext_x = N_x + 2; 
    int ext_y = N_y + 2;
    int ext_z = N_z + 2;

    // 分配网格
    std::vector<double> u_old(ext_x * ext_y * ext_z, 0.0);
    std::vector<double> u_new(ext_x * ext_y * ext_z, 0.0);

    // 初始化：填充初始值
    srand(time(NULL) + rank); // 为每个进程设置不同的随机数种子
    for (int z = 1; z <= N_z; z++) { 
        for (int y = 1; y <= N_y; y++) { 
            for (int x = 1; x <= N_x; x++) { 
                int idx = INDEX(x, y, z, ext_x, ext_y);
                assert(idx >= 0 && idx < ext_x * ext_y * ext_z);
                // u_old[idx] = static_cast<double>(rand()) / RAND_MAX;  // 随机初值[0.0, 1.0) 
                u_old[idx] = 1.0;
            }
        }
    }

    // 打印u_old
    if(print)
    {
        std::cout << "Process rank: " << rank
                  << ", Coordinates: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")" 
                  << ", Local grid size: (" << N_x << ", " << N_y << ", " << N_z << ")" 
                  << ", Local grid:\n"
                  << std::endl;
        for (int z = 1; z <= N_z; ++z) {
            for (int y = 1; y <= N_y; ++y) {
                for (int x = 1; x <= N_x; ++x) {
                    int idx = INDEX(x, y, z, ext_x, ext_y);
                    assert(idx >= 0 && idx < ext_x * ext_y * ext_z);
                    std::cout << std::setw(3) << u_old[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // 定义派生类型
    MPI_Datatype x_face_type_send = create_face_type(0, N_x, N_x, N_y, N_z);      // 发送到右邻居
    MPI_Datatype x_face_type_recv = create_face_type(0, 0, N_x, N_y, N_z);        // 接收自左邻居

    MPI_Datatype y_face_type_send = create_face_type(1, N_y, N_x, N_y, N_z);      // 发送到上邻居
    MPI_Datatype y_face_type_recv = create_face_type(1, 0, N_x, N_y, N_z);        // 接收自下邻居

    MPI_Datatype z_face_type_send = create_face_type(2, N_z, N_x, N_y, N_z);      // 发送到后邻居
    MPI_Datatype z_face_type_recv = create_face_type(2, 0, N_x, N_y, N_z);        // 接收自前邻居


    // 雅可比迭代
    double global_diff = 0.0; // 用于存储全局范数
    double start_time = MPI_Wtime(); // 记录开始时间

    for (int iter = 0; iter < n_iter; iter++)
    {
        // 找到左/右/上/下/前/后邻居的进程编号：
        int rank_left, rank_right, rank_down, rank_up, rank_front, rank_back;
        MPI_Cart_shift(cart_comm, 0, 1, &rank_left, &rank_right);
        MPI_Cart_shift(cart_comm, 1, 1, &rank_down, &rank_up);
        MPI_Cart_shift(cart_comm, 2, 1, &rank_front, &rank_back);

        // 交换边界
        // 发送到右邻居，接收自左邻居
        if (rank_right != MPI_PROC_NULL) {
            // 使用派生类型进行通信
            MPI_Sendrecv(u_old.data(), 1, x_face_type_send, rank_right, 0,
                         u_old.data(), 1, x_face_type_recv, rank_right, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 发送到左邻居，接收自右邻居
        if (rank_left != MPI_PROC_NULL) {
            MPI_Sendrecv(u_old.data(), 1, x_face_type_send, rank_left, 0,
                         u_old.data(), 1, x_face_type_recv, rank_left, 0,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 发送到上邻居，接收自下邻居
        if (rank_up != MPI_PROC_NULL) {
            MPI_Sendrecv(u_old.data(), 1, y_face_type_send, rank_up, 1,
                         u_old.data(), 1, y_face_type_recv, rank_up, 1,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 发送到下邻居，接收自上邻居
        if (rank_down != MPI_PROC_NULL) {
            MPI_Sendrecv(u_old.data(), 1, y_face_type_send, rank_down, 1,
                         u_old.data(), 1, y_face_type_recv, rank_down, 1,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 发送到后邻居，接收自前邻居
        if (rank_back != MPI_PROC_NULL) {
            MPI_Sendrecv(u_old.data(), 1, z_face_type_send, rank_back, 2,
                         u_old.data(), 1, z_face_type_recv, rank_back, 2,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 发送到前邻居，接收自后邻居
        if (rank_front != MPI_PROC_NULL) {
            MPI_Sendrecv(u_old.data(), 1, z_face_type_send, rank_front, 2,
                         u_old.data(), 1, z_face_type_recv, rank_front, 2,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // 更新Jacobi内部值
        for (int z = 1; z <= N_z; z++) {
            for (int y = 1; y <= N_y; y++) {
                for (int x = 1; x <= N_x; x++) {
                    int idx = INDEX(x, y, z, ext_x, ext_y); // 当前网格点的索引
                    assert(idx >= 0 && idx < ext_x * ext_y * ext_z);

                    // 计算 6 个邻居点的索引
                    int idx_x_minus = INDEX(x - 1, y, z, ext_x, ext_y);
                    int idx_x_plus = INDEX(x + 1, y, z, ext_x, ext_y);
                    int idx_y_minus = INDEX(x, y - 1, z, ext_x, ext_y);
                    int idx_y_plus = INDEX(x, y + 1, z, ext_x, ext_y);
                    int idx_z_minus = INDEX(x, y, z - 1, ext_x, ext_y);
                    int idx_z_plus = INDEX(x, y, z + 1, ext_x, ext_y);

                    // 计算Jacobi更新
                    u_new[idx] = (u_old[idx_x_minus] + u_old[idx_x_plus]
                                + u_old[idx_y_minus] + u_old[idx_y_plus]
                                + u_old[idx_z_minus] + u_old[idx_z_plus]) / 6.0;
                }
            }
        }

        // 计算全局范数
        double local_diff = 0.0; // 局部范数
        for (int z = 1; z <= N_z; z++) {
            for (int y = 1; y <= N_y; y++) {
                for (int x = 1; x <= N_x; x++) {
                    int idx = INDEX(x, y, z, ext_x, ext_y);
                    assert(idx >= 0 && idx < ext_x * ext_y * ext_z);
                    double diff = u_new[idx] - u_old[idx];
                    local_diff += diff * diff; 
                }
            }
        }
        MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm); // 汇总到根进程
        if (rank == 0 && (iter == 1 || iter == n_iter - 1)) {
            global_diff = sqrt(global_diff / (N * N * N));
            std::cout << "Iteration: " << iter << " global_diff:" << global_diff << std::endl;
        }

        // 交换指针
        u_old.swap(u_new);
    }

    double end_time = MPI_Wtime(); // 记录结束时间
    double total_time = end_time - start_time; // 计算运行时间

    // 输出结果（仅根进程）
    if (rank == 0) {
        std::cout << "Final difference: " << global_diff << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
    }

    // 释放派生数据类型
    MPI_Type_free(&x_face_type_send);
    MPI_Type_free(&x_face_type_recv);
    MPI_Type_free(&y_face_type_send);
    MPI_Type_free(&y_face_type_recv);
    MPI_Type_free(&z_face_type_send);
    MPI_Type_free(&z_face_type_recv);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
