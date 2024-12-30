#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <cstdlib> // for srand, rand
#include <ctime>   // for time

// 定义辅助宏：索引映射
#define INDEX(x, y, z, Nx, Ny) ((z) * (Ny) * (Nx) + (y) * (Nx) + (x))
// x,y,z：三维网格的坐标。
// Nx：每行包含的元素数量。
// Ny：每个平面（即在同一z层上）的行数。

int main(int argc, char** argv) {
    const int N = 240;    // 全局网格大小
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

    // 如果满足条件，输出拓扑结构信息
    if (rank == 0) {
        std::cout << "Valid topology created with dimensions: (" 
                  << dims[0] << ", " << dims[1] << ", " << dims[2] << ").\n";
    }

    // 若进程数很多，示例中做个简单检查(可按需要修改或删除)
    if (size >= 8) {
        bool valid = true;
        for (int i = 0; i < 3; i++) {
            if (dims[i] <= 1) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            if (rank == 0) {
                std::cerr << "Error: The number of processes cannot be divided into a 3D topology with each dimension > 1.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    // reorder=1 允许MPI在创建笛卡尔通信器时重新映射进程排布
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/1, &cart_comm);

    // 获取该进程在笛卡尔拓扑中的坐标
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // 定义局部网格大小
    int N_x = N / dims[0]; 
    int N_y = N / dims[1];
    int N_z = N / dims[2];

    // 包含 ghost cells 的局部网格维度（左右、上下、前后各多一层）
    int ext_x = N_x + 2; 
    int ext_y = N_y + 2;
    int ext_z = N_z + 2;

    // 分配网格 (u_old / u_new)
    std::vector<double> u_old(ext_x * ext_y * ext_z, 0.0);
    std::vector<double> u_new(ext_x * ext_y * ext_z, 0.0);

    // 初始化：用随机数填充内部区域（跳过ghost层）
    srand(time(NULL) + rank); // 确保每个进程有不同的随机种子
    for (int z = 1; z <= N_z; z++) {
        for (int y = 1; y <= N_y; y++) {
            for (int x = 1; x <= N_x; x++) {
                int idx = INDEX(x, y, z, ext_x, ext_y);
                // rand() / RAND_MAX -> [0,1)
                u_old[idx] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }

    // 定义三种派生类型：XY平面、YZ平面、XZ平面
    MPI_Datatype xy_plane_type, yz_plane_type, xz_plane_type;

    // 定义XY平面数据类型（内存中连续）
    MPI_Type_contiguous(ext_x * ext_y, MPI_DOUBLE, &xy_plane_type);
    MPI_Type_commit(&xy_plane_type);

    // 定义YZ平面数据类型（每行间隔N_x元素）
    MPI_Type_vector(ext_z * ext_y, 1, ext_x, MPI_DOUBLE, &yz_plane_type);
    MPI_Type_commit(&yz_plane_type);

    // 定义XZ平面数据类型（每层间隔N_x * N_y元素）
    MPI_Type_vector(ext_z, ext_x, ext_x * ext_y, MPI_DOUBLE, &xz_plane_type);
    MPI_Type_commit(&xz_plane_type);

    // 找到左/右/上/下/前/后邻居的进程编号：
    int rank_left, rank_right, rank_down, rank_up, rank_front, rank_back;
    MPI_Cart_shift(cart_comm, 0, 1, &rank_left,  &rank_right);
    MPI_Cart_shift(cart_comm, 1, 1, &rank_down,  &rank_up);
    MPI_Cart_shift(cart_comm, 2, 1, &rank_front, &rank_back);

    double start_time = MPI_Wtime(); 
    double global_diff = 0.0;  // 用于存储全局差异的范数

    // 交换边界
    for (int iter = 0; iter < n_iter; iter++) {
        MPI_Request req[12];
        int mpi_err;

        // 左/右通信：发送x=N_x给右边, 接收右边到x=N_x+1; 发送x=1给左边, 接收左边到x=0
        if (rank_right != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(N_x, 0, 0, ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(N_x + 1, 0, 0, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, yz_plane_type, rank_right, 0, cart_comm, &req[0]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << iter << ", to right neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, yz_plane_type, rank_right, 0, cart_comm, &req[1]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << iter << ", from right neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[0] = MPI_REQUEST_NULL;
            req[1] = MPI_REQUEST_NULL;
        }

        if (rank_left != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(1, 0, 0, ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(0, 0, 0, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, yz_plane_type, rank_left, 0, cart_comm, &req[2]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << iter << ", to left neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, yz_plane_type, rank_left, 0, cart_comm, &req[3]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << iter << ", from left neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[2] = MPI_REQUEST_NULL;
            req[3] = MPI_REQUEST_NULL;
        }

        // 下/上通信：发送y=N_y给上边, 接收上边到y=N_y+1; 发送y=1给下边, 接收下边到y=0
        if (rank_up != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(0, N_y, 0, ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(0, N_y + 1, 0, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, xz_plane_type, rank_up, 1, cart_comm, &req[4]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << ", to up neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, xz_plane_type, rank_up, 1, cart_comm, &req[5]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << ", from up neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[4] = MPI_REQUEST_NULL;
            req[5] = MPI_REQUEST_NULL;
        }

        if (rank_down != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(0, 1, 0, ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(0, 0, 0, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, xz_plane_type, rank_down, 1, cart_comm, &req[6]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << ", to down neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, xz_plane_type, rank_down, 1, cart_comm, &req[7]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << ", from down neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[6] = MPI_REQUEST_NULL;
            req[7] = MPI_REQUEST_NULL;
        }
        // 前/后通信：发送z=N_z给后边, 接收后边到z=N_z+1; 发送z=1给前边, 接收前边到z=0
        if (rank_back != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(0, 0, N_z,     ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(0, 0, N_z + 1, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, xy_plane_type, rank_back, 2, cart_comm, &req[8]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << iter << ", to back neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, xy_plane_type, rank_back, 2, cart_comm, &req[9]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << iter << ", from back neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[8] = MPI_REQUEST_NULL;
            req[9] = MPI_REQUEST_NULL;
        }

        if (rank_front != MPI_PROC_NULL) {
            double* sendbuf = &u_old[INDEX(0, 0, 1, ext_x, ext_y)];
            double* recvbuf = &u_old[INDEX(0, 0, 0, ext_x, ext_y)];
            mpi_err = MPI_Isend(sendbuf, 1, xy_plane_type, rank_front, 2, cart_comm, &req[10]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Isend failed at iter " << iter << ", to front neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
            mpi_err = MPI_Irecv(recvbuf, 1, xy_plane_type, rank_front, 2, cart_comm, &req[11]);
            if (mpi_err != MPI_SUCCESS) {
                std::cerr << "MPI_Irecv failed at iter " << iter << ", from front neighbor.\n";
                MPI_Abort(MPI_COMM_WORLD, mpi_err);
            }
        } else {
            req[10] = MPI_REQUEST_NULL;
            req[11] = MPI_REQUEST_NULL;
        }

        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);

        // 进行雅可比迭代更新 (内部点)
        for (int z = 1; z <= N_z; z++) {
            for (int y = 1; y <= N_y; y++) {
                for (int x = 1; x <= N_x; x++) {
                    int idx = INDEX(x, y, z, ext_x, ext_y);

                    // 计算 6 个邻居点的索引
                    int idx_x_minus = INDEX(x - 1, y, z, ext_x, ext_y);
                    int idx_x_plus = INDEX(x + 1, y, z, ext_x, ext_y);
                    int idx_y_minus = INDEX(x, y - 1, z, ext_x, ext_y);
                    int idx_y_plus = INDEX(x, y + 1, z, ext_x, ext_y);
                    int idx_z_minus = INDEX(x, y, z - 1, ext_x, ext_y);
                    int idx_z_plus = INDEX(x, y, z + 1, ext_x, ext_y);

                    // 简单平均
                    u_new[idx] = (u_old[idx_x_minus] + u_old[idx_x_plus]
                                 + u_old[idx_y_minus] + u_old[idx_y_plus]
                                 + u_old[idx_z_minus] + u_old[idx_z_plus]) / 6.0;
                }
            }
        }

        // 计算本次迭代与上次迭代的差值 (局部)
        double local_diff = 0.0;
        for (int z = 1; z <= N_z; z++) {
            for (int y = 1; y <= N_y; y++) {
                for (int x = 1; x <= N_x; x++) {
                    int idx = INDEX(x, y, z, ext_x, ext_y);
                    double diff = u_new[idx] - u_old[idx];
                    local_diff += diff * diff;
                }
            }
        }
        MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
        global_diff = sqrt(global_diff) / (N * N * N);

        if (rank == 0 && (iter == 1 || iter == n_iter - 1)) {
            std::cout << "Iteration: " << iter 
                      << ", global_diff: " << global_diff << std::endl;
        }

        u_old.swap(u_new);
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    if (rank == 0) {
        std::cout << "Final difference: " << global_diff << std::endl;
        std::cout << "Total time " << total_time << " seconds" << std::endl;
    }

    // 释放自定义类型
    MPI_Type_free(&xy_plane_type);
    MPI_Type_free(&yz_plane_type);
    MPI_Type_free(&xz_plane_type);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
