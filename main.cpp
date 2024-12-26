#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

//初始化矩阵 A (N×N) 和向量 B (N×1)
void initialize_matrices(int *A, int *B, int n, bool print) {
    // 设置随机种子
    //std::srand(std::time(0));
    srand(1); // 为了调试，使用固定的随机种子
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = std::rand() % 10; // 随机初始化A
        }
        B[i] = std::rand() % 10; // 随机初始化 B
    }

    // 打印矩阵 A
    if (print) {
        std::cout << "Matrix A:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << A[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // 打印向量 B
    if (print) {
        std::cout << "Vector B:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << B[i] << std::endl;
        }
    }
}


// 串行计算 A * B
std::vector<int> serial_matrix_vector_mult(const int *A, const int *B, int n) {
    std::vector<int> C(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i] += A[i * n + j] * B[j];
        }
    }
    return C;
}


// 主函数
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool print = false;

    // 设置矩阵大小 N 和其他参数
    //const int N = 8;  // 矩阵大小（假设可以整除进程数）
    const int N = 192;

    int dims[2] = {0, 0};
    // dims[0] = P_0, dims[1] = P_1
    MPI_Dims_create(size, 2, dims);  // 创建 2D 网格
    int rows_per_proc = N / dims[0];
    int cols_per_proc = N / dims[1];

    // 检查能否整除
    if (N % dims[0] != 0 || N % dims[1] != 0) {
        if (rank == 0) {
            std::cerr << "Error: Matrix size (" << N << ") cannot be evenly divided among "
                    << dims[0] << " x " << dims[1] << " processes." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    
    if (print && rank == 0)
    {
        std::cout << "P0: " << dims[0] << std::endl;
        std::cout << "P1: " << dims[1] << std::endl;
    }

    //确保P>2时，网格维度>2
    if (size > 2)
    {   if (dims[0] <= 1 or dims[1] <= 1)
        {
            if (rank == 0)
            {
                fprintf(stderr, "The dimension is less than 1!");
            }
            std::cout << "The dimension is less than 1!" << std::endl;
            
            MPI_Finalize();
            return 0;
        }
    }

    //创建2D笛卡尔拓扑
    int ndim = 2;
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm comm2D;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &comm2D);

    int id2D;
    int coords2D[2];
    MPI_Comm_rank(comm2D, &id2D);
    MPI_Cart_coords(comm2D, id2D, ndim, coords2D);
    if (print)
    {
        printf("Process ID: %d, Coordinates: (%d, %d)\n", id2D, coords2D[0], coords2D[1]);
    }

    int belongs[2];
    MPI_Comm commrow, commcol;

    belongs[0] = 0;
    belongs[1] = 1;
    MPI_Cart_sub(comm2D, belongs, &commrow);

    belongs[0] = 1;
    belongs[1] = 0;
    MPI_Cart_sub(comm2D, belongs, &commcol);

    std::vector<int> global_A;
    std::vector<int> global_B;
    std::vector<int> global_C;

    bool print_init = false;

    if (print)
    {
        bool print_init = (rank == 0); // 只有根进程打印
    }
    
    if (rank == 0) {
        global_A.resize(N * N);
        global_B.resize(N * 1);
        global_C.resize(N * 1, 0);
        initialize_matrices(global_A.data(), global_B.data(), N, print_init);
    } else {
        global_A.resize(N * N);
        global_B.resize(N * 1);
    }

    double start_time = MPI_Wtime();  // 开始计时

    // 初始化本地矩阵块A,B和C
    std::vector<int> local_A(rows_per_proc * cols_per_proc, 0);
    std::vector<int> local_B(N, -1);
    std::vector<int> local_C(rows_per_proc, 0);


    //使用RMA将全局矩阵A按块分配给各进程

    //创建窗口a
    MPI_Win win_a;
    MPI_Win_create(
        rank == 0 ? global_A.data() : nullptr,
        rank == 0 ? (N*N*sizeof(int)) : 0,
        sizeof(int),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win_a
    );

    MPI_Win_fence(0, win_a);

    int row_start = coords2D[0] * rows_per_proc;
    int row_end = row_start + rows_per_proc - 1;
    int col_start = coords2D[1] * cols_per_proc;
    int col_end = col_start + cols_per_proc - 1;


    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_a);
    for (int i = 0; i < rows_per_proc; i++) {
        // 计算该行在global_A中的线性偏移
        MPI_Aint row_disp = ( (row_start + i) * N + col_start );
        // 拿cols_per_proc个int到local_A的相应位置
        MPI_Get(&local_A[i*cols_per_proc], cols_per_proc, MPI_INT,
                0,
                row_disp,
                cols_per_proc, MPI_INT,
                win_a);
    }
    MPI_Win_unlock(0, win_a);
    MPI_Win_fence(0, win_a);

    // 打印读取结果
    if (print)
    {
        std::cout << "Process " << rank << " received block local_A:" << std::endl;
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < cols_per_proc; j++) {
                std::cout << local_A[i * cols_per_proc + j] << " ";
            }
            std::cout << std::endl;
        }
        printf("\n");
    }
    
    // 创建RMA窗口共享向量b
    MPI_Win win_b;
    // 所有进程一起调用MPI_Win_create
    MPI_Win_create(
        rank == 0 ? global_B.data() : nullptr,
        (rank == 0 ? N * sizeof(int) : 0), //非root进程: 传入大小=0, 表示没有本地可暴露的内存
        sizeof(int),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win_b
    );

    //各进程通过RMA读取根进程的B

    //非root进程用 MPI_Get 拉取B到本地
    MPI_Win_fence(0, win_b);// 同步，确保rank=0已经写好B
    
    // 锁窗口 -> RMA Get -> 解锁
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_b);
    MPI_Get(local_B.data(), N, MPI_INT, 0, 0, N, MPI_INT, win_b);
    MPI_Win_unlock(0, win_b);

    MPI_Win_fence(0, win_b);

    if (print)
    {
        // 打印读取结果
        printf("Process %d got B = ", rank);
        for (int i = 0; i < N; i++) {
            printf("%d ", local_B[i]);
        }
        printf("\n");
    }
    
    //进程计算local_A * local_B
    int col_offset = coords2D[1] * cols_per_proc; // 在全局坐标里列的起始列索引
    //进程负责的矩阵块范围为[col_offset, col_offset + cols_per_proc - 1]
    for (int i = 0; i < rows_per_proc; i++) {
        int global_row = coords2D[0]*rows_per_proc + i;
        int sum = 0;
        for (int k = 0; k < cols_per_proc; k++) {
            sum += local_A[i*cols_per_proc + k] * local_B[col_offset + k];
        }
        local_C[i] = sum;
    }
    if (print)
    {
        // 打印local_C计算结果
        printf("Process %d calculated C = ", rank);
        for (int i = 0; i < rows_per_proc; i++) {
            printf("%d ", local_C[i]);
        }
        printf("\n");
    }
    

    //local_C行内累加,并统计到行根进程 (coords2D[1]==0)上
    std::vector<int> row_C_sum(rows_per_proc, 0);

    bool isRowRoot = (coords2D[1] == 0);

    // 创建RMA窗口共享列
    MPI_Win win_c_row;
    // 所有进程一起调用MPI_Win_create
    MPI_Win_create(
        // 若是列根则暴露 row_C_sum.size() 个int，否则暴露0
        isRowRoot ? row_C_sum.data() : nullptr,
        isRowRoot ? (rows_per_proc * sizeof(int)) : 0,
        sizeof(int),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &win_c_row
    );

    MPI_Win_fence(0, win_c_row);

    // 查找行根的rank
    int rowRootRank;
    {
        int coordsOfRoot[2] = { coords2D[0], 0 }; // 同一行，col=0
        MPI_Cart_rank(comm2D, coordsOfRoot, &rowRootRank);
        // std::cout << "rank " << rank << " root rank of row:" << rowRootRank << std::endl;
    }
    
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rowRootRank, 0, win_c_row);
    for (int i = 0; i < rows_per_proc; i++) {
        MPI_Aint disp = i; // 行内下标
        // 累加1个int
        MPI_Accumulate(
            &local_C[i], 1, MPI_INT,
            rowRootRank,
            disp, 1, MPI_INT,
            MPI_SUM,
            win_c_row
        );
    }
    MPI_Win_unlock(rowRootRank, win_c_row);

    // 保证行累加完成
    MPI_Win_fence(0, win_c_row);
    
    if (print)
    {
        // 打印行根数据
        if (isRowRoot) {
            std::cout << "[rank " << rank << ", coords=(" << coords2D[0] << "," << coords2D[1]
                    << ")] row_C_sum = ";
            for (int i = 0; i < rows_per_proc; i++) {
                std::cout << row_C_sum[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    //统计到global_C
    // 只在“第0列”那几个进程里做 gather
    if (isRowRoot) {
        // 在行子通信器里，此时有 dims[1] 个进程
        // 每个进程send都可以是rows_per_proc
        MPI_Gather(row_C_sum.data(), rows_per_proc, MPI_INT, rank == 0 ? global_C.data() : nullptr, rows_per_proc, MPI_INT, 0, commcol);
    }

    double end_time = MPI_Wtime();  // 结束计时

    if (rank == 0) {
        // 串行计算A*B=C
        std::vector<int> C_serial = serial_matrix_vector_mult(global_A.data(), global_B.data(), N);

        // 比较global_C与C_serial
        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (global_C[i] != C_serial[i]) {
                correct = false;
                break;
            }
        }

        if (print)
        {
            std::cout << "C_serial: ";
            for (int i = 0; i < N; i++) {
                std::cout << C_serial[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // 输出比较结果
        if (correct) {
            std::cout << "Calculation correct!" << std::endl;
        } else {
            std::cout << "Calculation wrong!" << std::endl;
        }
    }

    if (print)
    {
        // 打印global_C
        if (rank == 0) {
            std::cout << "Final Global C: ";
            for (int i = 0; i < N; i++) {
                std::cout << global_C[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    if (rank == 0) {
        std::cout << "Total execution time (T): " << end_time - start_time << " seconds." << std::endl;
    }

    MPI_Win_free(&win_a);
    MPI_Win_free(&win_b);
    MPI_Win_free(&win_c_row);

    MPI_Finalize();
    return 0;
}