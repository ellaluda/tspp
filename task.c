#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define GRID_SIZE 2048        
#define ITERATION_LIMIT 1000
#define EPSILON 1e-6

void update_grid(double *curr_grid, double *new_grid, int rows, int columns) {
    double *above, *below, *current;
    for (int r = 1; r <= rows; r++) {
        above = curr_grid + (r - 1) * columns;
        current = curr_grid + r * columns;
        below = curr_grid + (r + 1) * columns;
        for (int c = 1; c < columns - 1; c++) {
            new_grid[r * columns + c] = 0.25 * (
                above[c] +
                below[c] +
                current[c - 1] +
                current[c + 1]
            );
        }
    }
}

double calculate_difference(double *matrix1, double *matrix2, int rows, int columns) {
    double sum = 0.0;
    for (int r = 1; r <= rows; r++) {
        for (int c = 0; c < columns; c++) {
            double delta = matrix1[r * columns + c] - matrix2[r * columns + c];
            sum += delta * delta;
        }
    }
    return sqrt(sum);
}

void setup_mpi_environment(int *argc, char ***argv, int *process_rank, int *num_processes) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, num_processes);
}

void allocate_memory(double **grid1, double **grid2, int row_count, int column_count) {
    *grid1 = (double*)malloc((row_count + 2) * column_count * sizeof(double));
    *grid2 = (double*)malloc((row_count + 2) * column_count * sizeof(double));

    if (*grid1 == NULL || *grid2 == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void initialize_data(double *grid, int rows, int columns) {
    for (int r = 1; r <= rows; r++) {
        for (int c = 0; c < columns; c++) {
            grid[r * columns + c] = rand() / (double)RAND_MAX;
        }
    }

    memset(grid, 0, columns * sizeof(double));
    memset(grid + (rows + 1) * columns, 0, columns * sizeof(double));
}

void execute_jacobi(double *curr_grid, double *next_grid, int rows, int columns, int rank, int total_procs) {
    double start = MPI_Wtime();
    double delta = 0.0;
    double *swap;
    int count;

    for (count = 0; count < ITERATION_LIMIT; count++) {
        update_grid(curr_grid, next_grid, rows, columns);

        double local_delta = calculate_difference(curr_grid, next_grid, rows, columns);
        MPI_Allreduce(&local_delta, &delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        MPI_Request send_request[2], recv_request[2];
        int num_requests = 0;

        if (rank > 0) {
            MPI_Isend(next_grid + columns, columns, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request[num_requests]);
            MPI_Irecv(curr_grid, columns, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request[num_requests]);
            num_requests++;
        }

        if (rank < total_procs - 1) {
            MPI_Isend(next_grid + rows*columns, columns, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_request[num_requests]);
            MPI_Irecv(curr_grid + (rows+1)*columns, columns, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request[num_requests]);
            num_requests++;
        }

        MPI_Waitall(num_requests, recv_request, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests, send_request, MPI_STATUSES_IGNORE);

        swap = curr_grid;
        curr_grid = next_grid;
        next_grid = swap;

        if (count % 100 == 0) {
            if (delta < EPSILON) break;
        }

    }

    double end = MPI_Wtime();
    double process_time = end - start;

    double max_time;
    MPI_Reduce(&process_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total execution time: %f seconds\n", max_time);
        printf("Max delta after %d iterations: %f\n", count, delta);
    }
}


void cleanup_mpi(double *grid1, double *grid2) {
    free(grid1);
    free(grid2);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    int rank, size, rows_per_proc, offset;
    double *grid_current, *grid_next;

    setup_mpi_environment(&argc, &argv, &rank, &size);

    int base_rows = GRID_SIZE / size;
    int extra = GRID_SIZE % size;
    if (rank < extra) {
        rows_per_proc = base_rows + 1;
        offset = rank * rows_per_proc;
    } else {
        rows_per_proc = base_rows;
        offset = rank * rows_per_proc + extra;
    }

    allocate_memory(&grid_current, &grid_next, rows_per_proc, GRID_SIZE);
    initialize_data(grid_current, rows_per_proc, GRID_SIZE);

    execute_jacobi(grid_current, grid_next, rows_per_proc, GRID_SIZE, rank, size);

    cleanup_mpi(grid_current, grid_next);

    return 0;
}
