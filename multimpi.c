#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000  // Define el tamaño de la matriz

void sparse_matrix_multiply(double *A, double *B, double *C, int n, int rank, int size) {
    int i, j, k;
    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *local_C = NULL;

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (rand() % 10) / 10.0;
                B[i * N + j] = (rand() % 10) / 10.0;
            }
        }
    }

    local_C = (double *)malloc((N / size) * N * sizeof(double));

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, (N / size) * N, MPI_DOUBLE, A, (N / size) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    sparse_matrix_multiply(A, B, local_C, N, rank, size);

    MPI_Gather(local_C, (N / size) * N, MPI_DOUBLE, C, (N / size) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Parte de la matriz resultante (10x10):\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%f ", C[i * N + j]);
            }
            printf("\n");
        }
        free(A);
        free(B);
        free(C);
    }

    free(local_C);
    MPI_Finalize();
    return 0;
}
