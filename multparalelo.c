#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000  // Define el tamaño de la matriz

// Función para multiplicar matrices dispersas
void sparse_matrix_multiply(double *A, double *B, double *C, int n) {
    int i, j, k;
    #pragma omp parallel for private(j, k) shared(A, B, C, n)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    // Asignar memoria para las matrices
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    // Inicializar las matrices con valores aleatorios
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (rand() % 10) / 10.0;
            B[i * N + j] = (rand() % 10) / 10.0;
            C[i * N + j] = 0.0;
        }
    }

    // Multiplicar las matrices
    sparse_matrix_multiply(A, B, C, N);

    // Imprimir una pequeña porción de la matriz resultante
    printf("Parte de la matriz resultante (10x10):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Liberar la memoria
    free(A);
    free(B);
    free(C);

    return 0;
}

