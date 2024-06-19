import numpy as np
from scipy.sparse import random, csr_matrix
from multiprocessing import Pool
import matplotlib.pyplot as plt

def multiply_row(row, sparse_matrix):
    return row.dot(sparse_matrix).toarray()[0]

def parallel_multiply(sparse_matrix1, sparse_matrix2, num_workers=4):
    with Pool(num_workers) as pool:
        result = pool.starmap(multiply_row, [(sparse_matrix1.getrow(i), sparse_matrix2) for i in range(sparse_matrix1.shape[0])])
    return csr_matrix(result)

# Establecer dimensiones de las matrices
rows, cols = 1000, 1000

# Crear dos matrices dispersas aleatorias
sparse_matrix1 = random(rows, cols, density=0.01, format='csr', dtype=np.float64)
sparse_matrix2 = random(rows, cols, density=0.01, format='csr', dtype=np.float64)

# Multiplicar las matrices dispersas en paralelo
result_matrix = parallel_multiply(sparse_matrix1, sparse_matrix2)

# Mostrar algunos detalles de la matriz resultado
print("Forma de la matriz resultado:", result_matrix.shape)
print("Número de elementos no cero en la matriz resultado:", result_matrix.nnz)

# Convertir a formato denso y mostrar parte de la matriz resultado
result_matrix_dense = result_matrix.toarray()
print("Parte de la matriz resultado (10x10):\n", result_matrix_dense[:10, :10])

# Visualización de una parte de la matriz resultado
plt.spy(result_matrix, markersize=1)
plt.title("Visualización de la matriz dispersa resultante")
plt.show()
