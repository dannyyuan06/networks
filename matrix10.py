import numpy as np

# Define your 4x4 matrix
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Raise the matrix to the power of 10
result = np.linalg.matrix_power(matrix, 10)

# Print the result
print(result)