# M(n, L, n) = n^(2−αβ) * L^β + n^2, and:
# Algorithm SMP(A, B)
# Input: Two n × n matrices A and B.
# Output: The product AB.
# 1. Let ak be the number of non-zero elements in A∗k, the k-th column of A, for 1 ≤ k ≤ n.
# 2. Let bk be the number of non-zero elements in Bk∗, the k-th row of B, for 1 ≤ k ≤ n.
# 3. Let π be a permutation for which aπ(1)bπ(1) ≥ aπ(2)bπ(2) ≥ . . . ≥ aπ(n)bπ(n).
# 4. Find an L,0 ≤ L ≤ n that minimizes M(n, L, n) + sigma_k>L ( aπ(k)bπ(k))
# 5. Let I = {π(1), . . . , π(L)} and J = {π(L + 1), . . . , π(n)}.
# 6. Compute C1 ← A∗IBI∗ using the fast dense rectangular matrix multiplication algorithm.
# 7. Compute C2 ← A∗JBJ∗ using the naive sparse matrix multiplication algorithm.
# 8. Output C1 + C2
# please give me the python code of SMP

import numpy as np
from scipy.sparse import csr_matrix
import timeit
import matplotlib.pyplot as plt
def O3multi(A, B):
    """
    Perform matrix multiplication using the simplest cubic time complexity approach (O(n^3)).
    
    Parameters:
    - A: NumPy array, the first matrix
    - B: NumPy array, the second matrix
    
    Returns:
    - C: NumPy array, the result of the matrix multiplication A * B
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions for multiplication")

    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C

def SMP(A, B):
    n = A.shape[0]

    # Step 1: Use np.sum instead of np.count_nonzero
    ak = np.array([np.sum(A[:, k] != 0) for k in range(n)])

    # Step 2: Use np.sum instead of np.count_nonzero
    bk = np.array([np.sum(B[k, :] != 0) for k in range(n)])
    
    # Step 3
    pi = np.argsort(ak * bk)[::-1]

    # Function to calculate M(n, L, n) + sigma_k>L(aπ(k)bπ(k))
    def objective_function(L):
        return np.sum(ak[pi[L:]] * bk[pi[L:]])

    # Step 4: Find L that minimizes the objective function
    L_values = range(n + 1)
    alpha = 0.294
    beta = 0.536
    L_optimal = min(L_values, key=lambda L: (n**(2 - alpha * beta) * L**beta + n**2) + objective_function(L))

    # Step 5
    I = pi[:L_optimal]
    J = pi[L_optimal:]

    # Step 6: Compute C1 using the fast dense rectangular matrix multiplication algorithm
    C1 = O3multi(A[:, I] , B[I, :])

    # Step 7: Compute C2 using the naive sparse matrix multiplication algorithm
    #TODO
    C2 = csr_matrix(A[:, J]) @ csr_matrix(B[J, :])

    # Step 8: Output C1 + C2
    return C1 + C2
# Example usage:
# A and B are numpy arrays representing two matrices
# C = SMP(A, B)
# A = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
# B = np.array([[6, 0, 7], [0, 8, 0], [9, 0, 10]])

# C = SMP(A, B)
# print(C)
# C = A@B
# print(C)
# Generate random dense and sparse matrices with size over 500x500



# # judge time complexity

# size = 500 


# dense_size = (size, size)  # Adjust the size as needed
# sparse_size = (size, size)  # Adjust the size as needed

# dense_A = np.random.rand(*dense_size)
# dense_B = np.random.rand(*dense_size)

# threshold = 0.9  # Adjust the threshold as needed
# sparse_A = np.random.rand(*dense_size)
# sparse_A[sparse_A < threshold] = 0
# sparse_B = np.random.rand(*dense_size)
# sparse_B[sparse_B < threshold] = 0
# # Time SMP(A, B)
# # judge correrctness
# # tolerance = 1e-6  # Adjust the tolerance as needed
# # are_equal = np.allclose(SMP(dense_A, dense_B), dense_A @ dense_B, atol=tolerance)
# # print(are_equal)


# # Time SMP(A, B)
# smp_time = timeit.timeit(lambda: SMP(dense_A, dense_B), number=1)
# print(f"SMP dense Time: {smp_time:.6f} seconds")
# # Time A @ B
# matmul_time = timeit.timeit(lambda: csr_matrix(dense_A) @ csr_matrix(dense_B), number=1)
# print(f"Matrix Multiplication dense Time_csr: {matmul_time:.6f} seconds")
# matmul_time = timeit.timeit(lambda: dense_A @ dense_B, number=1)
# print(f"Matrix Multiplication dense Time_np: {matmul_time:.6f} seconds")

# # Time SMP(A, B)
# smp_time = timeit.timeit(lambda: SMP(sparse_A, sparse_B), number=1)
# print(f"SMP sparese Time: {smp_time:.6f} seconds")
# # Time A @ B
# matmul_time = timeit.timeit(lambda: csr_matrix(sparse_A) @ csr_matrix(sparse_B), number=1)
# print(f"Matrix Multiplication sparse Time_csr: {matmul_time:.6f} seconds")
# matmul_time = timeit.timeit(lambda: sparse_A @ sparse_B, number=1)
# print(f"Matrix Multiplication sparse Time_np: {matmul_time:.6f} seconds")

# Function to measure time for a given size
def measure_time(size):
    dense_size = (size, size)

    dense_A = np.random.rand(*dense_size)
    dense_B = np.random.rand(*dense_size)

    threshold = 0.9
    sparse_A = np.random.rand(*dense_size)
    sparse_A[sparse_A < threshold] = 0
    sparse_B = np.random.rand(*dense_size)
    sparse_B[sparse_B < threshold] = 0

    # Time SMP(A, B) for dense matrices
    smp_dense_time = timeit.timeit(lambda: SMP(dense_A, dense_B), number=1)

    # Time SMP(A, B) for sparse matrices
    smp_sparse_time = timeit.timeit(lambda: SMP(sparse_A, sparse_B), number=1)

    # Time A @ B for dense matrices using csr_matrix
    matmul_dense_csr_time = timeit.timeit(lambda: csr_matrix(dense_A) @ csr_matrix(dense_B), number=1)

    # Time A @ B for dense matrices using numpy
    matmul_dense_np_time = timeit.timeit(lambda: O3multi(dense_A, dense_B), number=1)

    # Time A @ B for sparse matrices using csr_matrix
    matmul_sparse_csr_time = timeit.timeit(lambda: csr_matrix(sparse_A) @ csr_matrix(sparse_B), number=1)

    # Time A @ B for sparse matrices using numpy
    matmul_sparse_np_time = timeit.timeit(lambda: O3multi(sparse_A,sparse_B), number=1)

    return smp_dense_time, smp_sparse_time, matmul_dense_csr_time, matmul_dense_np_time, matmul_sparse_csr_time, matmul_sparse_np_time

# Define a range of sizes
sizes = [10,50,100, 200]

# Measure time for each size
results = [measure_time(size) for size in sizes]

# Extract results for plotting
smp_dense_times, smp_sparse_times, matmul_dense_csr_times, matmul_dense_np_times, matmul_sparse_csr_times, matmul_sparse_np_times = zip(*results)

# Plotting
plt.figure(figsize=(10, 6))

# Plot dense matrices
plt.plot(sizes, np.log(smp_dense_times), label='SMP Dense', marker='o')
plt.plot(sizes, np.log(matmul_dense_csr_times), label='MatMul Dense CSR', marker='o')
plt.plot(sizes, np.log(matmul_dense_np_times), label='MatMul Dense O3', marker='o')

# Plot sparse matrices
plt.plot(sizes, np.log(smp_sparse_times), label='SMP Sparse', marker='o')
plt.plot(sizes, np.log(matmul_sparse_csr_times), label='MatMul Sparse CSR', marker='o')
plt.plot(sizes, np.log(matmul_sparse_np_times), label='MatMul Sparse O3', marker='o')

plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.title('Comparison of SMP and Matrix Multiplication')
plt.legend()
plt.grid(True)
plt.show()