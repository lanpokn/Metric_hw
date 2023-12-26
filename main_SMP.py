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

def SMP(A, B):
    n = A.shape[0]

    # Step 1
    ak = np.array([np.count_nonzero(A[:, k]) for k in range(n)])
    # Step 2
    bk = np.array([np.count_nonzero(B[k, :]) for k in range(n)])
    
    # Step 3
    pi = np.argsort(ak * bk)[::-1]

    # Function to calculate M(n, L, n) + sigma_k>L(aπ(k)bπ(k))
    def objective_function(L):
        return np.sum(ak[pi[L:]] * bk[pi[L:]])

    # Step 4: Find L that minimizes the objective function
    L_values = range(n + 1)
    alpha = 0.294
    beta = 0.533
    L_optimal = min(L_values, key=lambda L: (n**(2 - alpha * beta) * L**beta + n**2) + objective_function(L))

    # Step 5
    I = pi[:L_optimal]
    J = pi[L_optimal:]

    # Step 6: Compute C1 using the fast dense rectangular matrix multiplication algorithm
    C1 = A[:, I] @ B[I, :]

    # Step 7: Compute C2 using the naive sparse matrix multiplication algorithm
    C2 = csr_matrix(A[:, J]) @ csr_matrix(B[J, :])

    # Step 8: Output C1 + C2
    return C1 + C2

# Example usage:
# A and B are numpy arrays representing two matrices
# C = SMP(A, B)
A = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
B = np.array([[6, 0, 7], [0, 8, 0], [9, 0, 10]])

C = SMP(A, B)
print(C)
C = A@B
print(C)