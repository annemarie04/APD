from mpi4py import MPI
import numpy as np

def matvec(comm, A, x, n):
    # Gather all local vectors to reconstruct the full vector of size n
    local_size = x.size
    all_sizes = comm.allgather(local_size)
    
    # Calculate displacements for Allgatherv
    displacements = [0]
    for i in range(len(all_sizes) - 1):
        displacements.append(displacements[-1] + all_sizes[i])
    
    # Create global vector of correct size n
    xg = np.zeros(n, dtype='d')
    
    # Use Allgatherv to handle different local sizes
    comm.Allgatherv([x, MPI.DOUBLE], [xg, all_sizes, displacements, MPI.DOUBLE])
    
    y = np.dot(A, xg)
    return y

comm = MPI.COMM_WORLD # communicator
rank = comm.Get_rank() # process rank
p = comm.Get_size() # total number of processes


n = 20  # matrix size
m_local = n // p  # local rows per process
remainder = n % p  # case n is not divisible by p

# Some processes get one extra row if there's a remainder
if rank < remainder:
    m_local += 1
    start_row = rank * (n // p + 1)
else:
    start_row = remainder * (n // p + 1) + (rank - remainder) * (n // p)

# Generate the matrix A and vector x
A_local = np.zeros((m_local, n), dtype='d')

# Each process gets m_local rows, creating an identity-like matrix
for i in range(m_local):
    global_row = start_row + i
    if global_row < n:
        A_local[i, global_row] = 1  # diagonal structure

# local portion of x - each process has m_local elements
x_local = np.zeros(m_local, dtype='d')
for i in range(m_local):
    global_idx = start_row + i
    if global_idx < n:
        x_local[i] = global_idx + 1.0  # local part of vector x = [1,2,3,...,n]
        
# Run the matrix-vector multiplication
y_local = matvec(comm, A_local, x_local, n)
print(f"Rank {rank} -> y_local = {y_local}")