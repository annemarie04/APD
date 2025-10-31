# scatter_matrix.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Procesul 0 creează matricea pxp
if rank == 0:
    A = np.arange(size * size).reshape(size, size)  # matrix
    print(f"[Rank {rank}] Full matrix A:\n{A}\n", flush=True)

    # Building send_data
    send_data = []
    for r in range(size):
        row = A[r, :]
        col = A[:, r]
        send_data.append((row, col))  # messages to be sent
else:
    send_data = None

# Scatter
data = comm.scatter(send_data, root=0)

# received data
row, col = data

print(f"[Rank {rank}] Received row = {row}")
print(f"[Rank {rank}] Received col = {col}\n")
