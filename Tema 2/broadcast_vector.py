from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    vec = np.array([1, 2, 3, 4, 5], dtype='i')
    print(f"[Rank {rank}] Root created vector: {vec}")
else:
    vec = np.empty(5, dtype='i')  

comm.Bcast([vec, MPI.INT], root=0)

print(f"[Rank {rank}] Received vector: {vec}")
