# broadcast_dict.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = None

if rank == 0:
    data = {
        'key1': [3, 24.62, 9 + 4j],
        'key2': ('fmi', 'unibuc')
    }
    print(f"[Rank {rank}] Root created data: {data}")

data = comm.bcast(data, root=0)

print(f"[Rank {rank}] Received data: {data}")
