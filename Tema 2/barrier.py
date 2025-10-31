from mpi4py import MPI
import time
import random
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def now():
    return datetime.now().strftime("%H:%M:%S")

# oot does initial work (sleep 4s)
if rank == 0:
    print(f"{now()} Rank {rank}: root starting initial work (sleeping 4s)", flush=True)
    time.sleep(4)
    print(f"{now()} Rank {rank}: root finished initial work", flush=True)

# All processes synchronize here
comm.Barrier()

# non-root processes announce they prepare
if rank != 0:
    print(f"{now()} Rank {rank}: received barrier; preparing to work (will sleep soon)", flush=True)

# every process simulates random work
random.seed(time.time() + rank * 12345) # seed random differently per rank for variability
work_time = random.uniform(0, 10)
print(f"{now()} Rank {rank}: starting work (sleep {work_time:.2f}s)...", flush=True)
time.sleep(work_time)
print(f"{now()} Rank {rank}: finished work after {work_time:.2f}s", flush=True)

# synchronize all processes again 
comm.Barrier()

# Only root prints the final conclusion
if rank == 0:
    print(f"{now()} Rank {rank}: all processes synchronized and finished. (size={size})", flush=True)
