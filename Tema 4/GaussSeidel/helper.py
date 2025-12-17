
from mpi4py import MPI
import numpy as np
import math
import time

def debug_print(message):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(f"[Rank {rank}] {message}")

def get_hypercube_neighbors(rank, dimension):
    neighbors = []
    for i in range(dimension):
        neighbor = rank ^ (1 << i)  # XOR to flip the i-th bit
        neighbors.append(neighbor)
    return neighbors


def gauss_seidel_hypercube( x0, max_iter=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check if size is a power of 2
    if size & (size - 1) != 0:
        if rank == 0:
            print(f"Error: Number of processes ({size}) must be a power of 2 for hypercube topology")
        return None, 0
    
    # Calculate hypercube dimension
    dimension = int(math.log2(size))
    
    n = len(x0)
    x = x0.copy()
    
    # Get hypercube neighbors
    neighbors = get_hypercube_neighbors(rank, dimension)
    
    # How many rows each process will get
    rows_per_proc = n // size
    remainder = n % size
    
    # Calculate the start and end row for each process
    if rank < remainder:
        start_row = rank * (rows_per_proc + 1)
        end_row = start_row + rows_per_proc + 1
    else:
        start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc
        end_row = start_row + rows_per_proc
    
    if rank == 0:
        print(f"Starting Hypercube Gauss-Seidel: Solving {n} equations on {size} processes")
        print(f"Hypercube dimension: {dimension}")
    
    # Store row ranges for all processes
    all_ranges = []
    for p in range(size):
        if p < remainder:
            p_start = p * (rows_per_proc + 1)
            p_end = p_start + rows_per_proc + 1
        else:
            p_start = remainder * (rows_per_proc + 1) + (p - remainder) * rows_per_proc
            p_end = p_start + rows_per_proc
        all_ranges.append((p_start, p_end))
    
    # Do the iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Each process updates its own rows
        for i in range(start_row, end_row):
            x[i] += 1
        
        # Hypercube-based all-gather using recursive doubling
        # In each dimension, exchange all data accumulated so far with neighbor
        # After d dimensions, all processes have all data
        
        # Track which ranges we currently have data for
        have_data = [False] * size
        have_data[rank] = True
        
        for dim in range(dimension):
            # Find neighbor in this dimension
            neighbor = rank ^ (1 << dim)
            
            if neighbor < size:
                # Determine what neighbor has: 
                # They have data from processes that match their bits up to dimension dim
                neighbor_has = [False] * size
                for p in range(size):
                    # Check if neighbor has data for process p
                    # Neighbor has p if p's XOR with neighbor differs only in lower 'dim' bits
                    diff = p ^ neighbor
                    if diff < (1 << dim):
                        neighbor_has[p] = True
                
                debug_print(f"Iteration {iteration}, Dimension {dim}, Rank {rank} exchanging with Neighbor {neighbor} who has {neighbor_has}")
                # Collect all data I currently have to send
                send_data_list = []
                for p in range(size):
                    if have_data[p]:
                        p_start, p_end = all_ranges[p]
                        send_data_list.append(x[p_start:p_end])
                
                send_data = np.concatenate(send_data_list) if send_data_list else np.array([])
                
                # Calculate receive buffer size based on what neighbor has
                recv_size = sum(all_ranges[p][1] - all_ranges[p][0] for p in range(size) if neighbor_has[p])
                recv_data = np.zeros(recv_size)
                
                # Exchange all accumulated data with neighbor
                comm.Sendrecv(send_data, dest=neighbor,
                            recvbuf=recv_data, source=neighbor)
                
                # Unpack received data into x
                offset = 0
                for p in range(size):
                    if neighbor_has[p]:
                        p_start, p_end = all_ranges[p]
                        count = p_end - p_start
                        x[p_start:p_end] = recv_data[offset:offset + count]
                        have_data[p] = True
                        offset += count
        
        if rank == 0:
            print(x)

    if rank == 0:
        print(f"\nReached maximum iterations ({max_iter})")
    
    return x


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Generate the A, b and x0 for the system Ax = b
        n = 100
        
        # x0
        x0 = np.zeros(n)
    else:
        x0 = None
    
    x0 = comm.bcast(x0, root=0)
    
    # Solve using Hypercube Gauss-Seidel
    x = gauss_seidel_hypercube(x0, max_iter=10)
    
    # Root process displays results
    if rank == 0 and x is not None:
        print(f"\nSolution x:")
        print(x)


if __name__ == "__main__":
    main()