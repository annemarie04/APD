
from mpi4py import MPI
import numpy as np
import math
import time

def get_hypercube_neighbors(rank, dimension):
    neighbors = []
    for i in range(dimension):
        neighbor = rank ^ (1 << i)  # XOR to flip the i-th bit
        neighbors.append(neighbor)
    return neighbors


def jacobi_hypercube(A, b, x0, max_iter=100, tol=1e-6):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check if size is a power of 2 (required for hypercube)
    if size & (size - 1) != 0:
        if rank == 0:
            print(f"Error: Number of processes ({size}) must be a power of 2 for hypercube topology")
        return None, 0, 0
    
    # Calculate hypercube dimension
    dimension = int(math.log2(size))
    
    n = len(b)
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
    
    num_rows = end_row - start_row
    
    if rank == 0:
        print(f"Starting Hypercube Jacobi: Solving {n} equations on {size} processes")
        print(f"Hypercube dimension: {dimension}")
        print(f"Each process has {dimension} neighbors")
        print(f"Tolerance: {tol}, Max iterations: {max_iter}")
    
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
    
    # Start timing
    start_time = time.time()
    
    # Start timing
    start_time = time.time()
    
    # Do the iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Each process computes its own rows using OLD values only (Jacobi)
        for i in range(start_row, end_row):
            # Calculate x[i] using the formula with x_old values
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x_old[j]
            
            x[i] = (b[i] - sigma) / A[i, i]
        
        # Process exchanges 
        for dim in range(dimension):
            # Find neighbor in this dimension
            neighbor = rank ^ (1 << dim)
            
            if neighbor < size:
                # Prepare my updated rows to send
                send_start, send_end = all_ranges[rank]
                send_data = x[send_start:send_end].copy()
                
                # Prepare to receive neighbor's rows
                recv_start, recv_end = all_ranges[neighbor]
                recv_data = np.zeros(recv_end - recv_start)
                
                # Exchange data with neighbor
                comm.Sendrecv(send_data, dest=neighbor,
                            recvbuf=recv_data, source=neighbor)
                
                # Update x 
                x[recv_start:recv_end] = recv_data
        
        send_start, send_end = all_ranges[rank]
        send_data = x[send_start:send_end].copy()
        
        # counts and displacements
        counts = [all_ranges[p][1] - all_ranges[p][0] for p in range(size)]
        displs = [all_ranges[p][0] for p in range(size)]

        comm.Allgatherv(send_data, [x, counts, displs, MPI.DOUBLE])
        
        diff = np.linalg.norm(x - x_old, ord=np.inf)
        max_diff = comm.allreduce(diff, op=MPI.MAX)
        
        if rank == 0 and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: max diff = {max_diff:.2e}")
        
        # Check if tolerance reached to stop
        if max_diff < tol:
            end_time = time.time()
            execution_time = end_time - start_time
            if rank == 0:
                print(f"\nTolerance reached! Converged in {iteration + 1} iterations.")
                print(f"Execution time: {execution_time:.6f} seconds")
            return x, iteration + 1, execution_time
        
    # Max iterations reached, so stop
    end_time = time.time()
    execution_time = end_time - start_time
    if rank == 0:
        print(f"\nReached maximum iterations ({max_iter})")
        print(f"Execution time: {execution_time:.6f} seconds")
    
    return x, max_iter, execution_time


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Generate the A, b and x0 for the system Ax = b
        n = 8
        # A - diagonally dominant matrix
        A = np.array([
            [10, -1,  2,  0,  0,  0,  0,  0],
            [-1, 11, -1,  3,  0,  0,  0,  0],
            [ 2, -1, 10, -1,  0,  0,  0,  0],
            [ 0,  3, -1,  8,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 12, -1,  0,  0],
            [ 0,  0,  0,  0, -1, 10,  2,  0],
            [ 0,  0,  0,  0,  0,  2,  9, -1],
            [ 0,  0,  0,  0,  0,  0, -1,  7]
        ], dtype=float)
        
        # b
        b = np.array([6, 25, -11, 15, 12, 9, 8, 6], dtype=float)
        
        # x0
        x0 = np.zeros(n)
        
        print("Matrix A:")
        print(A)
        print(f"\nVector b:")
        print(b)
        print(f"\nInitial value x0:")
        print(x0)
    else:
        A = None
        b = None
        x0 = None
    
    # Broadcast data to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x0 = comm.bcast(x0, root=0)
    
    # Solve using Hypercube Jacobi
    x, iterations, exec_time = jacobi_hypercube(A, b, x0, max_iter=100, tol=1e-6)
    
    # Root process displays results
    if rank == 0 and x is not None:
        print(f"\nSolution x:")
        print(x)

        # Numpy solution for comparison
        x_exact = np.linalg.solve(A, b)
        print(f"\nNumpy solution:")
        print(x_exact)
        
        # Calculate error
        error = np.linalg.norm(x - x_exact)
        print(f"\nError compared to exact solution: {error:.2e}")
        
        print(f"PERFORMANCE SUMMARY")
        print(f"Topology: Hypercube")
        print(f"Number of processes: {comm.Get_size()}")
        print(f"Hypercube dimension: {int(math.log2(comm.Get_size()))}")
        print(f"Iterations: {iterations}")
        print(f"Execution time: {exec_time:.6f} seconds")
        print(f"Time per iteration: {exec_time/iterations:.6f} seconds")


if __name__ == "__main__":
    main()
