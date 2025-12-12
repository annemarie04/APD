from mpi4py import MPI
import numpy as np
import time

def generate_test_data(n):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] += sum(np.abs(A[i]))
    b = np.random.rand(n) * 10
    return A, b


def gauss_seidel_ring(A, b, x0, max_iter=100, tol=1e-6):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = len(b)
    x = x0.copy()
    
    # Define ring neighbors
    next_rank = (rank + 1) % size  # Next process in ring
    prev_rank = (rank - 1 + size) % size  # Previous process in ring
    
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
        print(f"Starting Ring Gauss-Seidel: Solving {n} equations on {size} processes")
        print(f"Ring topology: Each process passes updates to next in ring")
        print(f"Tolerance: {tol}, Max iterations: {max_iter}")
    
    # Start timing
    start_time = time.time()
    
    # Do the iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Each process updates its own rows
        for i in range(start_row, end_row):
            # Calculate x[i] using the formula
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            
            x[i] = (b[i] - sigma) / A[i, i]
        
        # Ring communication: collect all updates by passing data around
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
        
        # Pass my updates around the ring (size-1) times
        send_start, send_end = all_ranges[rank]
        send_data = x[send_start:send_end].copy()
        
        for step in range(1, size):
            # Determine which process's data I'm receiving
            sender = (rank - step + size) % size
            recv_start, recv_end = all_ranges[sender]
            recv_data = np.zeros(recv_end - recv_start)
            
            # Send my original data forward, receive from previous
            comm.Sendrecv(send_data, dest=next_rank,
                         recvbuf=recv_data, source=prev_rank)
            
            # Update x with received data
            x[recv_start:recv_end] = recv_data
        
        # Check difference for convergence 
        diff = np.linalg.norm(x - x_old, ord=np.inf)
        
        # All processes need to agree on convergence
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
        # Generate the A, b and x0 for the the system Ax = b
        n = 100
        A, b = generate_test_data(n)
        
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
    
    # Solve using Ring Gauss-Seidel
    x, iterations, exec_time = gauss_seidel_ring(A, b, x0, max_iter=100, tol=1e-6)
    
    # Root process displays results
    if rank == 0:
        print(f"\nSolution x:")
        print(x)

        # Numpy solution
        x_exact = np.linalg.solve(A, b)
        print(f"Numpy solution:")
        print(x_exact)
        
        # Calculate error
        error = np.linalg.norm(x - x_exact)
        print(f"\nError compared to exact solution: {error:.2e}")
        
        print(f"PERFORMANCE SUMMARY")
        print(f"Topology: Ring")
        print(f"Number of processes: {comm.Get_size()}")
        print(f"Iterations: {iterations}")
        print(f"Execution time: {exec_time:.6f} seconds")
        print(f"Time per iteration: {exec_time/iterations:.6f} seconds")


if __name__ == "__main__":
    main()
