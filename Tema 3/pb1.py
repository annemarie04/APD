import itertools
from mpi4py import MPI
import numpy as np
import sys

def print_if(cond, msg):
    if cond:
        print(msg)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Matrix and vector size
    n = 5 # You can change this value
    
    # Data types
    np_dtype = np.float64
    mpi_type = MPI.DOUBLE
    
    # Root process (rank 0) prepares the data
    if rank == 0:
        # Create the matrix and vector
        A = np.zeros((n, n), dtype=np_dtype)
        for i in range(n):
            for j in range(n):
                A[i, j] = i * n + j + 1

        # Matrix B: identity matrix
        B = np.eye(n).astype(np_dtype)
        print("Original Matrix A:")
        print(A)
        print("Original Matrix B:")
        print(B)
        
        # Calculate how many columns each process gets
        base_cols = n // size
        extra_cols = n % size
        
        # Some processes may get one extra column if n is not divisible by size
        counts = []
        for i in range(size):
            if i < extra_cols:
                counts.append(base_cols + 1)
            else:
                counts.append(base_cols)
        
        # Calculate starting positions for each process
        displs = [0]
        for i in range(1, size):
            displs.append(displs[i-1] + counts[i-1])
        
        start_col = np.zeros(size, dtype=int)
        end_col = np.zeros(size, dtype=int)
        for i in range(size):
            start_col[i] = displs[i]
            end_col[i] = displs[i] + counts[i] - 1

        # Prepare matrix data for scattering
        # We need to pack matrix columns contiguously for each process
        A_packed = np.zeros(n * n, dtype=np_dtype)
        B_packed = np.zeros(n * n, dtype=np_dtype)
        pack_offset = 0
        
        for proc in range(size):
            start_col = displs[proc]
            num_cols = counts[proc]
            
            # Copy columns for this process
            for col in range(num_cols):
                actual_col = start_col + col
                for row in range(n):
                    B_packed[pack_offset + col * n + row] = B[row, actual_col]
            
            for row in range(n):
                for col in range(n):
                    A_packed[row * n + col] = A[row, col]
            pack_offset += num_cols * n
        print(A_packed)
        print(B_packed)
        # Prepare send counts and displacements for matrix scattering
        sendcounts_A = [count * n for count in counts]  # Each column has n elements
        displs_A = [0]
        for i in range(1, size):
            displs_A.append(displs_A[i-1] + sendcounts_A[i-1])
    
    else:
        A_packed = None
        B_packed = None
        counts = None
        displs = None
        sendcounts_A = None
        displs_A = None
    
    # Broadcast the distribution information to all processes
    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)
    sendcounts_A = comm.bcast(sendcounts_A, root=0)
    displs_A = comm.bcast(displs_A, root=0)
    
    # Each process determines how many columns it will receive
    cols_local = counts[rank]
    
    # Allocate local buffers
    A_local = np.zeros((cols_local, n), dtype=np_dtype)
    B_local = np.zeros((cols_local, n), dtype=np_dtype)
    
    # Scatter the matrix columns
    # A_local needs to be flattened for receiving
    comm.Scatterv([A_packed, sendcounts_A, displs_A, mpi_type],
                  A_local.ravel(), root=0)
    
    # Scatter the vector elements
    comm.Scatterv([B_packed, sendcounts_A, displs_A, mpi_type],
                  B_local.ravel(), root=0)

    debug = rank == 0
    
    print_if(debug, f"[{rank}]Matrix A columns:\n{A_local}")
    print_if(debug, f"[{rank}]Matrix B (Identity) elements: {B_local}")
    print_if(debug, "")

    result_local = []
    participating_size = min(size, n) # processors participating in mult
    print(f"{participating_size=}")

    # Ring topology
    next_proc = (rank + 1) % participating_size
    prev_proc = (rank - 1 + participating_size) % participating_size

    current_vector = B_local.copy()
    start_index = displs[(rank+1)%size]
    row_start = displs[(rank)%size]
    print(f"{rank=} {start_index=}")


    if rank < participating_size:
        # Ring algorithm: circulate vector segments
        for step in range(participating_size):
            # Compute sum of products
            start_index = (start_index - len(current_vector) + n) % n
            

            for row_index_A in range(len(A_local)):
                current_index = start_index
                for col_index_B in range(len(current_vector)):
                    matr_value = A_local[row_index_A]
                    vec_value = current_vector[col_index_B]
                    addition = np.dot(matr_value, vec_value)
                    result_local.append((row_start + row_index_A, current_index, addition))
                    if(rank == 0): 
                        print(f"Setting result[{row_start + row_index_A}, {current_index + col_index_B}] = {addition} for {matr_value} x {vec_value}")
                    # print(addition)
                    current_index = (current_index + 1) % n

            # Send and receive next vector segment
            recv_buf = np.zeros((counts[(rank-step-1+participating_size) % participating_size], n), dtype=np_dtype)
            send_buf = current_vector.copy()
            comm.Send(send_buf, dest=next_proc, tag=step)
            comm.Recv(recv_buf, source=prev_proc, tag=step)
            current_vector = recv_buf.copy()


    all_triplets = comm.gather(result_local, root=0)
    if rank == 0:
        result = np.zeros((n,n), dtype=np_dtype)
        for list in all_triplets:
            for (i, j, v) in list:
                result[i, j] = v
        expected = np.dot(A, B)
        print(f"reported={result}")
        print(f"expected={expected}")
        print(f"Correct? {np.all(result == expected)}")

if __name__ == "__main__":
    main()
