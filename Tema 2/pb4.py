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
    n = 43  # You can change this value
    
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
        
        # Vector: all elements are 1
        x = np.arange(n, dtype=np_dtype)
        x **= 2
        print("Original Matrix A:")
        print(A)
        print("Original Vector x:")
        print(x)
        
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
        pack_offset = 0
        
        for proc in range(size):
            start_col = displs[proc]
            num_cols = counts[proc]
            
            # Copy columns for this process
            for col in range(num_cols):
                actual_col = start_col + col
                for row in range(n):
                    A_packed[pack_offset + col * n + row] = A[row, actual_col]
            
            pack_offset += num_cols * n
        print(A_packed)
        # Prepare send counts and displacements for matrix scattering
        sendcounts_A = [count * n for count in counts]  # Each column has n elements
        displs_A = [0]
        for i in range(1, size):
            displs_A.append(displs_A[i-1] + sendcounts_A[i-1])
    
    else:
        A_packed = None
        x = None
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
    x_local = np.zeros(cols_local, dtype=np_dtype)
    
    # Scatter the matrix columns
    # A_local needs to be flattened for receiving
    comm.Scatterv([A_packed, sendcounts_A, displs_A, mpi_type],
                  A_local.ravel(), root=0)
    
    # Scatter the vector elements
    comm.Scatterv([x, counts, displs, mpi_type],
                  x_local, root=0)
    
    debug = rank == 0
    
    print_if(debug, f"[{rank}]Matrix columns:\n{A_local}")
    print_if(debug, f"[{rank}]Vector elements: {x_local}")
    print_if(debug, "")

    result_local = np.zeros(len(A_local), dtype=np_dtype)
    participating_size = min(size, n) # processors participating in mult
    print(f"{participating_size=}")

    # Ring topology
    next_proc = (rank + 1) % participating_size
    prev_proc = (rank - 1 + participating_size) % participating_size

    current_vector = x_local.copy()
    current_index = displs[(rank+1)%size]
    print(f"{rank=} {current_vector=}")

    if rank < participating_size:
        # Ring algorithm: circulate vector segments
        for step in range(participating_size):
            # Compute sum of products
            current_index = (current_index - len(current_vector) + n) % n
            for column_index in range(len(A_local)):
                for i in range(len(current_vector)):
                    matr_value = A_local[column_index, (current_index + i) % n]
                    vec_value = current_vector[i]
                    addition = matr_value * vec_value
                    result_local[column_index] += addition
                    print_if(debug, f"{rank=} {step=}\n\t{(current_index + i) % n=} "
                             f"\n\t{result_local[column_index]=} "
                             f"\n\t{addition=} "
                             f"\n\t{matr_value=} "
                             f"\n\t{vec_value=}")

            # Send and receive next vector segment
            recv_buf = np.zeros(counts[(rank-step-1+participating_size) % participating_size], dtype=np_dtype) 
            send_buf = current_vector.copy()
            comm.Send(send_buf, dest=next_proc, tag=step)
            comm.Recv(recv_buf, source=prev_proc, tag=step)
            current_vector = recv_buf.copy()
            print_if(debug, f"{len(current_vector)=}")


    result = comm.gather(result_local, root=0)
    if rank == 0:
        reported = np.array(list(itertools.chain(*result)))
        expected = np.dot(x, A)
        print(f"reported={reported}")
        print(f"expected={np.dot(x, A)}")
        print(f"Correct? {np.all(reported == expected)}")
    print(f"[{rank}]Local result: {result_local}")

if __name__ == "__main__":
    main()
