"""
Matrix Column Distribution and Reconstruction
---------------------------------------------
Usage: mpiexec -n <desired_number_of_processes> python matrix_columns.py <matrix_size>
Example: mpiexec -n 4 --oversubscribe python matrix_columns.py 8
"""

from mpi4py import MPI
import numpy as np
import sys

def parse_arguments():
    if len(sys.argv) != 2:
        print("Usage: mpiexec -n <desired_number_of_processes> python matrix_columns.py <matrix_size>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            print("Error: Matrix size n must be a positive integer")
            sys.exit(1)
        return n
    except ValueError:
        print("Error: Matrix size must be a valid integer")
        sys.exit(1)

def calculate_column_distribution(n, p):
    column_info = []
    
    # Calculate base number of columns per process and remainder
    base_cols = n // p
    extra_cols = n % p
    
    start_col = 0
    for rank in range(p):
        # Processes with rank < extra_cols get one extra column
        num_cols = base_cols + (1 if rank < extra_cols else 0)
        
        column_info.append({
            'rank': rank,
            'start_col': start_col,
            'num_cols': num_cols,
            'total_elements': n * num_cols  # n_rows * num_cols
        })
        
        start_col += num_cols
    
    return column_info

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()
    
    # Parse arguments (only root process)
    if rank == 0:
        n = parse_arguments()
        print(f"Matrix size: {n}×{n}")
        print(f"Number of processes: {p}")
    else:
        n = None
    
    # Broadcast matrix size to all processes
    n = comm.bcast(n, root=0)
    
    # Calculate column distribution
    if rank == 0:
        # Generate the original matrix with random elements
        original_matrix = np.random.rand(n, n)
        print(f"Generated {n}×{n} random matrix")
        
        # Calculate column distribution information for all processes
        column_info = calculate_column_distribution(n, p)
        
        print(f"Column distribution:")
        for info in column_info:
            if info['num_cols'] > 0:
                end_col = info['start_col'] + info['num_cols'] - 1
                print(f"  Process {info['rank']}: columns {info['start_col']}-{end_col} ({info['num_cols']} columns)")
            else:
                print(f"  Process {info['rank']}: no columns (idle process)")
        
        # Prepare data for scatter
        send_data = []
        send_counts = []
        
        for info in column_info:
            start_col = info['start_col']
            num_cols = info['num_cols']
            
            # Extract columns from original matrix
            if num_cols > 0:
                columns = original_matrix[:, start_col:start_col + num_cols]
                send_data.extend(columns.flatten())
                send_counts.append(n * num_cols)
            else:
                send_counts.append(0)
        
        # Calculate displacements for scatterv
        send_displs = [0]
        for i in range(len(send_counts) - 1):
            send_displs.append(send_displs[-1] + send_counts[i])
        
        send_data = np.array(send_data, dtype=np.float64)
        
    else:
        column_info = None
        send_data = None
        send_counts = None
        send_displs = None
        original_matrix = None
    
    # Broadcast column distribution info
    column_info = comm.bcast(column_info, root=0)
    send_counts = comm.bcast(send_counts, root=0)
    send_displs = comm.bcast(send_displs, root=0)
    
    # Each process receives its columns
    my_info = column_info[rank]
    recv_count = my_info['total_elements']
    
    # Scatter data
    recv_data = np.zeros(recv_count, dtype=np.float64)
    comm.Scatterv([send_data, send_counts, send_displs, MPI.DOUBLE], recv_data, root=0)
    
    # Reshape received data to columns
    if recv_count > 0:
        num_cols = my_info['num_cols']
        my_columns = recv_data.reshape(n, num_cols)
        
        end_col = my_info['start_col'] + num_cols - 1
        print(f"Process {rank}: received {num_cols} columns (columns {my_info['start_col']}-{end_col})")
        print(f"Process {rank}: columns shape = {my_columns.shape}")
        print(f"Process {rank}: column data =\n{my_columns}")
    else:
        my_columns = np.array([]).reshape(n, 0)
        print(f"Process {rank}: received no columns (idle process)")
    
    # Gather all columns back to root for reconstruction
    comm.Gatherv(recv_data, [send_data, send_counts, send_displs, MPI.DOUBLE], root=0)
    
    # Reconstruct and verify matrix at root
    if rank == 0:
        # Reconstruct matrix
        reconstructed_matrix = np.zeros((n, n), dtype=np.float64)
        data_offset = 0
        
        for info in column_info:
            start_col = info['start_col']
            num_cols = info['num_cols']
            total_elements = info['total_elements']
            
            if total_elements > 0:
                # Extract column data and reshape
                column_data = send_data[data_offset:data_offset + total_elements]
                columns = column_data.reshape(n, num_cols)
                
                # Place columns in reconstructed matrix
                reconstructed_matrix[:, start_col:start_col + num_cols] = columns
                
                data_offset += total_elements
        
        # Verify reconstruction
        matrices_identical = np.array_equal(original_matrix, reconstructed_matrix)
        if matrices_identical:
            print("SUCCESS: Matrix reconstruction PERFECT!")
        else:
            print("ERROR: Matrix reconstruction failed!")
        
        # Display matrices if small enough
        if n <= 10:
            print(f"\nORIGINAL MATRIX ({n}×{n}):")
            with np.printoptions(precision=6, suppress=True):
                print(original_matrix)
            
            print(f"\nRECONSTRUCTED MATRIX ({n}×{n}):")
            with np.printoptions(precision=6, suppress=True):
                print(reconstructed_matrix)
        else:
            print(f"\n(Matrices too large to display - {n}×{n})")

if __name__ == "__main__":
    main()
