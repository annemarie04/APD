"""
Matrix Line Distribution and Reconstruction
-------------------------------------------
Usage: mpiexec -n <desired_number_of_processes> python matrix_lines.py <matrix_size>
Example: mpiexec -n 4 --oversubscribe python matrix_lines.py 8

This script distributes a matrix by rows (lines) across MPI processes.
Each process receives a consecutive set of complete rows.
"""

from mpi4py import MPI
import numpy as np
import sys

def parse_arguments():
    if len(sys.argv) != 2:
        print("Usage: mpiexec -n <desired_number_of_processes> python matrix_lines.py <matrix_size>")
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

def calculate_line_distribution(n, p):
    """
    Calculate how to distribute n matrix rows among p processes.
    Returns information about which rows each process should handle.
    """
    line_info = []
    
    # Calculate base number of rows per process and remainder
    base_rows = n // p
    extra_rows = n % p
    
    start_row = 0
    for rank in range(p):
        # Processes with rank < extra_rows get one extra row
        num_rows = base_rows + (1 if rank < extra_rows else 0)
        
        line_info.append({
            'rank': rank,
            'start_row': start_row,
            'num_rows': num_rows,
            'total_elements': num_rows * n  # num_rows * n_cols
        })
        
        start_row += num_rows
    
    return line_info

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
    
    # Calculate line distribution
    if rank == 0:
        # Generate the original matrix with random elements
        original_matrix = np.random.rand(n, n)
        print(f"Generated {n}×{n} random matrix")
        
        # Calculate line distribution information for all processes
        line_info = calculate_line_distribution(n, p)
        
        print(f"Line distribution:")
        for info in line_info:
            if info['num_rows'] > 0:
                end_row = info['start_row'] + info['num_rows'] - 1
                print(f"  Process {info['rank']}: rows {info['start_row']}-{end_row} ({info['num_rows']} rows)")
            else:
                print(f"  Process {info['rank']}: no rows (idle process)")
        
        # Prepare data for scatter
        send_data = []
        send_counts = []
        
        for info in line_info:
            start_row = info['start_row']
            num_rows = info['num_rows']
            
            # Extract rows from original matrix
            if num_rows > 0:
                rows = original_matrix[start_row:start_row + num_rows, :]
                send_data.extend(rows.flatten())
                send_counts.append(num_rows * n)
            else:
                send_counts.append(0)
        
        # Calculate displacements for scatterv
        send_displs = [0]
        for i in range(len(send_counts) - 1):
            send_displs.append(send_displs[-1] + send_counts[i])
        
        send_data = np.array(send_data, dtype=np.float64)
        
    else:
        line_info = None
        send_data = None
        send_counts = None
        send_displs = None
        original_matrix = None
    
    # Broadcast line distribution info
    line_info = comm.bcast(line_info, root=0)
    send_counts = comm.bcast(send_counts, root=0)
    send_displs = comm.bcast(send_displs, root=0)
    
    # Each process receives its rows
    my_info = line_info[rank]
    recv_count = my_info['total_elements']
    
    # Scatter data
    recv_data = np.zeros(recv_count, dtype=np.float64)
    comm.Scatterv([send_data, send_counts, send_displs, MPI.DOUBLE], recv_data, root=0)
    
    # Reshape received data to rows
    if recv_count > 0:
        num_rows = my_info['num_rows']
        my_rows = recv_data.reshape(num_rows, n)
        
        end_row = my_info['start_row'] + num_rows - 1
        print(f"Process {rank}: received {num_rows} rows (rows {my_info['start_row']}-{end_row})")
        print(f"Process {rank}: rows shape = {my_rows.shape}")
        print(f"Process {rank}: row data =\n{my_rows}")
    else:
        my_rows = np.array([]).reshape(0, n)
        print(f"Process {rank}: received no rows (idle process)")
    
    # Gather all rows back to root for reconstruction
    comm.Gatherv(recv_data, [send_data, send_counts, send_displs, MPI.DOUBLE], root=0)
    
    # Reconstruct and verify matrix at root
    if rank == 0:
        # Reconstruct matrix
        reconstructed_matrix = np.zeros((n, n), dtype=np.float64)
        data_offset = 0
        
        for info in line_info:
            start_row = info['start_row']
            num_rows = info['num_rows']
            total_elements = info['total_elements']
            
            if total_elements > 0:
                # Extract row data and reshape
                row_data = send_data[data_offset:data_offset + total_elements]
                rows = row_data.reshape(num_rows, n)
                
                # Place rows in reconstructed matrix
                reconstructed_matrix[start_row:start_row + num_rows, :] = rows
                
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
