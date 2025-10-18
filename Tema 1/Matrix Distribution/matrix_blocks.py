"""
Matrix Block Distribution and Reconstruction
---------------------------------------------
Usage: mpiexec -n <desired_number_of_processes> python matrix_blocks.py <matrix_size>
Example: mpiexec -n 4 --oversubscribe python matrix_blocks.py 8
"""

from mpi4py import MPI
import numpy as np
import sys
import math

def parse_arguments():
    if len(sys.argv) != 2:
        print("Usage: mpiexec -n <desired_number_of_processes> python matrix_blocks.py <matrix_size>")
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

def find_optimal_grid(n, p):
    # Find the optimal grid layout (rows x cols) for p processes
    best_ratio = float('inf')
    best_grid = (1, p)
    
    # Try all possible factorizations of p
    for rows in range(1, int(math.sqrt(p)) + 1):
        if p % rows == 0:
            cols = p // rows
            
            # Check if we can distribute matrix blocks reasonably
            block_rows = n // rows
            block_cols = n // cols
            
            # We're looking for more square-like grids and making sure the blocks aren't too small
            # We calculate the ratio of the grid dimensions to choose the best one
            if block_rows > 0 and block_cols > 0:
                ratio = max(rows, cols) / min(rows, cols)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_grid = (rows, cols)
    
    return best_grid

def calculate_block_info(n, grid_rows, grid_cols, p):
    # Calculate block sizes and positions for each process
    block_info = []
    
    base_block_rows = n // grid_rows
    extra_rows = n % grid_rows
    
    base_block_cols = n // grid_cols
    extra_cols = n % grid_cols
    
    for rank in range(p):
        # Calculate grid position
        grid_row = rank // grid_cols
        grid_col = rank % grid_cols
        
        # Calculate block size for this process
        block_rows = base_block_rows + (1 if grid_row < extra_rows else 0)
        block_cols = base_block_cols + (1 if grid_col < extra_cols else 0)
        
        # Calculate starting position
        start_row = grid_row * base_block_rows + min(grid_row, extra_rows)
        start_col = grid_col * base_block_cols + min(grid_col, extra_cols)
        
        block_info.append({
            'rank': rank,
            'grid_pos': (grid_row, grid_col),
            'start_pos': (start_row, start_col),
            'block_size': (block_rows, block_cols),
            'total_elements': block_rows * block_cols
        })
    
    return block_info

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
    
    # Find optimal grid layout
    if rank == 0:
        # Getting the grid layout
        grid_rows, grid_cols = find_optimal_grid(n, p)
        print(f"Grid layout: {grid_rows}×{grid_cols}")
        
        # Generate the original matrix with random elements
        original_matrix = np.random.rand(n, n)
        print(f"Generated {n}×{n} random matrix")
        
        # Calculate block information for all processes
        block_info = calculate_block_info(n, grid_rows, grid_cols, p)
        
        # Prepare data for scatter
        send_data = []
        send_counts = []
        
        for info in block_info:
            start_row, start_col = info['start_pos']
            block_rows, block_cols = info['block_size']
            
            # Extract block from original matrix
            if block_rows > 0 and block_cols > 0:
                block = original_matrix[start_row:start_row+block_rows, start_col:start_col+block_cols]
                send_data.extend(block.flatten())
                send_counts.append(block_rows * block_cols)
            else:
                send_counts.append(0)
        
        # Calculate displacements for scatterv
        send_displs = [0]
        for i in range(len(send_counts) - 1):
            send_displs.append(send_displs[-1] + send_counts[i])
        
        send_data = np.array(send_data, dtype=np.float64)
        
    else:
        grid_rows = grid_cols = None
        block_info = None
        send_data = None
        send_counts = None
        send_displs = None
        original_matrix = None
    
    # Broadcast grid dimensions and block info
    grid_rows = comm.bcast(grid_rows, root=0)
    grid_cols = comm.bcast(grid_cols, root=0)
    block_info = comm.bcast(block_info, root=0)
    send_counts = comm.bcast(send_counts, root=0)
    send_displs = comm.bcast(send_displs, root=0)
    
    # Each process receives its block
    my_info = block_info[rank]
    recv_count = my_info['total_elements']
    
    # Scatter data
    recv_data = np.zeros(recv_count, dtype=np.float64)
    comm.Scatterv([send_data, send_counts, send_displs, MPI.DOUBLE], recv_data, root=0)
    
    # Reshape received data to a 2D block
    if recv_count > 0:
        block_rows, block_cols = my_info['block_size']
        my_block = recv_data.reshape(block_rows, block_cols)
        
        print(f"Process {rank}: received block of size {my_block.shape} at grid position {my_info['grid_pos']}")
        print(f"Process {rank}: block data =\n{my_block}")
    else:
        my_block = np.array([[]])
        print(f"Process {rank}: received empty block (idle process)")
    
    # Gather all blocks back to root for reconstruction
    comm.Gatherv(recv_data, [send_data, send_counts, send_displs, MPI.DOUBLE], root=0)
    
    # Reconstruct and verify matrix at root
    if rank == 0:
        # Reconstruct matrix
        reconstructed_matrix = np.zeros((n, n), dtype=np.float64)
        data_offset = 0
        
        for info in block_info:
            start_row, start_col = info['start_pos']
            block_rows, block_cols = info['block_size']
            total_elements = info['total_elements']
            
            if total_elements > 0:
                # Extract block data and reshape
                block_data = send_data[data_offset:data_offset + total_elements]
                block = block_data.reshape(block_rows, block_cols)
                
                # Place block in reconstructed matrix
                reconstructed_matrix[start_row:start_row+block_rows, 
                                   start_col:start_col+block_cols] = block
                
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
