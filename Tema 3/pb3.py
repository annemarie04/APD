from mpi4py import MPI
import numpy as np
import math

def get_hypercube_neighbors(rank, dimension):
    # Getting the neighbors of a process in the hypercube topology
    neighbors = []
    for i in range(dimension):
        neighbor = rank ^ (1 << i)  # Change a bit to find a neighbor
        neighbors.append(neighbor)
    return neighbors

def diffusion_step(comm, rank, data, neighbor_rank, step):
    # Exchange data with a specific neighbor
    print(f"Process {rank}, Step {step}: Exchanging with neighbor {neighbor_rank}")
    send_req = comm.Isend(data, dest=neighbor_rank, tag=step)
    neighbor_data = np.zeros_like(data)
    recv_req = comm.Irecv(neighbor_data, source=neighbor_rank, tag=step)
    send_req.Wait()
    recv_req.Wait()
    
    new_data = np.append(data, neighbor_data)
    
    print(f"Process {rank}: Original data: {data}")
    print(f"Process {rank}: Received from {neighbor_rank}: {neighbor_data}")
    print(f"Process {rank}: New data after diffusion: {new_data}")
    
    return new_data

def hypercube_diffusion(comm, rank, size, initial_data):
    # Check if number of processes is power of 2 = hypercube topology
    if size & (size - 1) != 0:
        if rank == 0:
            print(f"Error: Number of processes ({size}) must be a power of 2 for hypercube topology")
        return None
    
    dimension = int(math.log2(size))
    print(f"Process {rank}: Hypercube dimension = {dimension}")

    neighbors = get_hypercube_neighbors(rank, dimension)
    current_data = initial_data.copy()
    
    for step in range(dimension):
        neighbor = neighbors[step]
        current_data = diffusion_step(comm, rank, current_data, neighbor, step)
        
        # Synchronize all processes after each step
        comm.Barrier()
    
    return current_data

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # initial data per node
    initial_data = np.array([rank], dtype=np.float64)
    print(f"Process {rank}: Initial data = {initial_data}")
    
    # Synchronize before diffusion
    comm.Barrier()
    
    # hypercube diffusion
    final_data = np.sort(hypercube_diffusion(comm, rank, size, initial_data))
    print(f"\nProcess {rank}: Final data after diffusion = {final_data}")

if __name__ == "__main__":
    main()
        
