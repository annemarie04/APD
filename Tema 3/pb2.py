import numpy as np

def get_hypercube_neighbors(rank, dimension):
    # Getting the neighbors of a process in the hypercube topology
    neighbors = []
    for i in range(dimension):
        neighbor = rank ^ (1 << i)  # Change a bit to find a neighbor
        neighbors.append(neighbor)
    return neighbors

def send_packet(comm, rank, packet, id_in_path, path_id, paths):
    print(f"Process {rank}: Sending packet {packet} to next hop {paths[path_id][id_in_path + 1]}")
    send_req = comm.Isend(packet, dest=paths[path_id][id_in_path + 1], tag=path_id)
    send_req.Wait()
    
def receive_packet(comm, rank, id_in_path, path_id, paths):
    # Packet structure: [vector_element, path_pointer, path_id, original_source, final_destination]
    packet = np.zeros(2, dtype=np.float64)
    recv_req = comm.Irecv(packet, source=paths[path_id][id_in_path - 1], tag=path_id)
    recv_req.Wait()
    print(f"Process {rank}: Received packet {packet} from {paths[path_id][id_in_path - 1]}")
    send_packet(comm, rank, packet, id_in_path, path_id, paths)
    return packet

def find_independent_paths(source, destination, dimension):
    paths = []
    
    for path_id in range(dimension):
        path = [source]
        current_node = source
        
        # get intermediate node by flipping a bit
        intermediate_node = current_node ^ (1 << path_id)
        path.append(intermediate_node)
        current_node = intermediate_node
        
        remaining_diff = current_node ^ destination
        
        # flip bits in order 
        # but skip the bit we already flipped
        for bit_pos in range(dimension):
            if bit_pos != path_id and (remaining_diff & (1 << bit_pos)):
                current_node = current_node ^ (1 << bit_pos)
                path.append(current_node)
        
        # destination not reached yet
        # flip back
        if current_node != destination:
            if (current_node ^ destination) & (1 << path_id):
                current_node = current_node ^ (1 << path_id)
                path.append(current_node)
        
        paths.append(path)
    
    return paths



def main():
    # Import MPI here to avoid issues when running without MPI
    from mpi4py import MPI
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Hypercube parameters
    source = 1
    destination = 5
    dimension = 3
    n = 6
    
    # Find the independent paths (all processes need this)
    paths = find_independent_paths(source, destination, dimension)
    
    if rank == source:
        # Vector to send
        vector = np.array([i for i in range(n)], dtype=np.float64)
        print(f"Process {rank}: Sending vector {vector}")
        print(f"Found {len(paths)} independent paths:")
        
        for i, path in enumerate(paths):
            print(f"  Path {i+1}: {' -> '.join(map(str, path))}")
        
        # Split vector among paths and send directly to destination
        elements_per_path = len(vector) // len(paths)
        
        for path_id, path in enumerate(paths):
            start_idx = path_id * elements_per_path
            end_idx = start_idx + elements_per_path
            
            if start_idx < len(vector):
                path_data = vector[start_idx:end_idx]
                
                # Send data with path_id as tag
                print(f"Process {rank}: Sending {path_data} via path {path_id}: {' -> '.join(map(str, path))}")
                path = path[1:]  # Remove source from path for sending
                comm.Send(path_data, dest=path[0], tag=path_id)
    
    elif rank == destination:
        print(f"Dest process {rank}: Receiving data...")
        received_parts = []
        
        # Receive from each path
        for path_id in range(len(paths)):
            # Determine expected size (6 elements / 3 paths = 2 each)
            expected_size = 2
            data = np.zeros(expected_size, dtype=np.float64)
            len_path = len(paths[path_id])
            recv_req = comm.Irecv(data, source=paths[path_id][len_path - 2], tag=path_id)
            recv_req.Wait()
            print(f"Dest process {rank}: Received {data} via path {path_id}")
            received_parts.extend(data)

        print(f"Dest process {rank}: Complete received vector: {np.array(received_parts, dtype=np.float64)}")
    else:
        # Receive from each path
        for path_id in range(len(paths)):
            for i in range(1, len(paths[path_id])):
                # 6 elements / 3 paths = 2 each
                expected_size = 2
                if rank == paths[path_id][i]:
                    print(f"Process {rank}: Receiving data...")
                    data = receive_packet(comm, rank, i, path_id, paths)
    
    # Final synch
    comm.Barrier()

if __name__ == "__main__":
    main()

