"""
Parallel MPI implementation of the Vivaldi Network Coordinate System Algorithm

Each MPI process represents a node that communicates with all other nodes.
Uses MPI collective operations for efficient information exchange.

Algorithm:
1. Each node maintains a position coordinate in d-dimensional space
2. Nodes use MPI to exchange coordinates and error estimates
3. Each node adjusts its position to minimize prediction error using spring forces
4. The force is proportional to the difference between measured and predicted latency
"""

import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpi4py import MPI


class VivaldiNode:
    def __init__(self, node_id: int, dimensions: int = 2):
        self.node_id = node_id
        self.dimensions = dimensions
        # Initialize with small random perturbation to break symmetry
        np.random.seed(42 + node_id)
        self.position = np.random.uniform(-0.5, 0.5, dimensions)
        np.random.seed(None)
        # Local error estimate (starts high, decreases as system stabilizes)
        self.error = 1.0
        self.send_to = []  # List of node IDs to send updates to
        self.receive_from = []  # List of node IDs to receive updates from
        
    def distance_to_position(self, other_position):
        return np.linalg.norm(self.position - other_position)
    
    def update_position_vivaldi(self, other_position, other_error, 
                       measured_rtt: float, ce: float = 0.25, cc: float = 0.25):
        # Calculate predicted RTT based on current positions
        predicted_distance = self.distance_to_position(other_position)
        
        # (1) Sample weight balances local and remote error
        w = self.error / (self.error + other_error)
        
        # (2) Compute relative error of this sample
        if measured_rtt > 0:
            es = abs(predicted_distance - measured_rtt) / measured_rtt
        else:
            es = abs(predicted_distance - measured_rtt)
        
        # (3) Update weighted moving average of local error
        self.error = es * ce * w + self.error * (1 - ce * w)
    
        
        # Calculate the direction to move (unit vector u(xi - xj))
        direction = (self.position - other_position) / predicted_distance

        # (4) Update local coordinates
        # δ = cc × w
        delta = cc * w
        # xi = xi + δ × (rtt - ||xi - xj||) × u(xi - xj)
        old_position = self.position.copy()
        self.position = self.position + delta * (measured_rtt - predicted_distance) * direction
        if self.node_id == 5:
            print(f"Node {self.node_id} updated from {old_position} to {self.position} using measured RTT {measured_rtt} and predicted {predicted_distance}")
        
    def __repr__(self):
        return f"Node {self.node_id}: pos={self.position}, error={self.error:.4f}"
    
def generate_neighbors(self, rank, comm, size, num_close, num_far, rtt_matrix):
          # Build the receive_from list based on closest and furthest nodes by RTT
        all_other_ranks = [i for i in range(size) if i != rank]
    
        # Sort other nodes by RTT distance from this node
        sorted_by_rtt = sorted(all_other_ranks, key=lambda x: rtt_matrix[rank][x])
    
        # Select closest and furthest nodes
        actual_num_close = min(num_close, len(sorted_by_rtt))
        actual_num_far = min(num_far, len(sorted_by_rtt))
    
        close_nodes = sorted_by_rtt[:actual_num_close]
        far_nodes = sorted_by_rtt[-(actual_num_far):] if actual_num_far > 0 else sorted_by_rtt
    
        # Combine into receive_from list (avoiding duplicates if overlap)
        receive_from = list(set(close_nodes + far_nodes))
    
        # Exchange receive_from lists so each node knows who wants to receive from them
        # This determines the send_to list for each node
        all_receive_lists = comm.allgather(receive_from)
    
        # Build send_to list: I send to nodes that have me in their receive_from list
        send_to = []
        for other_rank in range(size):
            if other_rank != rank and rank in all_receive_lists[other_rank]:
                send_to.append(other_rank)
        return send_to, receive_from

def generate_rtt_matrix(num_nodes: int, dimensions: int = 2, seed: int = 42):
    np.random.seed(seed)
    
    # Generate true positions for nodes (ground truth)
    true_positions = np.random.uniform(-10, 10, (num_nodes, dimensions))
    
    # Calculate actual RTT matrix based on true positions
    rtt_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # True RTT is Euclidean distance + small random noise
                rtt_matrix[i][j] = np.linalg.norm(true_positions[i] - true_positions[j])
                # Add small measurement noise (5% of distance)
                noise = np.random.normal(0, rtt_matrix[i][j] * 0.05)
                rtt_matrix[i][j] = max(rtt_matrix[i][j] + noise, 0.01)

    return rtt_matrix, true_positions

def calculate_total_error(size, all_positions, rtt_matrix):
                    # Calculate average prediction error
                total_error = 0.0
                count = 0
                for i in range(size):
                    for j in range(i + 1, size):
                        actual_rtt = rtt_matrix[i][j]
                        predicted_rtt = np.linalg.norm(all_positions[i] - all_positions[j])
                        relative_error = (predicted_rtt - actual_rtt)**2
                        total_error += relative_error
                        count += 1
                return total_error, count

def run_simulation(max_rounds: int = 1000, convergence_threshold: float = 0.05,
                    dimensions: int = 2, num_close: int = 3, num_far: int = 2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Generare pozitii reale si matrice RTT
    rtt_matrix, true_positions = generate_rtt_matrix(size, dimensions, seed=42)
    
    # Initializare nod
    node = VivaldiNode(rank, dimensions)
    
    # Pozitia initiala a nodului
    initial_position = node.position.copy()

    # Alege cui trimite si de la cine primeste
    node.send_to, node.receive_from = generate_neighbors(node, rank, comm, size, num_close, num_far, rtt_matrix)

    if rank == 0:
        print(f"Starting parallel Vivaldi with {size} nodes, {dimensions}D coordinates")
        print(f"Algorithm parameters: ce=0.25, cc=0.25")
        print(f"Node selection: {num_close} closest + {num_far} furthest by RTT")
        print(f"Communication: each node receives from {len(node.receive_from)} nodes, sends to ~{len(node.send_to)} nodes")
        print("-" * 30)
    
    # Run simulation rounds
    for round_num in range(max_rounds):

        # Trimitere mesaje
        send_requests = []
        for dest_rank in node.send_to:
            my_info = {
            'position': node.position.copy(),
            'error': node.error,
            'rank': rank,
            'rtt': rtt_matrix[rank][dest_rank]
        }
            req = comm.isend(my_info, dest=dest_rank, tag=round_num)
            send_requests.append(req)
        
        # Primire mesaje
        received_infos = {}
        for src_rank in node.receive_from:
            received_infos[src_rank] = comm.recv(source=src_rank, tag=round_num)
        
        # Asteapta finalizarea trimiterilor
        MPI.Request.Waitall(send_requests)
        
        # Updateaza pozitiiile pe baza datelor primite
        for src_rank in node.receive_from:
            contact_info = received_infos[src_rank]
            
            # vivaldi update pentru fiecare mesaj primit
            node.update_position_vivaldi(contact_info['position'], contact_info['error'], contact_info['rtt'])

        # Synchronizare
        comm.Barrier()
        
        # Print periodic
        if round_num % 100 == 0:
            # Fiecare nod trimite pozitia catre rank 0 prin comunicare punct-la-punct
            if rank == 0:
                all_positions = [None] * size
                all_positions[0] = node.position.copy()

                # Primeste pozitiile nodurilor
                for src in range(1, size):
                    all_positions[src] = comm.recv(source=src, tag=round_num + 100000)
                
                # Calculeaza eroare de predictie
                total_error, count = calculate_total_error(size, all_positions, rtt_matrix)
                print(f"Round: {round_num:<8} Total Error:{total_error:<15.6f}")
                
                # Verifica convergenta
                if total_error < convergence_threshold:
                    print(f"\nConverged after {round_num + 1} rounds!")
                    converged = True
                else:
                    converged = False
            else:
                # Trimite pozitia catre rank 0
                comm.send(node.position.copy(), dest=0, tag=round_num + 100000)
                converged = False
            
            # Am ajuns la precizia dorita
            converged = comm.bcast(converged, root=0)
            if converged:
                break
    
    # Final gathering of all positions and errors via point-to-point communication
    if rank == 0:
        # Rank 0 collects all data
        final_positions = [None] * size
        initial_positions = [None] * size
        final_errors = [None] * size
        
        final_positions[0] = node.position.copy()
        initial_positions[0] = initial_position.copy()
        final_errors[0] = node.error
        
        # Receive from all other ranks
        for src in range(1, size):
            data = comm.recv(source=src, tag=999999)
            final_positions[src] = data['final_position']
            initial_positions[src] = data['initial_position']
            final_errors[src] = data['error']
    else:
        # Other ranks send their data to rank 0
        data = {
            'final_position': node.position.copy(),
            'initial_position': initial_position.copy(),
            'error': node.error
        }
        comm.send(data, dest=0, tag=999999)
    
    # Rank 0 handles output and visualization
    if rank == 0:
        print("=" * 70)
        print(f"  Total rounds: {round_num + 1}")
        
        # Calculeaza eroarea totala de predictie
        total_error, count = calculate_total_error(size, final_positions, rtt_matrix)
        print(f"  Final prediction error: {total_error:.6f}")
        
        # Plot 
        if dimensions == 2:
            plot_results(initial_positions, final_positions, true_positions, size)

def plot_results(initial_positions, final_positions, true_positions, num_nodes):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    initial_pos = np.array(initial_positions)
    final_pos = np.array(final_positions)
    
    # Pozitii initiale
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], c='blue', s=100, 
                alpha=0.6, edgecolors='black')
    for i, pos in enumerate(initial_pos):
        ax1.annotate(str(i), (pos[0], pos[1]), fontsize=8, ha='center')
    ax1.set_xlabel('X coordinate', fontsize=12)
    ax1.set_ylabel('Y coordinate', fontsize=12)
    ax1.set_title('Initial Node Positions', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Pozitii finale
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], c='red', s=100, 
                alpha=0.6, edgecolors='black')
    for i, pos in enumerate(final_pos):
        ax2.annotate(str(i), (pos[0], pos[1]), fontsize=8, ha='center')
    ax2.set_xlabel('X coordinate', fontsize=12)
    ax2.set_ylabel('Y coordinate', fontsize=12)
    ax2.set_title('Final Computed Positions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Pozitii reale
    ax3.scatter(true_positions[:, 0], true_positions[:, 1], c='green', s=100, 
                alpha=0.6, edgecolors='black')
    for i, pos in enumerate(true_positions):
        ax3.annotate(str(i), (pos[0], pos[1]), fontsize=8, ha='center')
    ax3.set_xlabel('X coordinate', fontsize=12)
    ax3.set_ylabel('Y coordinate', fontsize=12)
    ax3.set_title('True Network Positions', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vivaldi_positions_mpi.png', dpi=300, bbox_inches='tight')
    print(f"\nPosition plot saved to: vivaldi_positions_mpi.png")
    plt.show()


def main():
    # Parametrii Simulare
    max_rounds = 1000
    convergence_threshold = 20
    dimensions = 2
    num_close = 20
    num_far = 20

    # Rularea Simularii
    run_simulation(max_rounds, convergence_threshold, dimensions, num_close, num_far)


if __name__ == "__main__":
    main()
