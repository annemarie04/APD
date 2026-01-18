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
    """Represents a node in the Vivaldi coordinate system"""
    
    def __init__(self, node_id: int, dimensions: int = 2):
        """
        Initialize a Vivaldi node
        
        Args:
            node_id: Unique identifier for the node (MPI rank)
            dimensions: Number of dimensions in the coordinate space (default: 2)
        """
        self.node_id = node_id
        self.dimensions = dimensions
        # Initialize with small random perturbation to break symmetry
        np.random.seed(42 + node_id)
        self.position = np.random.uniform(-0.5, 0.5, dimensions)
        np.random.seed(None)
        # Local error estimate (starts high, decreases as system stabilizes)
        self.error = 1.0
        
    def distance_to_position(self, other_position: np.ndarray) -> float:
        """Calculate Euclidean distance to another position"""
        return np.linalg.norm(self.position - other_position)
    
    def update_position(self, other_position: np.ndarray, other_error: float, 
                       measured_rtt: float, ce: float = 0.25, cc: float = 0.25):
        """
        Update position based on measurement from another node
        
        Args:
            other_position: Position coordinates of the other node
            other_error: Error estimate of the other node
            measured_rtt: Measured RTT to the other node
            ce: Weight for local error update (default: 0.25)
            cc: Coordinate update weight (default: 0.25)
        """
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
        
        # Prevent error from getting too small
        self.error = max(self.error, 0.01)
        
        # Calculate the direction to move (unit vector u(xi - xj))
        if predicted_distance > 0.001:
            # Direction from other node to this node
            direction = (self.position - other_position) / predicted_distance
        else:
            # If nodes are at (nearly) same position, establish an initial direction
            direction = np.random.uniform(-1, 1, self.dimensions)
            direction = direction / np.linalg.norm(direction)
        
        # (4) Update local coordinates
        # δ = cc × w
        delta = cc * w
        # xi = xi + δ × (rtt - ||xi - xj||) × u(xi - xj)
        self.position = self.position + delta * (measured_rtt - predicted_distance) * direction
        
    def __repr__(self):
        return f"Node {self.node_id}: pos={self.position}, error={self.error:.4f}"


def generate_rtt_matrix(num_nodes: int, dimensions: int = 2, seed: int = 42):
    """
    Generate the RTT matrix that all nodes know
    
    Args:
        num_nodes: Number of nodes in the network
        dimensions: Number of dimensions for true positions
        seed: Random seed for reproducibility
        
    Returns:
        rtt_matrix: The RTT matrix
        true_positions: The true positions used to generate RTTs
    """
    np.random.seed(seed)
    
    # Generate true positions for nodes (ground truth)
    true_positions = np.random.uniform(-10, 10, (num_nodes, dimensions))
    
    # Calculate actual RTT matrix based on true positions
    rtt_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # True RTT is Euclidean distance + small random noise
                distance = np.linalg.norm(true_positions[i] - true_positions[j])
                # Add small measurement noise (5% of distance)
                noise = np.random.normal(0, distance * 0.05)
                rtt_matrix[i][j] = max(distance + noise, 0.01)
    
    return rtt_matrix, true_positions


def run_vivaldi_mpi(max_rounds: int = 1000, convergence_threshold: float = 0.05,
                    dimensions: int = 2, contacts_per_round: int = None):
    """
    Run Vivaldi algorithm in parallel using MPI with unrestricted communication
    
    Each MPI process represents a node that:
    - Maintains its own coordinates and error estimate
    - Exchanges information with all other nodes via MPI
    - Updates its position based on a random subset of other nodes each round
    
    Args:
        max_rounds: Maximum number of simulation rounds
        convergence_threshold: Stop when average error falls below this
        dimensions: Number of coordinate dimensions
        contacts_per_round: Number of other nodes to contact per round (None = all nodes)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # All nodes generate the same RTT matrix (they all know it)
    rtt_matrix, true_positions = generate_rtt_matrix(size, dimensions, seed=42)
    
    # Each node creates its own VivaldiNode instance
    node = VivaldiNode(rank, dimensions)
    
    # Store initial position for later visualization
    initial_position = node.position.copy()
    
    # Determine how many nodes to contact per round
    if contacts_per_round is None:
        contacts_per_round = size - 1  # Contact all other nodes
    else:
        contacts_per_round = min(contacts_per_round, size - 1)
    
    if rank == 0:
        print(f"Starting parallel Vivaldi with {size} nodes, {dimensions}D coordinates")
        print(f"Algorithm parameters: ce=0.25, cc=0.25")
        print(f"Communication: each node contacts {contacts_per_round} other nodes per round")
        print(f"{'Round':<8} {'Avg Error':<15}")
        print("-" * 30)
    
    # Run simulation rounds
    for round_num in range(max_rounds):
        # Determine which nodes to contact this round
        all_other_ranks = [i for i in range(size) if i != rank]
        
        if contacts_per_round >= len(all_other_ranks):
            # Contact all other nodes
            contact_ranks = all_other_ranks
        else:
            # Randomly select a subset
            random.seed(rank * 10000 + round_num)  # Deterministic for reproducibility
            contact_ranks = random.sample(all_other_ranks, contacts_per_round)
            random.seed(None)
        
        # Use allgather to exchange information with all nodes
        my_info = {
            'position': node.position.copy(),
            'error': node.error,
            'rank': rank
        }
        all_infos = comm.allgather(my_info)
        
        # Update position based on selected contact nodes
        for contact_rank in contact_ranks:
            contact_info = all_infos[contact_rank]
            
            # Get measured RTT to this contact
            measured_rtt = rtt_matrix[rank][contact_rank]
            
            # Update my position based on contact's information
            node.update_position(
                contact_info['position'],
                contact_info['error'],
                measured_rtt
            )
        
        # Calculate and print prediction error periodically
        if round_num % 10 == 0:
            # All ranks participate in gather
            all_positions = comm.gather(node.position, root=0)
            
            if rank == 0:
                # Only rank 0 calculates and prints
                total_error = 0.0
                count = 0
                for i in range(size):
                    for j in range(i + 1, size):
                        actual_rtt = rtt_matrix[i][j]
                        predicted_rtt = np.linalg.norm(all_positions[i] - all_positions[j])
                        relative_error = abs(predicted_rtt - actual_rtt) / actual_rtt
                        total_error += relative_error
                        count += 1
                
                avg_error = total_error / count if count > 0 else 0.0
                print(f"{round_num:<8} {avg_error:<15.6f}")
                
                # Check convergence
                if avg_error < convergence_threshold:
                    print(f"\nConverged after {round_num + 1} rounds!")
                    converged = True
                else:
                    converged = False
            else:
                converged = False
            
            # Broadcast convergence status to all ranks
            converged = comm.bcast(converged, root=0)
            if converged:
                break
    
    # Final gathering of all positions and errors
    final_positions = comm.gather(node.position, root=0)
    initial_positions = comm.gather(initial_position, root=0)
    final_errors = comm.gather(node.error, root=0)
    
    # Rank 0 handles output and visualization
    if rank == 0:
        print("\n" + "=" * 70)
        print("Prediction Accuracy (sample pairs):")
        print("=" * 70)
        
        sample_pairs = [(0, 1), (0, 2), (1, 2)] if size >= 3 else [(0, 1)]
        for i, j in sample_pairs:
            if j < size:
                actual = rtt_matrix[i][j]
                predicted = np.linalg.norm(final_positions[i] - final_positions[j])
                error = abs(predicted - actual) / actual * 100
                print(f"  Node {i} <-> Node {j}: "
                      f"Actual={actual:.4f}, Predicted={predicted:.4f}, "
                      f"Error={error:.2f}%")
        
        print("\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)
        print(f"  Total rounds: {round_num + 1}")
        print(f"  Average node error: {np.mean(final_errors):.6f}")
        
        # Calculate final overall error
        total_error = 0.0
        count = 0
        for i in range(size):
            for j in range(i + 1, size):
                actual_rtt = rtt_matrix[i][j]
                predicted_rtt = np.linalg.norm(final_positions[i] - final_positions[j])
                relative_error = abs(predicted_rtt - actual_rtt) / actual_rtt
                total_error += relative_error
                count += 1
        final_avg_error = total_error / count if count > 0 else 0.0
        print(f"  Final prediction error: {final_avg_error:.6f}")
        
        # Plot results
        if dimensions == 2:
            plot_results(initial_positions, final_positions, true_positions, size)


def plot_results(initial_positions, final_positions, true_positions, num_nodes):
    """Plot initial, final, and true positions"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    initial_pos = np.array(initial_positions)
    final_pos = np.array(final_positions)
    
    # Initial positions
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
    
    # Final positions
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
    
    # True positions
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
    """Main function to run the parallel Vivaldi simulation"""
    # You can adjust contacts_per_round to limit communication:
    # - None or size-1: contact all other nodes each round (unrestricted)
    # - 5-10: contact a random subset for efficiency
    run_vivaldi_mpi(max_rounds=1000, convergence_threshold=0.05, 
                    dimensions=2, contacts_per_round=None)


if __name__ == "__main__":
    main()
