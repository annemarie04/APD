"""
# TODO: Add description of parameter t
Algorithm Serial:
    Input: matricea de latenta L; coordonatele initiale x
    Output: coordonatele mai precise in x

    compute_coordinates(L, x):
    while (error(L, x) > tolerance):
        foreach i:
            F = 0
            foreach j:
                // Calculeaza eroare/forta arcului (1)
                e = L[i,j] - ||x[i] - x[j]||
                // Adauga vectorul de forta al acestui arc la forta totala (2)
                F = F + e * u(x[i] - x[j])
            // Muta un pas mic in directia fortei (3)
            x[i] = x[i] + t * F
"""

import numpy as np
import matplotlib.pyplot as plt


def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm


def compute_error(L, x):
    n = len(x)
    total_error = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
                predicted = np.linalg.norm(x[i] - x[j])
                actual = L[i, j]

                # TODO: This should be squared error?
                error = abs(predicted - actual) / actual
                total_error += error
                count += 1
    
    return total_error / count if count > 0 else 0.0


def compute_coordinates(L, x, tolerance=0.05, t=0.1):
    n = len(x)
    errors = []

    current_error = compute_error(L, x)
    errors.append(current_error)

    iteration = 0
    while current_error > tolerance:
        if iteration % 10 == 0:
            print(f"{iteration:<12} {current_error:<15.6f}")
        
        
        # Parcugerea nodurilor
        for i in range(n):
            F = np.zeros(x.shape[1])  # Forta totala asupra nodului i
            
            # Forta de la toate celelalte noduri
            for j in range(n):
                # TODO: assert(L[i, j] >= 0)
                if i != j and L[i, j] > 0:
                    # Calculeaza eroare/forta arcului (1)
                    predicted_distance = np.linalg.norm(x[i] - x[j])
                    e = L[i, j] - predicted_distance

                    # (2) Adauga vectorul de forta al acestui arc la forta totala
                    # F = F + e * u(x[i] - x[j])
                    direction = unit_vector(x[i] - x[j])
                    F = F + e * direction

            # (3) Muta un pas mic in directia fortei
            # x[i] = x[i] + t * F
            x[i] = x[i] + t * F
        
        iteration += 1
    
        # Calculeaza eroarea curenta
        current_error = compute_error(L, x)
        errors.append(current_error)
    
    return x, errors


def generate_latency_matrix(n, dimensions=2, noise_level=0.05):
    # Generarea coordonatelor
    true_positions = np.random.uniform(-10, 10, (n, dimensions))
    
    # Calcularea matricei bazata pe pozitiile reale
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(true_positions[i] - true_positions[j])
                # Adauga zgomot
                noise = np.random.normal(0, distance * noise_level)
                L[i, j] = max(distance + noise, 0.01)  # Pastreaza valorile > 0
    
    return L, true_positions



def main():
    np.random.seed(42)
    
    # Configuration
    n_nodes = 20
    dimensions = 2
    tolerance = 0.05
    timestep = 0.1
    
    # Genereaza matricea de latenta si coordonatele reale
    L, true_positions = generate_latency_matrix(n_nodes, dimensions, noise_level=0.05)
    print(f"Matricea de latenta: {L.shape}")
    print(f"Coordonatele reale: {true_positions.shape}")
    print()
    
    # Initializeaza coordonatele aleator
    x_initial = np.random.uniform(-1, 1, (n_nodes, dimensions))
    x = x_initial.copy()
    print(f"Coordonatele initiale: {x.shape}")
    
    # Algoritm Vivaldi Serial
    x_final, errors = compute_coordinates(L, x, tolerance=tolerance, t=timestep)
    
    # Compara rezultatele
    print("Sample Predictions:")
    print(f"{'Pair':<12} {'Actual RTT':<15} {'Predicted':<15} {'Error %':<10}")
    print("-" * 55)
    sample_pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (5, 6)]
    for i, j in sample_pairs:
        if j < n_nodes:
            actual = L[i, j]
            predicted = np.linalg.norm(x_final[i] - x_final[j])
            error_pct = abs(predicted - actual) / actual * 100
            print(f"({i:2d}, {j:2d})    {actual:<15.4f} {predicted:<15.4f} {error_pct:<10.2f}")
    
    print()



if __name__ == "__main__":
    main()
