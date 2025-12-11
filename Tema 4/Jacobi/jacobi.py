
import numpy as np
import time

def generate_test_data(n):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] += sum(np.abs(A[i]))
    b = np.random.rand(n) * 10
    return A, b


def jacobi_parallel(A, b, x0, max_iter=100, tol=1e-6):    
    n = len(b)
    x = x0.copy()
    
    print(f"Starting Jacobi: Solving {n} equations")
    print(f"Tolerance: {tol}, Max iterations: {max_iter}")
    
    # Do the iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Local updates - store in temporary array
        x = np.zeros(n)
        
        # Each process works on its own rows using the old values only
        for i in range(n):
            # Calculate x[i] using the formula with x_old values
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x_old[j]
            
            x[i] = (b[i] - sigma) / A[i, i]
        
        # Check difference for convergence 
        diff = np.linalg.norm(x - x_old, ord=np.inf)

        # Check if tolerance reached to stop
        if diff < tol:
            print(f"\nTolerance reached! Converged in {iteration + 1} iterations.")
        return x, iteration + 1

    print(f"\nReached maximum iterations ({max_iter})")
    return x, max_iter

def run_jacobi(A, b, n):
    # x0
    x0 = np.zeros(n)
        
    print("Matrix A:")
    print(A)
    print(f"\nVector b:")
    print(b)
    print(f"\nInitial value x0:")
    print(x0)
    
    start_time = time.time()
    # Solve using parallel Jacobi
    x, iterations = jacobi_parallel(A, b, x0, max_iter=100, tol=1e-6)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    print(f"\nSolution x:")
    print(x)
    return x
    
def main():
    n_size = [100, 1000, 50000]
    error = []
    for n in n_size:
        A, b = generate_test_data(n)
        print(f"\nRunning Jacobi for size {n}...")
        x = run_jacobi(A, b, n)

        # Numpy solution for comparison
        x_exact = np.linalg.solve(A, b)
        print(f"Numpy solution:")
        print(x_exact)



if __name__ == "__main__":
    main()
