import numpy as np
import time

def generate_test_data(n):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] += sum(np.abs(A[i]))
    b = np.random.rand(n) * 10
    return A, b


def gauss_seidel_serial(A, b, x0, max_iter=100, tol=1e-6):
    n = len(b)
    x = x0.copy()
    print(f"Tolerance: {tol}, Max iterations: {max_iter}")
    
    start_time = time.time()
    # Do the iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Each process works on its own rows and shares updates
        for i in range(n):
            # Calculate x[i] using the formula
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]           
            x[i] = (b[i] - sigma) / A[i, i]
        
        # Check difference for convergence 
        diff = np.linalg.norm(x - x_old, ord=np.inf)
    
        print(f"Iteration {iteration + 1}: max diff = {diff:.2e}")
        
        # Check if tolerance reached to stop
        if diff < tol:
            end_time = time.time()
            print(f"\nTolerance reached! Converged in {iteration + 1} iterations.")
            return x, iteration + 1, end_time - start_time

    # Max iterations reached, so stop
    end_time = time.time()
    print(f"\nReached maximum iterations ({max_iter})")
    return x, max_iter, end_time - start_time


def main():
    n = 100
    A, b = generate_test_data(n)
    # x0
    x0 = np.zeros(n)

    # Solve using Serial Gauss-Seidel
    x, iterations, exec_time = gauss_seidel_serial(A, b, x0, max_iter=100, tol=1e-6)
        
    print(f"PERFORMANCE SUMMARY")
    print(f"Iterations: {iterations}")
    print(f"Execution time: {exec_time:.6f} seconds")
    print(f"Time per iteration: {exec_time/iterations:.6f} seconds")


if __name__ == "__main__":
    main()