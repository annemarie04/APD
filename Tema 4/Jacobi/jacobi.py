import matplotlib.pyplot as plt
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
            print(f"\nTolerance reached! Converged in {iteration + 1} iterations.")
            return x, iteration + 1
        
    # Max iterations reached, so stop
    print(f"\nReached maximum iterations ({max_iter})")
    
    return x, max_iter

def jacobi_serial(A, b, x0, max_iter=100, tol=1e-6):    
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

    
def main():
    n_size = [100, 1000, 50000]
    gs_error = []
    j_error = []
    gs_times = []
    j_times = []
    gs_iterations = []
    j_iterations = []

    for n in n_size:
        A, b = generate_test_data(n)
        # x0
        x0 = np.zeros(n)
        
        print("Matrix A:")
        print(A)
        print(f"\nVector b:")
        print(b)
        print(f"\nInitial value x0:")
        print(x0)

        # Run Jacobi
        start_time = time.time()
        x, iterations = jacobi_serial(A, b, x0, max_iter=100, tol=1e-6)
        print(f"\nRunning Jacobi for size {n}...")
        end_time = time.time()
        j_times.append(end_time - start_time)
        j_iterations.append(iterations)
        err = np.linalg.norm(x - x0, ord=np.inf)
        j_error.append(err)

        # Run Gauss-Seidel
        start_time = time.time()
        x, iterations = gauss_seidel_serial(A, b, x0, max_iter=100, tol=1e-6)
        print(f"\nRunning Gauss-Seidel for size {n}...")
        end_time = time.time()
        gs_times.append(end_time - start_time)
        gs_iterations.append(iterations)
        err = np.linalg.norm(x - x0, ord=np.inf)
        gs_error.append(err)

        # Numpy solution for comparison
        x_exact = np.linalg.solve(A, b)
        print(f"Numpy solution:")
        print(x_exact)

    plt.figure(figsize=(10, 6))
    plt.plot(n_size, gs_error, marker='o')
    plt.plot(n_size, j_error, marker='o')
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Error")
    plt.title("Gauss-Seidel vs Jacobi Error")
    plt.legend(["Gauss-Seidel", "Jacobi"])
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(n_size, gs_times, marker='o')
    plt.plot(n_size, j_times, marker='o')
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Execution Time (s)")
    plt.title("Gauss-Seidel vs Jacobi Time")
    plt.legend(["Gauss-Seidel", "Jacobi"])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(n_size, gs_iterations, marker='o')
    plt.plot(n_size, j_iterations, marker='o')
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Iteration Count")
    plt.title("Gauss-Seidel vs Jacobi Iterations")
    plt.legend(["Gauss-Seidel", "Jacobi"])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
