from mpi4py import MPI
import numpy as np

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


def main():
        n = 8
        # A
        A = np.array([
            [10, -1,  2,  0,  0,  0,  0,  0],
            [-1, 11, -1,  3,  0,  0,  0,  0],
            [ 2, -1, 10, -1,  0,  0,  0,  0],
            [ 0,  3, -1,  8,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 12, -1,  0,  0],
            [ 0,  0,  0,  0, -1, 10,  2,  0],
            [ 0,  0,  0,  0,  0,  2,  9, -1],
            [ 0,  0,  0,  0,  0,  0, -1,  7]
        ], dtype=float)
        
        # b
        b = np.array([6, 25, -11, 15, 12, 9, 8, 6], dtype=float)
        
        # x0
        x0 = np.zeros(n)
        
        print("Matrix A:")
        print(A)
        print(f"\nVector b:")
        print(b)
        print(f"\nInitial value x0:")
        print(x0)


        # Solve using serial Gauss-Seidel
        x, iterations = gauss_seidel_serial(A, b, x0, max_iter=100, tol=1e-6)

        print(f"\nSolution x:")
        print(x)

        # Numpy solution
        x_exact = np.linalg.solve(A, b)
        print(f"Numpy solution:")
        print(x_exact)


if __name__ == "__main__":
    main()
