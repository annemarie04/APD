/usr/local/bin/mpiexec --oversubscribe -n 8 "/Users/anne/Desktop/FMI/FMI - V/APD/.venv/bin/python" hypc_gauss_seidel.py
Serial Gauss Seidel
_________________________________________________________________________________________________________
n = 8
Iterations: 10
Execution time: 0.000522 seconds
Time per iteration: 0.000052 seconds
_________________________________________________________________________________________________________
n = 100
Iterations: 8
Execution time: 0.031288 seconds
Time per iteration: 0.003911 seconds
____________________________________________________________________________________________________________________________________________________________________________________________________________________


Parallel Gauss Seidel 
_________________________________________________________________________________________________________
No of processors    |   Size(n) |           Ring Gauss Seidel        |     Hypercube Gauss Seidel
_________________________________________________________________________________________________________
        4           |       8   |      Iterations: 11                |      Iterations: 15
                    |           |      Execution time: 0.007882      |      Execution time: 0.006598
                    |           |      Time per iteration: 0.000717  |      Time per iterations: 0.000440
_________________________________________________________________________________________________________
        8           |       8   |      Iterations: 8                 |      Iterations: 18
                    |           |      Execution time: 0.008980      |      Execution time: 0.020637
                    |           |      Time per iteration: 0.001123  |      Time per iterations: 0.001146
_________________________________________________________________________________________________________
        4           |      100  |      Iterations: 34                |      Iterations: 34    
                    |           |      Execution time: 0.060005      |      Execution time: 0.053399
                    |           |      Time per iteration: 0.001765  |      Time per iteration: 0.001571
