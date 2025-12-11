Env setup:
source "/Users/anne/Desktop/FMI/FMI - V/APD/APD/.venv/bin/activate"
-------------------------------------------------------------------
1. Matrix x matrix multiplication on ring:
mpiexec -n 4 --oversubscribe python3 pb1.py 

2. Point-to-point communication on hypercube
mpiexec -n 4 --oversubscribe python3 pb2.py

3. Diffusion on hypercube
mpiexec -n 4 --oversubscribe python3 pb3.py