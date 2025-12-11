Env setup:
source "/Users/anne/Desktop/FMI/FMI - V/APD/APD/.venv/bin/activate"
-------------------------------------------------------------------
1. Barrier
mpiexec -n 8 --oversubscribe python3 barrier.py

2. Broadcast
mpiexec -n 8 --oversubscribe python3 broadcast.py
mpiexec -n 8 --oversubscribe python3 broadcast_vector.py

3. Scatter
mpiexec -n 8 --oversubscribe python3 scatter.py

4. Matrix x vector multiplication on ring
mpiexec -n 8 --oversubscribe python3 pb4.py