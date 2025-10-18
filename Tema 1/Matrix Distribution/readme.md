- The third problem is split into 3 different scripts: 
    matrix_blocks.py - distributes a matrix to processes by blocks
    matrix_lines.py - distributes a matrix to processes by lines
    matrix_columns.py - distributes a matrix to processes by columns

- The size of the matrix and the number of processes can be configured directly when running the command for each script:
    mpiexec -n <nodes_number> --oversubscribe python <python_script> <matrix_size>

- Distributing by lines works by attempting to distribute and even number of lines to all processes. When the number of lines is greater than the number of processes, some processes will receive more than 1 line each. 

- Distributing by collumns attempt to distribute columns as evenly as possible to all processes. When the number of columns is greater than the number of processes, some processes will receive more than 1 column each. 

- Distributing by block attempt to find distribute one block to each process. In some cases, the matrix can't be split into equally sized blocks. The script works by finding a grid that splits the matrix into exactly p blocks, trying to keep the blocks as close to a square as possible. Afterwards, the block sized are determined by splitting the matrix and adding leftout columns and rows to some blocks. This means that some blocks will have more columns/rows than other. These blocks are then sent to the p nodes. 

- At the end of each script, the root process attempts to rebuild the matrix by gathering the lines/columns/blocks of each process. 

- If the input matrix has a size of less than 12, both the original and rebuild matrices will be printed at the end. This threshold is enable running with larger matrix sizes and avoid verbosity in printed results.