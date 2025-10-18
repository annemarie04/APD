#!/bin/bash

# Define your list of files
files=("p2pcomm_0.py" "p2pcomm_1.py" "p2pcomm_2.py" "collective_comm_0.py" "collective_comm_1.py" "collective_comm_2.py" "collective_comm_3.py" "collective_comm_4.py" "collective_comm_5.py" "collective_comm_6.py")

echo "=== Running with 20 processes ==="

# Loop over them

for f in "${files[@]}"; do
    echo "Processing $f ..."
    mpiexec -n 20 --oversubscribe  "$f"
    status=$?
    
    # check if command failed (nonzero exit code)
    if [ $status -ne 0 ]; then
        echo "❌ Failed: $f (exit code $status)"
        failed_files+=("$f")
    else
        echo "✅ Success: $f"
    fi
done


echo "-----------------------------------"
echo "All files processed."

# print failed files if any
if [ ${#failed_files[@]} -gt 0 ]; then
    echo "❗The following files failed:"
    for f in "${failed_files[@]}"; do
        echo "   - $f"
    done
    exit 1  # indicate that not all succeeded
else
    echo "🎉 All files ran successfully!"
fi
