- All code samples from the Peer-2-peer Communication and Collective Communication chapters of the tutorial are separated in different python script files. 

- Some logs have been added in order to better visualize the sending and receiving of messages between processes. 

- All the scripts can be run at one using the runner.sh script. 

- To avoid verbosity, the script is run for a specific number of nodes for all files at once. For testing with other node numbers, the user needs to modify the specified number in the script. 

- Scripts can be run separately using the command: 
mpiexec -n <node_number> --oversubscribe <filename>