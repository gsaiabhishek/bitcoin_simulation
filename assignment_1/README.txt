code is written in python.

numpy, os, heapq libraries of python are used.
networkx and pydot need to be installed(using pip install networkx, pip install pydot).

Use 'python3 main.py <some arguments written in next line>' to start simulation.
The script takes the following positional arguments in order: n,z,Ttx(in ms ie., milli seconds), max_num_events_allowed,Tk(in ms), max_time(in ms).
Example: python3 main.py --n 11 --z 0.3 --Ttx 2000 --max_events 2000 --Tk 10000 --max_time 120000.
Tk here in example is the 'interarrivaltime' in the moodle post 'Formulae to calculate T_k'.
When either max_events or max_time is reached, the simulation is terminated.

Tree info files will be stored in 'tree_files' folder for each node (please make sure the folder exists).
If parent_block_id and block_id is same - it implies that the node is root.
The block num is the depth of that node from root measured in terms of number of nodes.

Blockchain as a tree will be stored in 'trees' folder (please make sure the folder exists).
The value in images in 'trees' folder represent block ID. Block ID may not be continuous in an image since I am using a global counter for block ID.

Tk_mean in code don't actually mean 'meanTk' term used in moodle post. It is the final mean of exponential distribution of each node.
Number of peers(n) should be > 10.

There are 5 events with event IDs 1 to 5. Following explains them:
1 - create txn and send (to neighbouring peers)
2 - receive txn and forward (to neibhouring peers(except the one we receive txn from))
3 - block creation
4 - block broadcast (to neighbouring peers)
5 - block receive and forwarding (to neighbouring peers(except the one we receive block from))

