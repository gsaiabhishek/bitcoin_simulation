code is written in python.

numpy, os, heapq libraries of python are used.
networkx, pydot, graphviz need to be installed(using pip install networkx, pip install pydot, sudo apt install graphviz).



->Use 'python3 main.py <some arguments written in next line>' to start simulation.

->The script takes the following positional arguments in order: 
n,
z,
eta(fraction of honest miners adversary is connected to. zeta in problem statement),
Ttx(in ms ie., milli seconds), 
max_num_events_allowed,
Tk(in ms), 
max_time(in ms), 
adv_hash(fraction of hasing power to adversary), 
is_selfish(if 0, will do stubborn mining. if 1 will do selfish mining), 
total_adversaries(if 1, we do attack. if 0 we do normal simulation of bitcoin ie, asssignment 1)

Examples:
1. python3 main.py --n 12 --z 0.5 --eta 0.5 --Ttx 5000 --max_events 2000 --Tk 10000 --max_time 18000 --adv_hash 0.3 --is_selfish 1 --total_adversaries 1.
2. python3 main.py --n 50 --Ttx 10000 --Tk 40000 --eta 0.9 --adv_hash 0.35 --max_events 40000 --max_time -1 --is_selfish 1.

->Tk here in example is the 'interarrivaltime' in the moodle post 'Formulae to calculate T_k'.
->When either max_events or max_time is reached, the simulation is terminated. If max_events is given as -1, we consider max_time limit. Similar is the case with max_time being -1.
->if total_adversaries == 0:
	we do normal simulation(assignment 1) irrespective of value of is_selfish
elif tota_adversaries == 1:
	if is_selfish == 1:
		we do selfish mining
	elif is_selfish == 0:
		we do stubborn mining


->Tree info files will be stored in 'tree_files' folder for each node (please make sure the folder exists).
->If parent_block_id and block_id is same - it implies that the node is root.
->The block num is the depth of that node from root measured in terms of number of nodes.

->Blockchain as a tree will be stored in 'trees' folder (please make sure the folder exists).
->The value in images in 'trees' folder represent block ID. Block ID may not be continuous in an image since I am using a global counter for block ID.

->The tree structure of folders for this assignment to store tree_files and visualisations should be as follows:
	a2/
	|
	|--main.py
	|
	|--trees/		}
	|				}- these folders are not needed for attacks
	|--tree_files/	}
	|
	|--selfish/
	|	|
	|	|--trees/
	|	|--tree_files/
	|
	|--stubborn/
		|
		|--trees/
		|--tree_files/



->Tk_mean in code don't actually mean 'meanTk' term used in moodle post. It is the final mean of exponential distribution of each node.
->Number of honest peers(n-total_adversaries) should be > 10.
->There are 5 events with event IDs 1 to 5. Following explains them:
	1 - create txn and send (to neighbouring peers)
	2 - receive txn and forward (to neibhouring peers(except the one we receive txn from))
	3 - block creation
	4 - block broadcast (to neighbouring peers)
	5 - block receive and forwarding (to neighbouring peers(except the one we receive block from))