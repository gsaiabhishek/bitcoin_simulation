import numpy as np
import heapq as hq
import os
import networkx as nx
import pydot
import argparse

np.random.seed(50) # todo: check np random seed

def get_topology(n, d, c, k):
#reference for building a P2P network: https://www.researchgate.net/publication/3235764_Building_low-diameter_peer-to-peer_networks
	links = np.zeros([n,n], dtype = bool)
	n_nei = np.zeros(n)

	# initialising k peers so that they can be in cache(refer paper for cache)
	for i in range(k):
		next_d = np.arange(d)
		next_d += (i+1)
		next_d %= k
		links[i, next_d] = True
		links[next_d, i] = True

	n_nei[:k] = 2*d
	cache = np.arange(k)
	replacements = {} # int(key)->list. key replaces the peers(in list) in cache

	for i in range(k, n):
		d_conn = np.random.choice(cache, d, replace = False) #randomly choosing d peers from cache for peer i
		n_nei[d_conn] += 1
		n_nei[i] = d
		links[d_conn, i] = True
		links[i, d_conn] = True

		for j in range(np.size(cache)): # checking if degree of peers in cache exceeds c, if so replace them from cache
			node = cache[j]
			if n_nei[node] <= c:
				continue
			else:
				curr = node
				index_repl = 0
				d_node = None
				while True:
					nei_curr = np.where(links[curr])[0]
					lessthanc_nei_curr_indices = np.where(n_nei[nei_curr] < c)[0]
					lessthanc_nei_curr = nei_curr[lessthanc_nei_curr_indices]
					pot_nei = np.setdiff1d(lessthanc_nei_curr, cache) # find peers which are neighbours to curr but has degree < c
					if np.size(pot_nei) == 0: # if no such peers are found, go to the peer which curr replaced in cache and try again
						if node in replacements:
							if index_repl == len(replacements[node]):
								print("Panic: list of replacements is exhausted for node")
								exit(1)
							curr = replacements[node][index_repl]
							index_repl += 1
						else:
							print("Panic: no previous replacements available for node")
							exit(1)
					else: # potentional peer found to replace cache[j] in cache
						d_node = pot_nei[0]
						break

				if d_node is None:
					print("Panic: d_node is not found - this shouldn't be printed")
					exit(1)
				if node in replacements: # adjust replacements to reflect d_node replaces node in cache
					repl = replacements.pop(node, None)
					repl.insert(0, node)
					replacements[d_node] = repl
				else:
					replacements[d_node] = [node]
				cache[j] = d_node
				links[d_node, node] = True
				links[node, d_node] = True
	return links

class Block:
	def __init__(self, bid, parent_bid, txns = []):
		self.bid = bid # block id
		self.parent_bid = parent_bid
		self.txns = txns
	def add_txn(self, txn):
		self.txns.append(txn)

class Txn:
	def __init__(self, idx, idy, c, tid):
		self.idx = idx # sender
		self.idy = idy # receiver
		self.c = c
		self.tid = tid # txn id

class Peer:
	def __init__(self, is_slow = False):
		self.is_slow = is_slow
		self.all_txns = [] # stores all txns seen so far
		self.all_txns_toa = [] # time of arrival/creation
		self.my_txn_indices = []

		self.current_block = None # that is being created/mined
		self.all_blocks = [] # stores all blocks seen so far
		self.my_block_indices = [] # need not be same size as of all_blocks
		self.all_blocks_toa = [] # time of arrival/creation
		self.is_leaf = [] # same size as of all_blocks
		self.depth = [] # same size as of all_blocks. depth is number of nodes(not edges) till that node from root
		self.path_from_root = {} # same size as of all_blocks. Each index of all_block will have a key here which maps to list of indices which are path from root till this node
		self.root_index = None # will be initialised to 0 when genisis block is created
		self.parent_block_index = [] # same size as of all_blocks
		# special parents: itself means root, -1 means block not in blockchain yet/waiting for parent, -2 means invalid block
		self.block_children_index = {} # map: index in all_blocks -> [index_in_all_blocks]

	def create_genisis_block(self, bid, curr_time):
		genisis_block = Block(bid, bid)
		self.root_index = len(self.all_blocks)
		self.all_blocks.append(genisis_block)
		self.my_block_indices.append(self.root_index)
		self.is_leaf.append(True)
		self.depth.append(1)
		self.path_from_root[self.root_index] = [self.root_index]
		self.all_blocks_toa.append(curr_time)
		self.parent_block_index.append(self.root_index) # genisis is parent to itself, hence it is root of blockchain

	def get_longest_chain(self): # path, length returned will be from leaf to root
		best_leaf_index = -1
		best_leaf_depth = -1
		for i in range(len(self.all_blocks)):
			if (self.parent_block_index[i] < 0) or not (self.is_leaf[i]):
				continue
			if self.depth[i] > best_leaf_depth:
				best_leaf_index, best_leaf_depth = i, self.depth[i]
			elif self.depth[i] == best_leaf_depth:
				if self.all_blocks[i].bid < self.all_blocks[best_leaf_index].bid:
					best_leaf_index, best_leaf_depth = i, self.depth[i]

		temp_path = (self.path_from_root[best_leaf_index]).copy()
		temp_path.reverse()
		return temp_path, self.depth[best_leaf_index]


	def create_new_block(self, path, bid, peer_id, tid): # just create a block from "txns in txn pool but not in longest path"
		
		# finding allowed txn to choose from
		existing_txns = []
		for block_index in path:
			existing_txns.extend(self.all_blocks[block_index].txns)
		allowed_txns_index = []
		for i in range(len(self.all_txns)):
			if not( self.all_txns[i] in existing_txns ):
				allowed_txns_index.append(i)
		
		low_bound = 1
		if len(allowed_txns_index) == 0:
			low_bound = 0 # setting low_bound as 0, if no txns are there to randomly choose from
		no_of_txns = np.random.randint(low_bound, min(len(allowed_txns_index)+1, 1000), 1)[0] # random count of txns choosen
		txns_index = np.random.choice(np.array(allowed_txns_index), no_of_txns, replace = False) # random txns choosen
		
		self.current_block = Block(bid, self.all_blocks[path[0]].bid)
		for i in txns_index:
			self.current_block.add_txn(self.all_txns[i])
		mining_fee_txn = Txn(None, peer_id, 50, tid) # mining fee txn, convention: idx == None => mining fee txn
		self.current_block.add_txn(mining_fee_txn)
		return

	def is_valid_block(self, parent_index, block, n):
		path = self.path_from_root[parent_index] # get path from root till block's parent
		balance = np.zeros(n)

		# find balance from root till block's parent
		for index in path:
			for txn in self.all_blocks[index].txns:
				if not txn.idx == None:
					balance[txn.idx] -= txn.c
				balance[txn.idy] += txn.c

		for txn in block.txns:
			if not txn.idx == None:
				balance[txn.idx] -= txn.c
			balance[txn.idy] += txn.c
		for j in balance:
			if j < 0:
				return False
		return True

	def add_block(self, block, curr_time, n):
		'''
		return value:
		0 - not added since parent is not yet there
		1 - successfully added
		2 - invalid block
		'''
		parent_present = False
		parent_index = None
		for i in range(len(self.all_blocks)):
			if (not parent_present) and (self.all_blocks[i].bid == block.parent_bid) and (not (self.parent_block_index[i] == -1)):
				parent_index = i
				parent_present = True
				break

		if not parent_present: # if parent is not yet arrvied, store block in all_blocks but set its parent index to -1
			self.all_blocks.append(block)
			self.all_blocks_toa.append(curr_time)
			self.parent_block_index.append(-1)
			self.is_leaf.append(False)
			self.depth.append(-1)
			return 0

		is_valid = self.is_valid_block(parent_index, block, n)
		if not is_valid: # if invalid, store block in all_blocks but set its parent index to -2
			self.all_blocks.append(block)
			self.all_blocks_toa.append(curr_time)
			self.parent_block_index.append(-2)
			self.is_leaf.append(False)
			self.depth.append(-1)
			return 2

		# adding new block
		new_block_index = len(self.all_blocks)
		self.all_blocks.append(block)
		self.all_blocks_toa.append(curr_time)
		self.parent_block_index.append(parent_index)
		self.is_leaf.append(True)
		self.is_leaf[parent_index] = False
		self.depth.append(self.depth[parent_index] + 1)
		self.path_from_root[new_block_index] = (self.path_from_root[parent_index]).copy()
		self.path_from_root[new_block_index].append(new_block_index)
		if (not (parent_index in self.block_children_index)) or (len(self.block_children_index[parent_index]) == 0):
			self.block_children_index[parent_index] = [new_block_index]
		else:
			self.block_children_index[parent_index].append(new_block_index)

		# checking if new block is parent to any blocks whose parent is not yet arrived
		blocks_without_parent_indices = np.where(self.parent_block_index == -1)[0]
		for i in blocks_without_parent_indices:
			if self.all_blocks[i].parent_bid == block.bid:
				is_valid = self.is_valid_block(new_block_index, self.all_blocks[i], n)
				if not is_valid: # check if the block, whose parent is not yet arrived, is valid
					self.parent_block_index[i] = -2
					continue
				self.parent_block_index[i] = new_block_index
				self.is_leaf[i] = True
				self.is_leaf[new_block_index] = False
				self.depth[i] = self.depth[new_block_index] + 1
				self.path_from_root[i] = (self.path_from_root[new_block_index]).copy()
				self.path_from_root[i].append(i)
				if (not (new_block_index in self.block_children_index)) or (len(self.block_children_index[new_block_index]) == 0):
					self.block_children_index[new_block_index] = [i]
				else:
					self.block_children_index[new_block_index].append(i)
		
		return 1

class Event:
	def __init__(self, time, eid = 0):
		self.time = time
		self.eid = eid # event id
		'''
		1 - create txn and send (to neighbouring peers)
		2 - receive txn and forward (to neibhouring peers(except the one we receive txn from))
		3 - block creation
		4 - block broadcast (to neighbouring peers)
		5 - block receive and forwarding (to neighbouring peers(except the one we receive block from))
		'''
	# below are some init_* functions to store some relevant infor for each event
	def init_1(self, origin_peer_id):
		self.origin_peer_id = origin_peer_id
	def init_2(self, from_peer_id, to_peer_id, txn):
		self.from_peer_id = from_peer_id
		self.to_peer_id = to_peer_id
		self.txn = txn
	def init_3(self, origin_peer_id):
		self.origin_peer_id = origin_peer_id
	def init_4(self, origin_peer_id, longest_path, longest_length):
		self.origin_peer_id = origin_peer_id
		self.longest_path = longest_path
		self.longest_length = longest_length
	def init_5(self, from_peer_id, to_peer_id, block):
		self.from_peer_id = from_peer_id
		self.to_peer_id = to_peer_id
		self.block = block
	def __lt__(self, nxt):
		return self.time < nxt.time



def get_dij(cij):
	return np.random.exponential(96/cij, 1)[0] # in milli seconds

def get_new_time_gap(T): # will be used for both Ttx, Tk time sampling
	return np.random.exponential(T, 1)[0] # in milli seconds

def get_random_peer(curr_peer, n):
	while 1:
		new_peer = np.random.choice(n, 1)[0]
		if not new_peer == curr_peer: # we don't allow txns(the ones which are generated with interarrival time Ttx) among same peer
			return new_peer



if __name__ == "__main__":
	
	#parameters that can be argparsed
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", default = 12) # >10 needed for later stages
	parser.add_argument("--z", default = 0.5) # percent of slow peers
	parser.add_argument("--Ttx", default = 5000) # in milli seconds, interarrival time for txns mean
	parser.add_argument("--max_events", default = 2000) # halt if the number of events executed exceeds this
	parser.add_argument("--Tk", default = 10000) # in milli seconds, interarrival time for blocks
	parser.add_argument("--max_time", default = 120000) # 2 min in milli seconds
	args=parser.parse_args()
	n = int(args.n)
	z = float(args.z)
	Ttx = int(args.Ttx) # in milli seconds
	max_events = int(args.max_events)
	Tk = int(args.Tk)  # in milli seconds
	max_time = int(args.max_time) # in milli seconds

	hashing_power = np.random.choice(np.arange(1,11), n, replace = True)
	hashing_power = hashing_power*100/np.sum(hashing_power)
	Tk_mean = Tk/hashing_power # the one with less hashing power has high mean for exponential distribution
	max_coins = 10 # a random amount of coins in [1, max_coins] are sent in each txn
	tree_folder = './tree_files/' # folder to save tree file info in, 8th question
	trees = './trees/' # folder to store all visualisations of blockchains for each node

	assert n > 10

	txn_size = 8 # in kilo bits
	tid_counter = 0 # global counter for txn id
	bid_counter = 0 # global counter for block id
	event_queue = [] # will be heapified later
	curr_time = 0 # in milli seconds

	peers = []
	for i in range(n):
		new_peer = Peer()
		new_peer.create_genisis_block(bid_counter, curr_time)
		peers.append(new_peer)
	bid_counter += 1
	no_of_slow_peers = int(np.floor(z*n))
	slow_peer_indices = np.random.choice(n, no_of_slow_peers, replace = False) # randomly choosing z% slow peers
	fast_mask = np.full(n, True)
	slow_mask = np.full(n, False)
	for i in slow_peer_indices:
		fast_mask[i] = False
		slow_mask[i] = True
		peers[i].is_slow = True


	D = max(1, int(np.floor((n//2-2)//3)))  # got by C(==3*D+2) = n/2. each peer have degree in [D,C]
	K = D+7 # pls refer paper for terminology. K is no.of peers in cache
	links = get_topology(n, D, 3*D+2, K)
	#reference for building a P2P network: https://www.researchgate.net/publication/3235764_Building_low-diameter_peer-to-peer_networks

	rho = np.random.uniform(10, 500, (n,n)) # in milli seconds, light propation delay
	rho = (rho + rho.T)/2 # making rho symmetric
	c = np.full((n,n), 0)
	c[links] = 5 # all links to speed 5Mbps
	fnf_mask = np.full((n,n), False) # fast and fast mask
	fnf_mask[fast_mask, :] = True # making fast rows true
	fnf_mask[:, slow_mask] = False # making slow cloumns fast - hence fnf_mask now only have true values if both column and row are fast
	fnf_mask = np.logical_and(fnf_mask, links) # make those in fnf_mask, which dont have links, false
	c[fnf_mask] = 100 # fast-to-fast links speed set to 100Mbps

	hq.heapify(event_queue)
	for i in range(n): # putting 'create txn and send' events with some time delay in event queue
		new_time_gap = get_new_time_gap(Ttx)
		new_event = Event(curr_time+new_time_gap, 1)
		new_event.init_1(i)
		hq.heappush(event_queue, new_event)

	# putting 'block creation' events in event queue
	# each peer generate a block(with only coin-base txn) at time 0 to get some coins initially.
	# From later on he will include txns apart from coin-base
	for i in range(n):
		new_event = Event(curr_time, 3)
		new_event.init_3(i)
		hq.heappush(event_queue, new_event)

	count_loops = 0
	while ((curr_time < max_time) or (max_time == -1)) and len(event_queue) > 0:
		curr_event = hq.heappop(event_queue) # get current event using heap
		curr_time = curr_event.time
		eid = curr_event.eid

		count_loops += 1
		if not max_events == -1:
			if count_loops > max_events:
				print("Simulation ended due to predefined limit on maximum number of events to be evaluated.")
				break
			if count_loops%500 == 0:
				print(count_loops, " events done")

		# handle the event based on eid
		if eid == 1:
			origin_peer_id = curr_event.origin_peer_id
			idy = get_random_peer(origin_peer_id, n)
			coins = np.random.uniform(1, max_coins, 1)[0]
			new_txn = Txn(origin_peer_id, idy, coins, tid_counter)
			tid_counter += 1
			peers[origin_peer_id].my_txn_indices.append(len(peers[origin_peer_id].all_txns))
			peers[origin_peer_id].all_txns.append(new_txn)
			peers[origin_peer_id].all_txns_toa.append(curr_time)
			
			new_time_gap = get_new_time_gap(Ttx) # event 'create txn and send' triggering anohter event of same kind
			new_event = Event(curr_time+new_time_gap, 1)
			new_event.init_1(origin_peer_id)
			hq.heappush(event_queue, new_event)

			nei_peers = np.where(links[origin_peer_id])[0]
			for nei_peer in nei_peers: # sending txn to neighbouring peers
				new_time_gap = get_dij(c[origin_peer_id, nei_peer])+rho[origin_peer_id, nei_peer]+(txn_size/c[origin_peer_id, nei_peer]) # network delay
				new_event = Event(curr_time+new_time_gap, 2) # triggering event to 'receive and forward txn' in neighbouring peers
				new_event.init_2(origin_peer_id, nei_peer, new_txn)
				hq.heappush(event_queue, new_event)

		elif eid == 2:
			from_peer_id, to_peer_id = curr_event.from_peer_id, curr_event.to_peer_id
			txn_obtained = curr_event.txn
			txn_present = False
			for temp_txn in peers[to_peer_id].all_txns:
				if txn_obtained.tid == temp_txn.tid:
					txn_present = True
					break
			if txn_present: # if we have already seen this txn, then don't forward
				continue

			peers[to_peer_id].all_txns.append(txn_obtained)
			peers[origin_peer_id].all_txns_toa.append(curr_time)
			
			nei_peers = np.where(links[to_peer_id])[0]
			for nei_peer in nei_peers: # forward txn to neighbouring peers
				if nei_peer == from_peer_id: # don't forward to where it came from
					continue
				new_time_gap = get_dij(c[to_peer_id, nei_peer])+rho[to_peer_id, nei_peer]+(txn_size/c[to_peer_id, nei_peer]) # network delay
				new_event = Event(curr_time+new_time_gap, 2) # triggering event to 'receive and forward txn' in neighbouring peers
				new_event.init_2(to_peer_id, nei_peer, txn_obtained)
				hq.heappush(event_queue, new_event)

		elif eid == 3:
			origin_peer_id = curr_event.origin_peer_id
			longest_path, longest_length = peers[origin_peer_id].get_longest_chain()
			peers[origin_peer_id].create_new_block(longest_path, bid_counter, origin_peer_id, tid_counter)
			bid_counter += 1 # used for new block
			tid_counter += 1 # used for mining fee txn
			new_time_gap = get_new_time_gap(Tk_mean[origin_peer_id])
			new_event = Event(curr_time+new_time_gap, 4) # as we created block, now trigger an event to broadcast it with some time gap/delay
			new_event.init_4(origin_peer_id, longest_path, longest_length)
			hq.heappush(event_queue, new_event)

		elif eid == 4:
			origin_peer_id = curr_event.origin_peer_id
			longest_path, longest_length = peers[origin_peer_id].get_longest_chain()
			old_longest_path, old_longest_length = curr_event.longest_path, curr_event.longest_length
			
			if (longest_length == old_longest_length) and (longest_path[0] == old_longest_path[0]): # checking if longest path is changed between block creation and forwarding
				
				# if longest path is not changed - add block to current peer's info(all_blocks, is_leaf, depth, ...)
				new_block_index = len(peers[origin_peer_id].all_blocks)
				new_block = peers[origin_peer_id].current_block
				peers[origin_peer_id].my_block_indices.append(new_block_index)
				peers[origin_peer_id].all_blocks.append(new_block)
				peers[origin_peer_id].all_blocks_toa.append(curr_time)
				parent_to_new_block_index = longest_path[0]
				peers[origin_peer_id].is_leaf.append(True)
				peers[origin_peer_id].is_leaf[parent_to_new_block_index] = False
				peers[origin_peer_id].depth.append(peers[origin_peer_id].depth[parent_to_new_block_index] + 1)
				peers[origin_peer_id].path_from_root[new_block_index] = (peers[origin_peer_id].path_from_root[parent_to_new_block_index]).copy()
				peers[origin_peer_id].path_from_root[new_block_index].append(new_block_index)
				peers[origin_peer_id].parent_block_index.append(parent_to_new_block_index)
				
				if (not(parent_to_new_block_index in peers[origin_peer_id].block_children_index)) or \
					(len(peers[origin_peer_id].block_children_index[parent_to_new_block_index]) == 0):
					peers[origin_peer_id].block_children_index[parent_to_new_block_index] = [new_block_index]
				else:
					(peers[origin_peer_id].block_children_index[parent_to_new_block_index]).append(new_block_index)
				
				new_event = Event(curr_time, 3) # set another event to create block without any delay. event 3 adds delay between it and event 4
				new_event.init_3(origin_peer_id)
				hq.heappush(event_queue, new_event)
				
				nei_peers = np.where(links[origin_peer_id])[0]
				for nei_peer in nei_peers: # forward block to neighbouring peers
					new_time_gap = get_dij(c[origin_peer_id, nei_peer])+rho[origin_peer_id, nei_peer]+(txn_size*(len(new_block.txns))/c[origin_peer_id, nei_peer])
					new_event = Event(curr_time+new_time_gap, 5)
					new_event.init_5(origin_peer_id, nei_peer, new_block)
					hq.heappush(event_queue, new_event)

		elif eid == 5:
			from_peer_id, to_peer_id = curr_event.from_peer_id, curr_event.to_peer_id
			block_obtained = curr_event.block
			block_present = False
			for temp_block in peers[to_peer_id].all_blocks:
				if block_obtained.bid == temp_block.bid:
					block_present = True
					break
			if block_present: # if block is already seen, then don't process/forward it
				continue

			before_longest_path, before_longest_length = peers[to_peer_id].get_longest_chain() # store longest path before adding the freshly received block
			added_value = peers[to_peer_id].add_block(block_obtained, curr_time, n) # add the freshly received block
			
			if added_value == 2: # if block is invalid, the don't forward
				continue
			
			if added_value == 1: # if block is succesfully added to block chain - check if longest path is changed
				after_longest_path, after_longest_length = peers[to_peer_id].get_longest_chain()
				if not((before_longest_length == after_longest_length) and (before_longest_path[0] == after_longest_path[0])):
					new_event = Event(curr_time, 3) # if longest path is changed - add an event to create a block immediately. event 3 adds delay between it and event 4
					new_event.init_3(to_peer_id)
					hq.heappush(event_queue, new_event)

			nei_peers = np.where(links[to_peer_id])[0]
			for nei_peer in nei_peers: # forward block to neibouring peers
				if nei_peer == from_peer_id: # don't forward block to where it came from
					continue
				new_time_gap = get_dij(c[to_peer_id, nei_peer])+rho[to_peer_id, nei_peer]+(txn_size*(len(block_obtained.txns))/c[to_peer_id, nei_peer])
				new_event = Event(curr_time+new_time_gap, 5)
				new_event.init_5(to_peer_id, nei_peer, block_obtained)
				hq.heappush(event_queue, new_event)

		else:
			print("Panic: unknown eid in event_queue")
			exit(1)

	if curr_time > max_time:
		print("Simulation ended due to predefined time limit")


	# printing tree files for each node - question 8
	if not os.path.exists(tree_folder):
		print("Panic: the folder in the variable tree_folder doesn't exist")
	else:
		for i in range(n):
			file_name = tree_folder+"node_"+str(i)+".txt"
			file = open(file_name, 'w')
			file.write("block id\t|\tblock num(depth in nodes)\t|\tparent block id\t|\ttime of arrival(in s)\t|\tcomments\n")
			for j in range(len(peers[i].all_blocks)):
				comment = ""
				if peers[i].parent_block_index[j] < 0:
					comment = "Parent not yet arrived" if peers[i].parent_block_index[j] == -1 else "Invalid block"
				file.write(str(peers[i].all_blocks[j].bid)+"\t"+str(peers[i].depth[j])+"\t"+str(peers[i].all_blocks[j].parent_bid)+"\t"+\
					str(round(peers[i].all_blocks_toa[j]/1000, 5))+"\t"+comment+"\n")
			file.close()


	# finding the ratio - ratio of the number of blocks generated by each node in the longest chain 
	# of the tree to the total number of blocks it generates at the end of the simulation
	the_ratio = np.zeros(n)
	for i in range(n):
		longest_path, longest_length = peers[i].get_longest_chain()
		numer = 0
		for j in longest_path:
			if j in peers[i].my_block_indices:
				numer += 1
		the_ratio[i] = numer/len(peers[i].my_block_indices)
	print("Printing ratio of the number of blocks generated by each node in the longest chain of the tree to the total number of blocks it generates at the end of the simulation.")
	print("peer_id\t|\tthe_ratio\t|\tis_fast\t|\thashing_power_percent")
	for i in range(n):
		print(i,"\t",round(the_ratio[i], 4),"\t",fast_mask[i],"\t",round(hashing_power[i], 4))
	

	# printing images of blockchains for each node
	if not os.path.exists(trees):
		print("Panic: the folder in the variable trees doesn't exist")
	else:
		for i in range(n):
			file_name = trees+ "node_"+str(i)+".png"
			l = []
			for block_index in peers[i].block_children_index:
				for child_index in peers[i].block_children_index[block_index]:
					l.append((peers[i].all_blocks[block_index].bid, peers[i].all_blocks[child_index].bid))
			g=nx.DiGraph()
			g.add_edges_from(l)
			p=nx.drawing.nx_pydot.to_pydot(g)
			p.write_png(file_name)









