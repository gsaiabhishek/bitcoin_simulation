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
	def __init__(self, is_slow = False, is_adversary = False):
		self.is_slow = is_slow
		self.all_txns = [] # stores all txns seen so far
		self.all_txn_tids = [] # to make computation in creating a block faster
		self.all_txns_toa = [] # time of arrival/creation
		self.my_txn_indices = []

		self.current_block = None # that is being created/mined
		self.all_blocks = [] # stores all blocks seen so far
		self.my_block_indices = [] # need not be same size as of all_blocks
		self.all_blocks_toa = [] # time of arrival/creation. This doesn not have private blocks, in case of adversary
		self.is_leaf = [] # same size as of all_blocks
		self.depth = [] # same size as of all_blocks. depth is number of nodes(not edges) till that node from root
		self.path_from_root = {} # same size as of all_blocks. Each index of all_block will have a key here which maps to list of indices which are path from root till this node
		self.root_index = None # will be initialised to 0 when genisis block is created
		self.parent_block_index = [] # same size as of all_blocks
		# special parents: itself means root, -1 means block not in blockchain yet/waiting for parent, -2 means invalid block
		self.block_children_index = {} # map: index in all_blocks -> [index_in_all_blocks]

		self.is_adversary = is_adversary
		self.private_block_list = []
		self.mining_on_block_index = None # the point of contact in main chain where private chain is started. If none, it means we need to find a new longest chain to build private chain on.
		self.num_private_blocks_gen = 0
		self.is_competition = False # this is true if we are in state 0' and we are doing selfish mining.

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
				if self.all_blocks_toa[i] < self.all_blocks_toa[best_leaf_index]:
					best_leaf_index, best_leaf_depth = i, self.depth[i]

		temp_path = (self.path_from_root[best_leaf_index]).copy()
		temp_path.reverse()
		return temp_path, self.depth[best_leaf_index]


	def create_new_block(self, path, bid, peer_id, tid): # just create a block from "txns in txn pool but not in longest path"
		
		parent_bid = self.all_blocks[path[0]].bid
		if self.is_adversary:
			if self.mining_on_block_index == None:
				self.mining_on_block_index = path[0]
			else:
				path = (self.path_from_root[self.mining_on_block_index]).copy()
				path.reverse()
				if len(self.private_block_list) > 0:
					parent_bid = self.private_block_list[-1].bid
				else:
					parent_bid = self.all_blocks[path[0]].bid

		# finding allowed txn to choose from
		existing_txns = []
		for block_index in path:
			existing_txns.extend(self.all_blocks[block_index].txns)

		if self.is_adversary:
			for block in self.private_block_list:
				existing_txns.extend(block.txns)

		existing_txn_tids = [txn_.tid for txn_ in existing_txns]
		allowed_txns_index = []
		for i in range(len(self.all_txn_tids)):
			if not( self.all_txn_tids[i] in existing_txn_tids ):
				allowed_txns_index.append(i)
		
		low_bound = 1
		if len(allowed_txns_index) == 0:
			low_bound = 0 # setting low_bound as 0, if no txns are there to randomly choose from
		no_of_txns = np.random.randint(low_bound, min(len(allowed_txns_index)+1, 1000), 1)[0] # random count of txns choosen
		txns_index = np.random.choice(np.array(allowed_txns_index), no_of_txns, replace = False) # random txns choosen
		
		self.current_block = Block(bid, parent_bid)
		for i in txns_index:
			self.current_block.add_txn(self.all_txns[i])
		mining_fee_txn = Txn(None, peer_id, 50, tid) # mining fee txn, convention: idx == None => mining fee txn
		self.current_block.add_txn(mining_fee_txn)
		
		# return self.mining_on_block_index if self.is_adversary else path[0]
		return bid

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

	def add_block(self, block, curr_time, n): # function for adding others blocks to my chain
		'''
		return value:
		0 - not added since parent is not yet there
		1 - successfully added
		2 - invalid block
		'''
		parent_present = False
		parent_index = None
		for i in range(len(self.all_blocks)):
			if (not parent_present) and (self.all_blocks[i].bid == block.parent_bid) and (not (self.parent_block_index[i] < 0)):
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

	def add_my_block(self, parent_to_new_block_index, new_block, curr_time): # function for adding my blocks to my chain
		
		if parent_to_new_block_index == None:
			print("Panic: parent index in None in add_my_block in Peer. Stopping.")
			exit(0)

		if self.all_blocks[parent_to_new_block_index].bid == new_block.bid:
			print("Panic: parent bid and new_block bid are same in add_my_block in Peer. eid, delta_prev: ", eid, delta_prev)
			return parent_to_new_block_index

		new_block_index = len(self.all_blocks)
		self.my_block_indices.append(new_block_index)
		self.all_blocks.append(new_block)
		self.all_blocks_toa.append(curr_time)
		self.is_leaf.append(True)
		self.is_leaf[parent_to_new_block_index] = False
		self.depth.append(self.depth[parent_to_new_block_index] + 1)
		self.path_from_root[new_block_index] = (self.path_from_root[parent_to_new_block_index]).copy()
		self.path_from_root[new_block_index].append(new_block_index)
		self.parent_block_index.append(parent_to_new_block_index)
		
		if (not(parent_to_new_block_index in self.block_children_index)) or \
			(len(self.block_children_index[parent_to_new_block_index]) == 0):
			self.block_children_index[parent_to_new_block_index] = [new_block_index]
		else:
			(self.block_children_index[parent_to_new_block_index]).append(new_block_index)

		return new_block_index


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
	def init_4(self, origin_peer_id, bid):
		self.origin_peer_id = origin_peer_id
		self.bid = bid #the block id of block which is mined during event 3
		# self.longest_path_leaf = longest_path_leaf # mining blk index(which is public) in case of adversary, leaf index in all_blocks list in case of honest miners
		# self.longest_length = longest_length
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

def allot_adversary(index, total_adversaries):
	return True if index < total_adversaries else False

if __name__ == "__main__":
	
	#parameters that can be argparsed
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", default = 12) # n-total_adversaries >10 needed for later stages
	parser.add_argument("--z", default = 0.5) # percent of slow peers among honest miners
	parser.add_argument("--eta", default = 0.5) # percent of honest peers adversary is connected to
	parser.add_argument("--Ttx", default = 5000) # in milli seconds, interarrival time for txns mean
	parser.add_argument("--max_events", default = 2000) # halt if the number of events executed exceeds this, -1 => no limit on events
	parser.add_argument("--Tk", default = 10000) # in milli seconds, interarrival time for blocks
	parser.add_argument("--max_time", default = 18000) # 18 s in milli seconds, -1 => no limit on time
	parser.add_argument("--adv_hash", default = 0.3) # hash power of one adversary
	parser.add_argument("--is_selfish", default = 1) # 1 for selfish, 0 for stubborn
	parser.add_argument("--total_adversaries", default = 1) # 1 for any attack, 0 for normal(100% honest miners) simulation
	args=parser.parse_args()
	n = int(args.n)
	z = float(args.z)
	eta = float(args.eta)
	Ttx = int(args.Ttx) # in milli seconds
	max_events = int(args.max_events)
	Tk = int(args.Tk)  # in milli seconds
	max_time = int(args.max_time) # in milli seconds
	adv_hash = float(args.adv_hash)
	is_selfish = bool(int(args.is_selfish))
	total_adversaries = int(args.total_adversaries) # number of adversaries of a single pool. First these many peers will be adversaries

	max_coins = 10 # a random amount of coins in [1, max_coins] are sent in each txn
	tree_folder = './tree_files/' # folder to save tree file info in
	trees = './trees/' # folder to store all visualisations of blockchains for each node
	if total_adversaries > 0:
		if is_selfish:
			tree_folder = './selfish/tree_files/' # folder to save tree file info in
			trees = './selfish/trees/' # folder to store all visualisations of blockchains for each node
		else:
			tree_folder = './stubborn/tree_files/' # folder to save tree file info in
			trees = './stubborn/trees/' # folder to store all visualisations of blockchains for each node

	honest_n = n-total_adversaries
	assert honest_n > 10
	assert adv_hash*total_adversaries < 1

	txn_size = 8 # in kilo bits
	tid_counter = 0 # global counter for txn id
	bid_counter = 0 # global counter for block id
	event_queue = [] # will be heapified later
	curr_time = 0 # in milli seconds

	hashing_power = np.zeros(n)
	peers = []
	for i in range(n):
		new_peer = Peer(is_adversary = True) if allot_adversary(i, total_adversaries) else Peer() # peer 0 is adversary
		new_peer.create_genisis_block(bid_counter, curr_time)
		peers.append(new_peer)
		if allot_adversary(i, total_adversaries):
			hashing_power[i] = adv_hash*100 # hash power to adversaries

	hashing_power[total_adversaries:] = np.random.choice(np.arange(1,11), honest_n, replace = True)
	hashing_power[total_adversaries:] = hashing_power[total_adversaries:]*100*(1-(adv_hash*total_adversaries))/np.sum(hashing_power[total_adversaries:])
	Tk_mean = Tk/hashing_power # the one with less hashing power has high mean for exponential distribution

	bid_counter += 1 # 0 bid is for genisis blocks of all peers
	
	no_of_slow_peers = int(np.floor(z*(honest_n)))
	slow_peer_indices = np.random.choice(np.arange(total_adversaries,n), no_of_slow_peers, replace = False) # randomly choosing z% slow peers among honest miners
	# above, we start from total_adversaries to choose slow peers, because adversaries are in range [0, total_adversaries-1] and are always fast
	fast_mask = np.full(n, True)
	slow_mask = np.full(n, False)
	for i in slow_peer_indices:
		fast_mask[i] = False
		slow_mask[i] = True
		peers[i].is_slow = True

	D = max(1, int(np.floor((honest_n//2-2)//3)))  # got by C(==3*D+2) = honest_n/2. each peer have degree in [D,C]
	K = D+7 # pls refer paper for terminology. K is no.of peers in cache
	honest_links = get_topology(honest_n, D, 3*D+2, K)
	#reference for building a P2P network: https://www.researchgate.net/publication/3235764_Building_low-diameter_peer-to-peer_networks
	links = np.zeros([n,n], dtype = bool)
	links[total_adversaries:n, total_adversaries:n] = honest_links
	no_of_honest_for_each_adversary = int(np.floor(eta*(honest_n)))
	for i in range(total_adversaries):
		honest_nei = np.random.choice(np.arange(total_adversaries,n), no_of_honest_for_each_adversary, replace = False)
		links[i, honest_nei] = True
		links[honest_nei, i] = True
		#todo: may need to include some connections between adversaries if there are > 1 of them in pool
		#todo: if there is a pool, may need changes below in the event simulator loop as well

	rho = np.random.uniform(10, 500, (n,n)) # in milli seconds, light propation delay
	rho = (rho + rho.T)/2 # making rho symmetric
	c = np.full((n,n), 0)
	c[links] = 5 # all links to speed 5Mbps
	fnf_mask = np.full((n,n), False) # fast and fast mask
	fnf_mask[fast_mask, :] = True # making fast rows true
	fnf_mask[:, slow_mask] = False # making slow cloumns false - hence fnf_mask now only have true values if both column and row are fast
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
			print(count_loops, " events done. curr time:", round(curr_time, 3),"ms")

		# handle the event based on eid
		if eid == 1:
			origin_peer_id = curr_event.origin_peer_id
			idy = get_random_peer(origin_peer_id, n)
			coins = np.random.uniform(1, max_coins, 1)[0]
			new_txn = Txn(origin_peer_id, idy, coins, tid_counter)
			tid_counter += 1
			peers[origin_peer_id].my_txn_indices.append(len(peers[origin_peer_id].all_txns))
			peers[origin_peer_id].all_txns.append(new_txn)
			peers[origin_peer_id].all_txn_tids.append(new_txn.tid)
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
			peers[to_peer_id].all_txn_tids.append(txn_obtained.tid)
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
			bid_ = peers[origin_peer_id].create_new_block(longest_path, bid_counter, origin_peer_id, tid_counter)
			bid_counter += 1 # used for new block
			tid_counter += 1 # used for mining fee txn
			new_time_gap = get_new_time_gap(Tk_mean[origin_peer_id])
			new_event = Event(curr_time+new_time_gap, 4) # as we created block, now trigger an event to broadcast it with some time gap/delay
			new_event.init_4(origin_peer_id, bid_)
			hq.heappush(event_queue, new_event)

		elif eid == 4:
			origin_peer_id = curr_event.origin_peer_id

			if peers[origin_peer_id].current_block.bid == curr_event.bid:

				is_adversary = peers[origin_peer_id].is_adversary
				longest_path, longest_length = peers[origin_peer_id].get_longest_chain()

				if not is_adversary:
					new_block = peers[origin_peer_id].current_block
					_ = peers[origin_peer_id].add_my_block(longest_path[0], new_block, curr_time)
					
					new_event = Event(curr_time, 3) # set another event to create block without any delay. event 3 adds delay between it and event 4
					new_event.init_3(origin_peer_id)
					hq.heappush(event_queue, new_event)
					
					nei_peers = np.where(links[origin_peer_id])[0]
					for nei_peer in nei_peers: # forward block to neighbouring peers
						new_time_gap = get_dij(c[origin_peer_id, nei_peer])+rho[origin_peer_id, nei_peer]+(txn_size*(len(new_block.txns))/c[origin_peer_id, nei_peer])
						new_event = Event(curr_time+new_time_gap, 5)
						new_event.init_5(origin_peer_id, nei_peer, new_block)
						hq.heappush(event_queue, new_event)

				elif is_adversary:
					prev_len_private_chain = peers[origin_peer_id].depth[peers[origin_peer_id].mining_on_block_index] + len(peers[origin_peer_id].private_block_list)
					delta_prev = prev_len_private_chain - longest_length

					if delta_prev == 0 and peers[origin_peer_id].is_competition and is_selfish: # in case of stubborn, we go from 0' to 1 and not 0.
						new_block = peers[origin_peer_id].current_block
						new_block_index = peers[origin_peer_id].add_my_block(peers[origin_peer_id].mining_on_block_index, new_block, curr_time)

						peers[origin_peer_id].mining_on_block_index = new_block_index
						peers[origin_peer_id].private_block_list = []
						peers[origin_peer_id].is_competition = False

						nei_peers = np.where(links[origin_peer_id])[0]
						for nei_peer in nei_peers: # forward block to neighbouring peers
							new_time_gap = get_dij(c[origin_peer_id, nei_peer])+rho[origin_peer_id, nei_peer]+(txn_size*(len(new_block.txns))/c[origin_peer_id, nei_peer])
							new_event = Event(curr_time+new_time_gap, 5)
							new_event.init_5(origin_peer_id, nei_peer, new_block)
							hq.heappush(event_queue, new_event)
					else:
						peers[origin_peer_id].private_block_list.append(peers[origin_peer_id].current_block)

					peers[origin_peer_id].num_private_blocks_gen += 1
					
					new_event = Event(curr_time, 3)
					new_event.init_3(origin_peer_id)
					hq.heappush(event_queue, new_event)

		elif eid == 5:
			from_peer_id, to_peer_id = curr_event.from_peer_id, curr_event.to_peer_id
			block_obtained = curr_event.block
			blocks_to_forward = [block_obtained]
			block_present = False
			for temp_block in peers[to_peer_id].all_blocks:
				if block_obtained.bid == temp_block.bid:
					block_present = True
					break
			if block_present: # if block is already seen, then don't process/forward it
				continue

			before_longest_path, before_longest_length = peers[to_peer_id].get_longest_chain() # store longest path before adding the freshly received block
			added_value = peers[to_peer_id].add_block(block_obtained, curr_time, n) # add the freshly received block

			if added_value == 2: # if block is invalid, then don't forward
				continue
			
			prev_len_private_chain, delta_prev = None, 0 # delta_prev = length of private chain-length of honest chain before adding the received block
			is_adversary = peers[to_peer_id].is_adversary
			if is_adversary and not (peers[to_peer_id].mining_on_block_index == None):
				prev_len_private_chain = peers[to_peer_id].depth[peers[to_peer_id].mining_on_block_index] + len(peers[to_peer_id].private_block_list)
				delta_prev = prev_len_private_chain - before_longest_length

			if added_value == 1: # if block is succesfully added to block chain - check if longest path is changed
				if not is_adversary:
					after_longest_path, after_longest_length = peers[to_peer_id].get_longest_chain()
					if not((before_longest_length == after_longest_length) and (before_longest_path[0] == after_longest_path[0])):
						new_event = Event(curr_time, 3) # if longest path is changed - add an event to create a block immediately. event 3 adds delay between it and event 4
						new_event.init_3(to_peer_id)
						hq.heappush(event_queue, new_event)
				else:
					blocks_to_forward = [] # making list empty => not forwarding the block we got from honest miners
					if delta_prev < 0: # we put this just to check if its possible. we didn't observe these cases.
						print("Note: delay_prev is < 0.")
						peers[to_peer_id].private_block_list = []
						peers[to_peer_id].mining_on_block_index = None
						new_event = Event(curr_time, 3)
						new_event.init_3(to_peer_id)
						hq.heappush(event_queue, new_event)
					elif delta_prev == 0:
						peers[to_peer_id].private_block_list = []
						new_event = Event(curr_time, 3)
						new_event.init_3(to_peer_id)
						hq.heappush(event_queue, new_event)
						peers[to_peer_id].mining_on_block_index = None
						peers[to_peer_id].is_competition = False
					elif delta_prev == 1:
						if not len(peers[to_peer_id].private_block_list) == 1:
							print("Panic: delta_prev is 1 but no of private blocks are not 1. Not stopping.")
						blocks_to_forward.append(peers[to_peer_id].private_block_list[0])
						new_block_index = peers[to_peer_id].add_my_block(peers[to_peer_id].mining_on_block_index, peers[to_peer_id].private_block_list[0], curr_time)
						peers[to_peer_id].mining_on_block_index = new_block_index
						peers[to_peer_id].private_block_list = peers[to_peer_id].private_block_list[1:]
						peers[to_peer_id].is_competition = True # ie, we are going to be in state 0'

					elif delta_prev == 2 and is_selfish: # in case of stubborn, we forward only one private block even if delta_prev = 2 which is handled in else.
						while len(peers[to_peer_id].private_block_list) > 0:
							blocks_to_forward.append(peers[to_peer_id].private_block_list[0])
							new_block_index = peers[to_peer_id].add_my_block(peers[to_peer_id].mining_on_block_index, peers[to_peer_id].private_block_list[0], curr_time)
							peers[to_peer_id].mining_on_block_index = new_block_index
							peers[to_peer_id].private_block_list = peers[to_peer_id].private_block_list[1:]
						peers[to_peer_id].private_block_list = []

					else:
						if not len(peers[to_peer_id].private_block_list) == 0:
							blocks_to_forward.append(peers[to_peer_id].private_block_list[0])
							new_block_index = peers[to_peer_id].add_my_block(peers[to_peer_id].mining_on_block_index, peers[to_peer_id].private_block_list[0], curr_time)
							peers[to_peer_id].mining_on_block_index = new_block_index
							peers[to_peer_id].private_block_list = peers[to_peer_id].private_block_list[1:]
						else:
							print("Note: length of private chain is 0 in epoch(indexed from 1) ", count_loops, ".")

			if len(blocks_to_forward) == 0:
				continue

			nei_peers = np.where(links[to_peer_id])[0]
			for nei_peer in nei_peers: # forward block to neibouring peers
				if nei_peer == from_peer_id and blocks_to_forward[0].bid == block_obtained.bid: # don't forward block to where it came from if to_peer_id is not adversary
					continue
				for temp_block in blocks_to_forward:
					new_time_gap = get_dij(c[to_peer_id, nei_peer])+rho[to_peer_id, nei_peer]+(txn_size*(len(block_obtained.txns))/c[to_peer_id, nei_peer])
					new_event = Event(curr_time+new_time_gap, 5)
					new_event.init_5(to_peer_id, nei_peer, temp_block)
					hq.heappush(event_queue, new_event)

		else:
			print("Panic: unknown eid in event_queue.")
			exit(1)

	if curr_time > max_time:
		print("Simulation ended due to predefined time limit.")


	# printing tree files for each node - question 8
	if not os.path.exists(tree_folder):
		print("Panic: the folder in the variable tree_folder doesn't exist.")
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
	mpu_node_adv = None # todo: assuming 1 adversary here
	mpu_node_overall_numer, mpu_node_overall_demomin = 0, 0
	R_pool = np.zeros(n)
	R_pool_denomin = np.zeros(n) # this has the longest chain of each node in their block chain
	for i in range(n):
		longest_path, longest_length = peers[i].get_longest_chain()
		numer = 0
		mpu_node_overall_demomin += (len(peers[i].my_block_indices)-1)
		for j in longest_path:
			if j == peers[i].root_index:
				continue
			if j in peers[i].my_block_indices:
				numer += 1
		the_ratio[i] = numer/(len(peers[i].my_block_indices)-1) if len(peers[i].my_block_indices) > 1 else -1
		R_pool[i] = numer/(longest_length-1) if longest_length > 1 else -1
		R_pool_denomin[i] = (longest_length-1)
		if peers[i].is_adversary:
			mpu_node_adv = (numer+len(peers[i].private_block_list))/peers[i].num_private_blocks_gen if peers[i].num_private_blocks_gen > 0 else -1 # todo: check numerator in both mpu ratios
			mpu_node_overall_numer = longest_length -1 + len(peers[i].private_block_list)
			mpu_node_overall_demomin += len(peers[i].private_block_list)

	print("Upto peer id", total_adversaries-1, "are adversaries.")
	print("Printing 'the ratio' of the number of blocks generated by each node in the longest chain of the tree to the total number of blocks it generates at the end of the simulation.")
	print("peer_id\t|\tthe_ratio( denomin )\t|\tis_fast\t|\thashing_power_percent")
	for i in range(n):
		print(i,"\t",round(the_ratio[i], 3),"(",len(peers[i].my_block_indices)-1,")","\t",fast_mask[i],"\t",round(hashing_power[i], 3))

	mpu_node_overall = mpu_node_overall_numer/mpu_node_overall_demomin if mpu_node_overall_demomin > 0 else -1
	print("Mpu node avg = # blocks of adversary in main chain(including private chain)/# total blocks generated by adversary")
	print("Mpu node overall = # blocks in main chain of adversary 0(including private chain)/# total blocks generated by all nodes(including private blocks)")
	if not mpu_node_adv == None:
		print("Mpu node adv:", round(mpu_node_adv, 3), ", Mpu node overall:", round(mpu_node_overall, 3))
	

	print("adversary peer id\t|\tno of private blocks left\t|\tR_pool( denomin )")
	# printing images of blockchains for each node
	if not os.path.exists(trees):
		print("Panic: the folder in the variable trees doesn't exist.")
	else:
		for i in range(n):
			file_name = trees+ "node_"+str(i)+".png"
			l = []
			for block_index in peers[i].block_children_index:
				for child_index in peers[i].block_children_index[block_index]:
					l.append((peers[i].all_blocks[block_index].bid, peers[i].all_blocks[child_index].bid))
			if peers[i].is_adversary:
				for block in peers[i].private_block_list:
					l.append((block.parent_bid, block.bid))
				print(i, "\t", len(peers[i].private_block_list), "\t", round(R_pool[i], 3), "(", R_pool_denomin[i], ")")
			g=nx.DiGraph()
			g.add_edges_from(l)
			p=nx.drawing.nx_pydot.to_pydot(g)
			p.write_png(file_name)
	print("Note: The blockchain visualisations contain private chains too.")








