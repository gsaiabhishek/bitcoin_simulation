import sys
import time
import pprint

from web3 import *
from solc import compile_source
import os

from networkx import powerlaw_cluster_graph, is_connected
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compile_source_file(file_path):
   with open(file_path, 'r') as f:
      source = f.read()
   return compile_source(source)

def registerUser(sort_contract, uid, uname):
    tx_hash = sort_contract.functions.registerUser(uid, uname).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    return tx_hash

def createAcc(sort_contract, uid1, uid2, each_bal):
    tx_hash = sort_contract.functions.createAcc(uid1, uid2, each_bal).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    return tx_hash

def sendAmount(sort_contract, uid1, uid2):
    tx_hash = sort_contract.functions.sendAmount(uid1, uid2).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    return tx_hash

def closeAccount(sort_contract, uid1, uid2):
    tx_hash = sort_contract.functions.closeAccount(uid1, uid2).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    return tx_hash

def is_last_txn_success(sort_contract):
    last_txn_success = sort_contract.functions.is_last_txn_success().call()
    return last_txn_success

# wait till we get a receipt of tx_hash ie, wait till trasaction gets into a block
def process_txhash(tx_hash):
    receipt = w3.eth.getTransactionReceipt(tx_hash)

    while receipt is None:
        time.sleep(0.1)
        receipt = w3.eth.getTransactionReceipt(tx_hash)

    receipt = w3.eth.getTransactionReceipt(tx_hash)
    return receipt


if __name__ == '__main__':

    num_users = 10
    avg_edges = 2
    prob_triangle = 0.05

    addr = None
    # each run gets a datapoint in final graph. each run takes success fraction of epoch-no.of sendAmount txns.
    epochs = 10
    runs = 3
    sleep_min = 1.5 # amount of time to sleep before starting to call any sendAmount funtions
    stats_path = 'stats'
    contract_source_path = os.environ['HOME']+'/HW3/dapp.sol'

    np.random.seed(50)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--is_construct", default = 0, type = int)
    # args=parser.parse_args()

    w3 = Web3(IPCProvider(os.environ['HOME']+'/HW3/test-eth1/geth.ipc', timeout=100000))

    with open(os.environ['HOME']+'/HW3/contractAddressList') as fp:
        for line in fp:
            a_,b_ = line.rstrip().split(':', 1)
            if a_ == "empty":
                addr = b_
            time.sleep(0.01)

    print('addr:', addr)

    compiled_sol = compile_source_file(contract_source_path)
    contract_id, contract_interface = compiled_sol.popitem()
    sort_contract = w3.eth.contract(address=addr, abi=contract_interface['abi'])

    w3.miner.start(1)
    txn_hashes = []

    # registering users below
    for i in range(num_users):
        txn_hash = registerUser(sort_contract, i, 'a')
        txn_hashes.append(txn_hash)
        # receipt = process_txhash(tx_hash)
    print('All users are registed')

    # generating power law graph below
    graph = None
    while True:
      graph = powerlaw_cluster_graph(num_users, avg_edges, prob_triangle, seed = 25)
      if is_connected(graph): # if not connected, generate power law graph again
        break
    edges = graph.edges
    print('No of edges:', len(edges))

    # creating accounts between users below
    for edge in edges:
        bal = np.random.exponential(10,1)[0]
        each_bal = int(bal/2)
        txn_hash = createAcc(sort_contract, edge[0], edge[1], each_bal)
        txn_hashes.append(txn_hash)
        # receipt = process_txhash(tx_hash)
    print('Accounts have been created between users')

    #sleeping to give time for previous transactions to get into blocks
    print('Sleeping..')
    time.sleep(sleep_min*60)
    print('Woke up..')

    #check if all transactions above are inside a block, else wait
    t1 = time.time()
    for txn_hash in txn_hashes:
        receipt = process_txhash(txn_hash)
    delta_t1 = time.time() - t1
    print('Elapsed time for processing register, create txns:', delta_t1)

    #send unit amounts below. for each run we plot a point in final graph. each run takes success fraction of epochs-no.of sendAmount txns.
    frac_success = np.zeros(runs)
    for i in range(runs):
        print('Run',i+1,'/',runs,'started')
        for j in range(epochs):
            nodes = np.random.choice(num_users, 2, replace = False)
            txn_hash = sendAmount(sort_contract, int(nodes[0]), int(nodes[1]))
            try:
                receipt = process_txhash(txn_hash)
                frac_success[i] += int(is_last_txn_success(sort_contract))
            except:
                print(receipt)
                print(len(receipt['logs']))
                print('Failed at', i, j)
                print('While sending amount between nodes:', nodes)
                exit(0)
            if (j+1) % 5 == 0:
                print('    Epoch',j+1,'/',epochs,'completed')
        frac_success[i] /= epochs

    for edge in edges:
        txn_hash = closeAccount(sort_contract, edge[0], edge[1])
        # receipt = process_txhash(tx_hash)
    print('Accounts have been closed between users')

    w3.miner.stop()

    #plot the graph of success fraction vs run
    plt.plot(1+np.arange(runs), frac_success, color = 'b')
    plt.xlabel('run, each run is '+str(epochs)+' transactions of sendAmount')
    plt.ylabel('fraction of successful transactions')
    plt.title('Num users: '+str(num_users)+'. Avg edges: '+str(avg_edges)+'. Clustering coeff: '+str(prob_triangle))
    plt.savefig(stats_path+'_'+str(num_users)+'_'+str(avg_edges)+'_'+str(prob_triangle)+'.png')
    plt.close()