import sys
import time
import pprint

from web3 import *
from solc import compile_source
import os

def compile_source_file(file_path):
   with open(file_path, 'r') as f:
      source = f.read()
   return compile_source(source)

def read_address_file(file_path):
    file = open(file_path, 'r')
    addresses = file.read().splitlines() 
    return addresses

def connectWeb3():
    # print(os.environ['HOME']+'/HW3/test-eth1/geth.ipc')
    return Web3(IPCProvider(os.environ['HOME']+'/HW3/test-eth1/geth.ipc', timeout=100000))

def deployEmptyContract(contract_source_path, w3, account):
    compiled_sol = compile_source_file(contract_source_path)
    contract_id, contract_interface3 = compiled_sol.popitem()
    curBlock = w3.eth.getBlock('latest')
    tx_hash = w3.eth.contract(
            abi=contract_interface3['abi'],
            bytecode=contract_interface3['bin']).constructor().transact({'txType':"0x0", 'from':account, 'gas':40000000, 'gasPrice':100})
    return tx_hash

def deployContracts(w3, account):
    tx_hash = deployEmptyContract(empty_source_path, w3, account)
    receipt = w3.eth.getTransactionReceipt(tx_hash)

    while receipt is None:
        time.sleep(1)
        receipt = w3.eth.getTransactionReceipt(tx_hash)

    w3.miner.stop()

    if receipt is not None:
        print("empty:{0}".format(receipt['contractAddress']))

if __name__ == '__main__':

    empty_source_path = os.environ['HOME']+'/HW3/dapp.sol'
    w3 = connectWeb3()
    w3.miner.start(1)
    time.sleep(4)
    deployContracts(w3, w3.eth.accounts[0])