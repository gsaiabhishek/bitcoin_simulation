install go(as given in instructions before)
install solidity 0.4.25 and geth 1.9.3

Not sharing all files in the submission. Only sharing those which need to be added/replaced.

place runEthereumNode.sh, deploy_contract.py, sendTxns.py, dapp.sol in the same folder where deployContract.py, sendTransaction.py is there which is given to us as HW3.

run:
$ $HOME/go-ethereum/build/bin/geth --datadir $HOME/HW3/test-eth1/ --password $HOME/HW3/password.txt account new
copy the address to the genesis.json in the alloc section

$ sh runEthereumNode.sh
to setup ethereum node

$ python3 deploy_contract.py > contractAddressList
to deploy contract

set the required variables in sendTxns.py. Default uses 100 users. then run:
$ python3 sendTxns.py
to fire transactions for registering users, creating accounts, sending unit amounts, closing accounts, plotting graph.