pragma solidity ^0.4.24;

contract MyContract{
    struct User{
        uint uid;
        string uname;
        uint[] nei; // uids of neighbours
        uint[] bal;
        int par; // indices of parent in users array
        int depth;
    }
    
    event Deposit(address indexed _from, uint _value);
    
    User[] public users;
    uint num_users = 0;
    uint constant unit_amount = 1;
    uint public last_txn_success;
    mapping(uint => uint) uid_to_index;
    
    mapping(uint => uint) queue;
    uint first = 1;
    uint last = 0;
    
    constructor() public{
    }
    
    //////////////////// queue functions start ////////////////////
    function push(uint data) internal {
        last += 1;
        queue[last] = data;
    }
    
    function length() internal view returns (int data){
        data = int(last)-int(first)+1;
        return data;
    }
    
    function reinit() internal {
        last = 0;
        first = 1;
    }

    function pop() internal returns (uint data) {
        require(last >= first);  // non-empty queue
        data = queue[first];
        delete queue[first];
        first += 1;
        return data;
    }
    //////////////////// queue functions end ////////////////////
    
    function getNeiIndex(uint _index, uint _uid) internal view returns(uint nei_index){
        bool nei_present = false;
        for(uint curr = 0; curr < users[_index].nei.length; curr++){
            if(users[_index].nei[curr] == _uid){
                nei_index = curr;
                nei_present = true;
                break;
            }
        }
        require(nei_present, "given user ids are not neighbours in the graph");
        return nei_index;
    }
    
    function removeNei(uint _index, uint _uid) internal {
        uint nei_index = getNeiIndex(_index, _uid);
        uint num_nei = users[_index].nei.length;
        require(num_nei == users[_index].bal.length, "num nei and num balances are not equal");
    
        if(num_nei > 1){
            users[_index].nei[nei_index] = users[_index].nei[num_nei-1];
            users[_index].bal[nei_index] = users[_index].bal[num_nei-1];
        }
        users[_index].nei.length--;
        users[_index].bal.length--;
    
    }
    
    function registerUser(uint _uid, string _uname) public {
        uint curr;
        bool is_present = false;
        for(curr = 0; curr < num_users; curr++){
            if(users[curr].uid == _uid){
                is_present = true;
                break;
            }
        }
        require(!is_present, "user id already exists");
        if(!is_present){
            User memory user;
            user.uid = _uid;
            user.uname = _uname;
            user.par = -1;
            user.depth = -1;
            users.push(user);
            
            uid_to_index[_uid] = num_users;
            num_users++;
        }
    }
    
    function createAcc(uint _uid1, uint _uid2, uint _each_bal) public {
        uint index1 = uid_to_index[_uid1];
        uint index2 = uid_to_index[_uid2];

        bool is_connected = false;
        uint nei1;
        uint nei2;
        uint curr;
        for(curr = 0; curr < users[index1].nei.length; curr++){
            if(users[index1].nei[curr] == _uid2){
                is_connected = true;
                nei1 = curr;
                break;
            }
        }
        if(!is_connected){
            users[index1].nei.push(_uid2);
            users[index1].bal.push(_each_bal);
            
            users[index2].nei.push(_uid1);
            users[index2].bal.push(_each_bal);
        }else{
            for(curr = 0; curr < users[index2].nei.length; curr++){
                if(users[index2].nei[curr] == _uid1){
                    nei2 = curr;
                    break;
                }
            }
            users[index1].bal[nei1] = _each_bal;
            users[index2].bal[nei2] = _each_bal;
        }
    }

    function is_last_txn_success() view public returns (uint) {
        return last_txn_success;
    }
    
    function sendAmount(uint _uid1, uint _uid2) public{
        uint curr;
        for(curr = 0; curr < num_users; curr++){
            users[curr].par = -1;
            users[curr].depth = -1;
            uid_to_index[users[curr].uid] = curr;
        }
        
        uint index1 = uid_to_index[_uid1];
        uint index2 = uid_to_index[_uid2];
        uint nei_index;
        users[index1].par = int(index1); //root of bfs if itself is parent
        users[index1].depth = 0;
        reinit();
        push(index1);
        while(length() > 0){
            uint front_index = pop();
            for(curr = 0; curr < users[front_index].nei.length; curr++){
                nei_index = uid_to_index[users[front_index].nei[curr]];
                if(users[nei_index].par == -1 && users[front_index].bal[curr] >= unit_amount){
                    users[nei_index].par = int(front_index);
                    users[nei_index].depth = users[front_index].depth+1;
                    if(nei_index == index2){
                        reinit();
                        break;
                    }
                    push(nei_index);
                }
            }
        }
        if(users[index2].par == -1){
            last_txn_success = 0;
            emit Deposit(msg.sender, 0);
        }else{
            uint curr_par_index = uint(users[index2].par);
            uint curr_child_uid = _uid2;
            while(users[index1].uid != curr_child_uid){
                nei_index = getNeiIndex(curr_par_index, curr_child_uid);
                users[curr_par_index].bal[nei_index] -= unit_amount;
                
                nei_index = getNeiIndex(uid_to_index[curr_child_uid], users[uint(curr_par_index)].uid);
                users[uid_to_index[curr_child_uid]].bal[nei_index] += unit_amount;
                
                curr_child_uid = users[curr_par_index].uid;
                curr_par_index = uint(users[curr_par_index].par);
            }
            last_txn_success = 1;
            emit Deposit(msg.sender, 1);
        }
        
    }
    
    function closeAccount(uint _uid1, uint _uid2) public {
        uint index1 = uid_to_index[_uid1];
        uint index2 = uid_to_index[_uid2];

        removeNei(index1, _uid2);
        removeNei(index2, _uid1);
    }
}