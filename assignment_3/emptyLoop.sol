pragma solidity ^0.4.24;

contract Sorter {
    uint public loopVar;
    event Deposit(address indexed _from, uint _value);
    

    constructor(uint initVal) public {
        loopVar = initVal*5;
    }

    function runLoop(uint _p) view public{

    	uint a = _p;
	    for (uint i = 0; i < loopVar; i++) 
	    { 
	        a++;
	    } 
        emit Deposit(msg.sender, 0);
    }

}