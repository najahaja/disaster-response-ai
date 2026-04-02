// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DisasterAudit {
    struct EventLog {
        uint256 timestamp;
        string agentId;
        string actionType;
        string location;
    }

    EventLog[] public auditLogs;

    // --- Level 2 & 3: Decentralized DAO Economy ---
    mapping(string => uint256) public agentBalances;
    mapping(string => address payable) public agentWallets; // Map agent string IDs to real ETH wallets
    
    // Bounties in WEI (We will pay out actual ETH/WEI)
    uint256 public discoveryReward = 0.05 ether; // 0.05 ETH reward
    uint256 public rescueReward = 0.15 ether;   // 0.15 ETH reward

    event LogCreated(uint256 timestamp, string agentId, string actionType, string location);
    event RewardIssued(string agentId, uint256 amount, string reason);
    event FundDonated(address donor, uint256 amount);
    event AgentWalletRegistered(string agentId, address wallet);

    // --- Level 3: DAO Sponsorship / Relief Fund ---
    // Anyone can donate real ETH to the public rescue fund
    function sponsorFund() public payable {
        emit FundDonated(msg.sender, msg.value);
    }

    // Register a real Ethereum wallet for an AI Agent
    function registerAgentWallet(string memory _agentId, address payable _wallet) public {
        agentWallets[_agentId] = _wallet;
        emit AgentWalletRegistered(_agentId, _wallet);
    }
    
    // Get the total ETH currently in the relief fund pot
    function getFundBalance() public view returns (uint256) {
        return address(this).balance;
    }

    function logDisasterEvent(string memory _agentId, string memory _actionType, string memory _location) public {
        EventLog memory newLog = EventLog({
            timestamp: block.timestamp,
            agentId: _agentId,
            actionType: _actionType,
            location: _location
        });
        
        auditLogs.push(newLog);
        emit LogCreated(block.timestamp, _agentId, _actionType, _location);

        // --- Level 3: Automated Real ETH Bounty Payouts ---
        if (keccak256(abi.encodePacked(_actionType)) == keccak256(abi.encodePacked("SURVIVOR_DISCOVERED"))) {
            agentBalances[_agentId] += discoveryReward;
            
            // If the Smart Contract has enough ETH and the agent runs a wallet, send real ETH!
            if (address(this).balance >= discoveryReward && agentWallets[_agentId] != address(0)) {
                agentWallets[_agentId].transfer(discoveryReward);
            }
            emit RewardIssued(_agentId, discoveryReward, "Discovered Survivor");
            
        } else if (keccak256(abi.encodePacked(_actionType)) == keccak256(abi.encodePacked("SURVIVOR_RESCUED"))) {
            agentBalances[_agentId] += rescueReward;
            
            // If the Smart Contract has enough ETH and the agent runs a wallet, send real ETH!
            if (address(this).balance >= rescueReward && agentWallets[_agentId] != address(0)) {
                agentWallets[_agentId].transfer(rescueReward);
            }
            emit RewardIssued(_agentId, rescueReward, "Rescued Survivor");
        }
    }

    function getLogsCount() public view returns (uint256) {
        return auditLogs.length;
    }

    function getLog(uint256 index) public view returns (uint256, string memory, string memory, string memory) {
        require(index < auditLogs.length, "Log index out of bounds");
        EventLog memory l = auditLogs[index];
        return (l.timestamp, l.agentId, l.actionType, l.location);
    }

    // --- Level 2: Balance check ---
    function getAgentBalance(string memory _agentId) public view returns (uint256) {
        return agentBalances[_agentId];
    }
}
