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

    // --- Level 2: Tokenized Rewards Economy ---
    mapping(string => uint256) public agentBalances;
    uint256 public discoveryReward = 5;
    uint256 public rescueReward = 15;

    event LogCreated(uint256 timestamp, string agentId, string actionType, string location);
    event RewardIssued(string agentId, uint256 amount, string reason);

    function logDisasterEvent(string memory _agentId, string memory _actionType, string memory _location) public {
        EventLog memory newLog = EventLog({
            timestamp: block.timestamp,
            agentId: _agentId,
            actionType: _actionType,
            location: _location
        });
        
        auditLogs.push(newLog);
        emit LogCreated(block.timestamp, _agentId, _actionType, _location);

        // --- Level 2: Automated Bounty Issuance ---
        if (keccak256(abi.encodePacked(_actionType)) == keccak256(abi.encodePacked("SURVIVOR_DISCOVERED"))) {
            agentBalances[_agentId] += discoveryReward;
            emit RewardIssued(_agentId, discoveryReward, "Discovered Survivor");
        } else if (keccak256(abi.encodePacked(_actionType)) == keccak256(abi.encodePacked("SURVIVOR_RESCUED"))) {
            agentBalances[_agentId] += rescueReward;
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
